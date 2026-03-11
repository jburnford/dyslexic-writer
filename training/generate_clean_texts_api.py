#!/usr/bin/env python3
"""
Generate clean source texts via Gemini API (no error injection).

Step 1 of the 3-step pipeline:
  1. Generate clean texts (this script)
  2. Inject errors with Python engine
  3. Augment with Gemini

Usage:
    python generate_clean_texts_api.py --needed 5600
    python generate_clean_texts_api.py --needed 5600 --restart
"""

import asyncio
import json
import os
import random
import re
import sys
import time
import argparse
from pathlib import Path

API_KEY = os.environ.get("GOOGLE_API_KEY", "")
MODEL = "gemini-2.0-flash"

_DIR = Path(__file__).parent
SEEDS_FILE = _DIR / "topic_seeds.json"
OUTPUT_FILE = _DIR / "clean_texts.jsonl"

TEXTS_PER_REQUEST = 10
MAX_CONCURRENT = 15  # parallel requests to Gemini

SYSTEM_PROMPT = """You are a creative writing simulator. Write short texts exactly as real children would write them — natural voice, age-appropriate vocabulary, imperfect grammar is fine.

Output a JSON array of objects, each with:
- "text": the writing sample (80-150 words)
- "index": the story number (1-based)

Rules:
- Write naturally for the specified age — don't be too polished or literary
- Include imperfect grammar where natural ("me and Jake went", "we was", "she don't")
- Use the child's authentic voice — excited for young, opinionated for teens
- Include specific details, names, and places
- Output ONLY the JSON array, no other text"""


def build_prompt(topics: list[dict]) -> str:
    lines = [f"Write {len(topics)} short texts:\n"]
    for i, t in enumerate(topics, 1):
        lines.append(f"{i}. {t['topic']}")
    lines.append("\nOutput ONLY a JSON array. No other text.")
    return "\n".join(lines)


async def process_batch(client, semaphore, batch_idx, topics, results_queue):
    """Process a single batch with concurrency limiting."""
    async with semaphore:
        prompt = build_prompt(topics)

        for attempt in range(3):
            try:
                response = await client.aio.models.generate_content(
                    model=MODEL,
                    contents=prompt,
                    config={
                        "system_instruction": SYSTEM_PROMPT,
                        "temperature": 0.9,
                        "max_output_tokens": 8192,
                        "response_mime_type": "application/json",
                    },
                )
                text = response.text or ""
                break
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower() or "RESOURCE_EXHAUSTED" in str(e):
                    wait = 5 * (attempt + 1)
                    await asyncio.sleep(wait)
                elif attempt == 2:
                    print(f"  Batch {batch_idx}: FAILED after 3 attempts: {e}", flush=True)
                    await results_queue.put((batch_idx, []))
                    return
                else:
                    await asyncio.sleep(2)
        else:
            await results_queue.put((batch_idx, []))
            return

        if not text:
            await results_queue.put((batch_idx, []))
            return

        try:
            text = re.sub(r"^```(?:json)?\s*", "", text.strip())
            text = re.sub(r"\s*```$", "", text.strip())
            stories = json.loads(text)
        except json.JSONDecodeError:
            await results_queue.put((batch_idx, []))
            return

        if not isinstance(stories, list):
            await results_queue.put((batch_idx, []))
            return

        batch_texts = []
        for i, story in enumerate(stories):
            if not isinstance(story, dict):
                continue
            t = story.get("text", "").strip()
            if not t or len(t.split()) < 40:
                continue
            topic = topics[i] if i < len(topics) else {}
            age = topic.get("age", 10)
            band = "young" if age <= 9 else "middle" if age <= 12 else "teen"
            batch_texts.append({
                "text": t,
                "genre": topic.get("genre", "unknown"),
                "age_band": band,
                "age": age,
            })

        await results_queue.put((batch_idx, batch_texts))


async def writer_task(output_path, results_queue, total_batches):
    """Write results as they come in, report progress."""
    total_texts = 0
    done = 0
    errors = 0
    start = time.time()

    with open(output_path, "a") as f:
        while done < total_batches:
            batch_idx, texts = await results_queue.get()
            done += 1
            if texts:
                for t in texts:
                    f.write(json.dumps(t) + "\n")
                f.flush()
                total_texts += len(texts)
            else:
                errors += 1

            if done % 20 == 0 or done == total_batches:
                elapsed = time.time() - start
                rate = total_texts / elapsed if elapsed > 0 else 0
                print(f"  [{done}/{total_batches}] texts={total_texts} errors={errors} "
                      f"({rate:.1f} texts/s, {elapsed:.0f}s elapsed)", flush=True)

    return total_texts, errors


async def run(args):
    from google import genai
    client = genai.Client(api_key=API_KEY)

    with open(SEEDS_FILE) as f:
        seeds = json.load(f)

    random.seed(args.seed)

    # Build topic list
    all_topics = []
    per_band = args.needed // 3
    for band in ["young", "middle", "teen"]:
        band_seeds = seeds[band]
        band_topics = []
        while len(band_topics) < per_band:
            shuffled = band_seeds.copy()
            random.shuffle(shuffled)
            band_topics.extend(shuffled)
        all_topics.extend(band_topics[:per_band])

    random.shuffle(all_topics)

    # Group into batches
    batches = []
    for i in range(0, len(all_topics), TEXTS_PER_REQUEST):
        batches.append(all_topics[i:i + TEXTS_PER_REQUEST])

    # Resume support: count existing texts
    existing_count = 0
    if OUTPUT_FILE.exists() and not args.restart:
        with open(OUTPUT_FILE) as f:
            existing_count = sum(1 for line in f if line.strip())
        if existing_count > 0:
            # Skip batches we've already done (approx)
            skip = existing_count // TEXTS_PER_REQUEST
            batches = batches[skip:]
            print(f"Resuming: {existing_count} texts exist, skipping ~{skip} batches", flush=True)

    if args.restart and OUTPUT_FILE.exists():
        OUTPUT_FILE.unlink()

    print(f"Generating {len(batches)} batches ({len(batches) * TEXTS_PER_REQUEST} target texts) "
          f"with {MAX_CONCURRENT} concurrent requests", flush=True)

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    results_queue = asyncio.Queue()

    # Launch writer
    writer = asyncio.create_task(writer_task(OUTPUT_FILE, results_queue, len(batches)))

    # Launch all batch tasks
    tasks = []
    for i, batch_topics in enumerate(batches):
        task = asyncio.create_task(process_batch(client, semaphore, i, batch_topics, results_queue))
        tasks.append(task)

    # Wait for all API calls to finish
    await asyncio.gather(*tasks)

    # Wait for writer to finish
    total_texts, errors = await writer

    # Count final total
    with open(OUTPUT_FILE) as f:
        final = sum(1 for line in f if line.strip())
    print(f"\nDone! {final} total texts in {OUTPUT_FILE} ({errors} batch errors)", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--needed", type=int, default=5600)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
