"""
Flask API server for the Dyslexic Writer tool.
Provides /correct and /speak endpoints for the React frontend.

Usage:
    python server.py                          # passthrough mode (TTS only, no model)
    python server.py --backend ollama         # use Ollama with local model
    python server.py --backend transformers --model-path /path/to/model

Endpoints:
    POST /correct   {"text": "..."} -> {"original", "corrected", "changed", "changes"}
    POST /speak     {"text": "..."} -> audio/wav
    GET  /health    -> {"status": "ok", "backend": "..."}
"""

import argparse
import io
import sys
import wave

import numpy as np
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

from corrector import create_corrector, Correction
from tts import synthesize, VOICE

app = Flask(__name__)
CORS(app)

# Global corrector instance (set in main)
_corrector = None
_backend_name = "passthrough"
_voice = VOICE


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "backend": _backend_name, "voice": _voice})


@app.route("/correct", methods=["POST"])
def correct():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400

    text = data["text"].strip()
    if not text:
        return jsonify({"error": "Empty text"}), 400

    result = _corrector.correct(text)
    return jsonify({
        "original": result.original,
        "corrected": result.corrected,
        "changed": result.changed,
        "changes": result.changes,
    })


@app.route("/speak", methods=["POST"])
def speak():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400

    text = data["text"].strip()
    if not text:
        return jsonify({"error": "Empty text"}), 400

    voice = data.get("voice", _voice)

    # Synthesize to WAV
    audio, sr = synthesize(text, voice)

    # Convert to WAV bytes
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sr)
        wav_file.writeframes((audio * 32767).astype(np.int16).tobytes())

    wav_buffer.seek(0)
    return send_file(wav_buffer, mimetype="audio/wav")


@app.route("/correct-and-speak", methods=["POST"])
def correct_and_speak():
    """Correct text and return both the correction result and audio."""
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400

    text = data["text"].strip()
    if not text:
        return jsonify({"error": "Empty text"}), 400

    # Correct
    result = _corrector.correct(text)

    # Synthesize the corrected text
    voice = data.get("voice", _voice)
    audio, sr = synthesize(result.corrected, voice)

    # Encode audio as base64 WAV for JSON response
    import base64
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sr)
        wav_file.writeframes((audio * 32767).astype(np.int16).tobytes())

    audio_b64 = base64.b64encode(wav_buffer.getvalue()).decode("ascii")

    return jsonify({
        "original": result.original,
        "corrected": result.corrected,
        "changed": result.changed,
        "changes": result.changes,
        "audio_wav_base64": audio_b64,
    })


def main():
    global _corrector, _backend_name, _voice

    parser = argparse.ArgumentParser(description="Dyslexic Writer API Server")
    parser.add_argument("--backend", default="passthrough",
                        choices=["ollama", "transformers", "passthrough"],
                        help="Correction backend")
    parser.add_argument("--model", default="smollm2-1.7b-spell",
                        help="Model name (Ollama) or path (Transformers)")
    parser.add_argument("--ollama-url", default="http://localhost:11434",
                        help="Ollama API URL")
    parser.add_argument("--voice", default=VOICE, help="Piper voice name")
    parser.add_argument("--port", type=int, default=5000, help="Server port")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    args = parser.parse_args()

    _voice = args.voice
    _backend_name = args.backend

    # Create corrector
    if args.backend == "ollama":
        _corrector = create_corrector("ollama", model=args.model, base_url=args.ollama_url)
        print(f"Using Ollama backend: {args.model}")
    elif args.backend == "transformers":
        _corrector = create_corrector("transformers", model_path=args.model)
        print(f"Using Transformers backend: {args.model}")
    else:
        _corrector = create_corrector("passthrough")
        print("Using passthrough backend (no correction, TTS only)")

    print(f"Voice: {args.voice}")
    print(f"Server: http://{args.host}:{args.port}")
    print()

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
