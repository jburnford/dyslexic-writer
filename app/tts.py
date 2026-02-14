"""
Text-to-Speech module using Piper TTS.
Uses en_US-lessac-high voice for natural-sounding speech.

Piper downloads voice models on first use to ~/.local/share/piper_voices/
"""

import io
import subprocess
import sys
import wave
from pathlib import Path
from typing import Optional

import numpy as np

# Voice config
VOICE = "en_US-lessac-high"
SAMPLE_RATE = 22050  # Piper's default output rate


def _find_piper_binary() -> str:
    """Find the piper binary or use the Python module."""
    # Try piper in PATH
    try:
        subprocess.run(["piper", "--version"], capture_output=True, check=True)
        return "piper"
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    return ""


def synthesize(text: str, voice: str = VOICE) -> tuple[np.ndarray, int]:
    """
    Synthesize text to audio using Piper TTS.

    Returns:
        (audio_array, sample_rate) - numpy float32 array + sample rate
    """
    try:
        from piper import PiperVoice
        from piper.download import ensure_voice_exists, get_voices

        # Download voice if needed
        data_dir = Path.home() / ".local" / "share" / "piper_voices"
        data_dir.mkdir(parents=True, exist_ok=True)

        voices_info = get_voices(data_dir, update_voices=False)

        # Try to update voices list if our voice isn't found
        if voice not in voices_info:
            voices_info = get_voices(data_dir, update_voices=True)

        ensure_voice_exists(voice, data_dir, data_dir, voices_info)

        # Find the model file
        voice_dir = data_dir / voice
        onnx_files = list(voice_dir.glob("*.onnx"))
        if not onnx_files:
            raise FileNotFoundError(f"No .onnx model found in {voice_dir}")

        model_path = onnx_files[0]
        config_path = model_path.with_suffix(".onnx.json")

        # Load and synthesize
        piper_voice = PiperVoice.load(str(model_path), str(config_path))

        # Synthesize to WAV in memory
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            piper_voice.synthesize(text, wav_file)

        # Read back the WAV data
        wav_buffer.seek(0)
        with wave.open(wav_buffer, "rb") as wav_file:
            sr = wav_file.getframerate()
            frames = wav_file.readframes(wav_file.getnframes())
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

        return audio, sr

    except ImportError:
        # Fallback: use piper CLI binary
        return _synthesize_cli(text, voice)


def _synthesize_cli(text: str, voice: str = VOICE) -> tuple[np.ndarray, int]:
    """Fallback: synthesize using piper CLI."""
    piper_bin = _find_piper_binary()
    if not piper_bin:
        raise RuntimeError(
            "Piper TTS not found. Install with: pip install piper-tts\n"
            "Also install espeak-ng: brew install espeak-ng (macOS) "
            "or apt install espeak-ng (Linux)"
        )

    result = subprocess.run(
        [piper_bin, "--model", voice, "--output-raw"],
        input=text.encode("utf-8"),
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Piper failed: {result.stderr.decode()}")

    audio = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    return audio, SAMPLE_RATE


def play(audio: np.ndarray, sample_rate: int) -> None:
    """Play audio through speakers."""
    try:
        import sounddevice as sd
        sd.play(audio, sample_rate)
        sd.wait()
    except Exception:
        # Fallback: write to temp file and play with system command
        import tempfile
        import soundfile as sf

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio, sample_rate)
            if sys.platform == "darwin":
                subprocess.run(["afplay", f.name])
            elif sys.platform == "linux":
                subprocess.run(["aplay", f.name])


def speak(text: str, voice: str = VOICE) -> None:
    """Synthesize and play text. One-call convenience function."""
    audio, sr = synthesize(text, voice)
    play(audio, sr)


def save(text: str, output_path: str, voice: str = VOICE) -> None:
    """Synthesize text and save to WAV file."""
    import soundfile as sf
    audio, sr = synthesize(text, voice)
    sf.write(output_path, audio, sr)


if __name__ == "__main__":
    test_text = "Bob was a boring guy. He never went to a party he didn't need to for work."
    print(f"Speaking: {test_text}")
    speak(test_text)
    print("Done.")
