
import sys
import time
import subprocess

VOICE_MODEL = "en_US-lessac-medium"


def speak(text):
    start = time.perf_counter()

    print("\n[TTS] Generating Claude's voice...")

    subprocess.run(
        [
            sys.executable,
            "-m", "piper",
            "-m", VOICE_MODEL,
            "-f", "response.wav",
            "--", text
        ],
        check=True
    )

    generation_time = time.perf_counter() - start

    print("[TTS] Claude is speaking...")

    subprocess.run(
        [
            "ffplay",
            "-autoexit",
            "-nodisp",
            "-loglevel", "error",
            "response.wav"
        ],
        check=True
    )

    return generation_time
