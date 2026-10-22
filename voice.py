
import csv
import sys
import time
import threading
import subprocess
from collections import deque
from pathlib import Path

import numpy as np
import sounddevice as sd
import torch

from scipy.signal import resample_poly
from faster_whisper import WhisperModel
from silero_vad import load_silero_vad

from ai_engine import ask_ai


# ============================================
# CONFIGURATION
# ============================================

MIC_DEVICE = 7
SAMPLE_RATE = 44100
VAD_RATE = 16000

VAD_FRAME_SIZE = 512
FRAMES_PER_BLOCK = 10

# 14112 input samples produce exactly
# 5120 samples at 16000 Hz
BLOCK_SIZE = 14112

SPEECH_THRESHOLD = 0.45
END_SILENCE_SECONDS = 1.2
MAX_RECORD_SECONDS = 20

PRE_ROLL_BLOCKS = 3

# Experimental voice interruption
INTERRUPT_THRESHOLD = 0.65
INTERRUPT_FRAMES = 5

VOICE_MODEL = "en_US-lessac-medium"

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

PERFORMANCE_FILE = (
    RESULTS_DIR / "performance_silero.csv"
)

AUDIO_FILE = RESULTS_DIR / "claude_response.wav"

torch.set_num_threads(1)


# ============================================
# INITIALIZE MODELS
# ============================================

print("[SYSTEM] Loading Faster-Whisper...")

whisper = WhisperModel(
    "base.en",
    device="cpu",
    compute_type="int8"
)

print("[SYSTEM] Loading Silero VAD...")

# Separate models prevent streaming-state conflicts.
record_vad = load_silero_vad()
interrupt_vad = load_silero_vad()

print("[SYSTEM] All models ready.")


# ============================================
# PERFORMANCE LOGGING
# ============================================

def save_performance(
    question,
    stt,
    ai,
    tts,
    playback,
    interrupted,
    status,
    error=""
):
    is_new = not PERFORMANCE_FILE.exists()

    with PERFORMANCE_FILE.open(
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        if is_new:
            writer.writerow([
                "timestamp",
                "question",
                "stt_seconds",
                "ai_seconds",
                "tts_generation_seconds",
                "playback_seconds",
                "processing_total",
                "interrupted",
                "status",
                "error"
            ])

        writer.writerow([
            time.strftime("%Y-%m-%d %H:%M:%S"),
            question,
            round(stt, 3),
            round(ai, 3),
            round(tts, 3),
            round(playback, 3),
            round(stt + ai + tts, 3),
            interrupted,
            status,
            error
        ])


# ============================================
# SILERO FRAME DETECTION
# ============================================

def vad_probabilities(chunk, model):
    """
    Resample microphone audio to 16 kHz.

    Return speech probability for each
    512-sample Silero frame.
    """

    audio_16k = resample_poly(
        chunk.flatten(),
        VAD_RATE,
        SAMPLE_RATE
    ).astype(np.float32)

    probabilities = []

    with torch.no_grad():

        for i in range(
            0,
            len(audio_16k) - VAD_FRAME_SIZE + 1,
            VAD_FRAME_SIZE
        ):

            frame = audio_16k[
                i:i + VAD_FRAME_SIZE
            ]

            tensor = torch.from_numpy(
                frame.copy()
            )

            probability = float(
                model(tensor, VAD_RATE).item()
            )

            probabilities.append(probability)

    return probabilities


# ============================================
# AUTOMATIC MICROPHONE RECORDING
# ============================================

def record_audio(initial_audio=None):

    print("\n[MIC] Listening automatically...")

    record_vad.reset_states()

    frames = []
    pre_roll = deque(
        maxlen=PRE_ROLL_BLOCKS
    )

    speaking = initial_audio is not None
    silence_seconds = 0.0
    frame_count = 0

    if initial_audio is not None:
        frames.append(
            np.asarray(
                initial_audio,
                dtype=np.float32
            ).reshape(-1, 1)
        )

    with sd.InputStream(
        device=MIC_DEVICE,
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=BLOCK_SIZE
    ) as stream:

        while True:

            chunk, overflowed = stream.read(
                BLOCK_SIZE
            )

            if overflowed:
                print("[MIC] Input overflow detected")

            chunk = chunk.copy()

            probabilities = vad_probabilities(
                chunk,
                record_vad
            )

            detected = any(
                p >= SPEECH_THRESHOLD
                for p in probabilities
            )

            if not speaking:

                pre_roll.append(chunk)

                if detected:
                    speaking = True

                    print(
                        "[VAD] Human speech detected!"
                    )

                    frames.extend(pre_roll)
                    pre_roll.clear()

                    frame_count = sum(
                        len(x) for x in frames
                    )

            else:
                frames.append(chunk)
                frame_count += len(chunk)

            if speaking:

                if detected:
                    silence_seconds = 0.0

                else:
                    silence_seconds += (
                        len(chunk) / SAMPLE_RATE
                    )

                if (
                    silence_seconds
                    >= END_SILENCE_SECONDS
                ):
                    print(
                        "[VAD] End of speech detected."
                    )
                    break

                if (
                    frame_count / SAMPLE_RATE
                    >= MAX_RECORD_SECONDS
                ):
                    print(
                        "[MIC] Maximum recording duration."
                    )
                    break

    if not frames:
        return None

    return np.concatenate(
        frames,
        axis=0
    ).flatten()


# ============================================
# SPEECH TO TEXT
# ============================================

def transcribe_audio(audio):

    print("\n[STT] Recognizing speech...")

    start = time.perf_counter()

    audio_16k = resample_poly(
        audio,
        VAD_RATE,
        SAMPLE_RATE
    ).astype(np.float32)

    segments, _ = whisper.transcribe(
        audio_16k,
        language="en",
        vad_filter=True,
        beam_size=5
    )

    question = " ".join(
        segment.text.strip()
        for segment in segments
    ).strip()

    latency = time.perf_counter() - start

    return question, latency


# ============================================
# PIPER SPEECH GENERATION
# ============================================

def generate_speech(text):

    print("\n[TTS] Generating speech...")

    start = time.perf_counter()

    subprocess.run(
        [
            sys.executable,
            "-m",
            "piper",
            "-m",
            VOICE_MODEL,
            "-f",
            str(AUDIO_FILE),
            "--",
            text
        ],
        check=True,
        stdout=subprocess.DEVNULL
    )

    return time.perf_counter() - start


# ============================================
# INTERRUPTION MONITOR
# ============================================

def monitor_interruption(
    stop_event,
    interrupted_event,
    audio_capture
):

    interrupt_vad.reset_states()

    recent_chunks = deque(
        maxlen=PRE_ROLL_BLOCKS
    )

    consecutive_speech = 0

    try:

        with sd.InputStream(
            device=MIC_DEVICE,
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=BLOCK_SIZE
        ) as stream:

            while not stop_event.is_set():

                chunk, overflowed = stream.read(
                    BLOCK_SIZE
                )

                if overflowed:
                    continue

                chunk = chunk.copy()
                recent_chunks.append(chunk)

                probabilities = vad_probabilities(
                    chunk,
                    interrupt_vad
                )

                for probability in probabilities:

                    if (
                        probability
                        >= INTERRUPT_THRESHOLD
                    ):
                        consecutive_speech += 1
                    else:
                        consecutive_speech = 0

                    if (
                        consecutive_speech
                        >= INTERRUPT_FRAMES
                    ):

                        audio_capture.extend(
                            recent_chunks
                        )

                        interrupted_event.set()
                        return

    except Exception as error:
        print(
            "\n[INTERRUPTION MONITOR ERROR]",
            error
        )


# ============================================
# INTERRUPTIBLE AUDIO PLAYBACK
# ============================================

def play_audio():

    print("\n[CLAUDE] Speaking...")
    print("[INTERRUPT] You can speak now.")

    stop_event = threading.Event()
    interrupted_event = threading.Event()

    captured = []

    monitor = threading.Thread(
        target=monitor_interruption,
        args=(
            stop_event,
            interrupted_event,
            captured
        ),
        daemon=True
    )

    player = None
    start = time.perf_counter()

    try:

        monitor.start()

        player = subprocess.Popen(
            [
                "ffplay",
                "-autoexit",
                "-nodisp",
                "-loglevel",
                "error",
                str(AUDIO_FILE)
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        while player.poll() is None:

            if interrupted_event.is_set():

                print(
                    "\n[INTERRUPT] User speech detected."
                )

                print(
                    "[CLAUDE] Stopping playback."
                )

                player.terminate()

                try:
                    player.wait(timeout=2)

                except subprocess.TimeoutExpired:
                    player.kill()
                    player.wait()

                break

            time.sleep(0.02)

        if (
            not interrupted_event.is_set()
            and player.returncode != 0
        ):
            raise RuntimeError(
                "FFplay exited unsuccessfully"
            )

    finally:

        stop_event.set()

        if (
            player is not None
            and player.poll() is None
        ):
            player.terminate()

            try:
                player.wait(timeout=2)
            except subprocess.TimeoutExpired:
                player.kill()
                player.wait()

        if monitor.ident is not None:
            monitor.join(timeout=2)

    playback_time = (
        time.perf_counter() - start
    )

    interrupted = interrupted_event.is_set()

    pending_audio = None

    if interrupted and captured:

        pending_audio = np.concatenate(
            captured
        ).flatten()

    return (
        interrupted,
        pending_audio,
        playback_time
    )


# ============================================
# DISPLAY PERFORMANCE
# ============================================

def show_performance(
    stt,
    ai,
    tts,
    playback,
    interrupted
):

    print("\n======================================")
    print("          PERFORMANCE REPORT")
    print("======================================")

    print(f"Whisper STT     : {stt:.3f}s")
    print(f"Claude AI       : {ai:.3f}s")
    print(f"Piper generation: {tts:.3f}s")
    print(f"Audio playback  : {playback:.3f}s")

    print("--------------------------------------")

    print(
        f"Processing total: "
        f"{stt + ai + tts:.3f}s"
    )

    print(f"Interrupted     : {interrupted}")

    print("======================================")


# ============================================
# MAIN APPLICATION
# ============================================

def main():

    print("\n======================================")
    print("         CLAUDE AI VOICE TUTOR")
    print("======================================")

    print("AI Model       : Anthropic Claude")
    print("Speech to Text : Faster-Whisper")
    print("Text to Speech : Piper TTS")
    print("Voice Detector : Silero VAD")
    print("Microphone     : Device 7 (Pulse)")
    print("Interruption   : Experimental")

    print("--------------------------------------")

    print("Speak naturally.")
    print("Use headphones for interruption tests.")
    print("Say 'exit' or 'quit' to stop.")
    print("Press Ctrl+C to close.")

    print("======================================")

    pending_audio = None

    while True:

        question = ""

        stt_time = 0.0
        ai_time = 0.0
        tts_time = 0.0
        playback_time = 0.0

        interrupted = False

        try:

            # STEP 1: Listen
            audio = record_audio(
                pending_audio
            )

            pending_audio = None

            if audio is None:
                continue

            # STEP 2: Convert voice to text
            question, stt_time = transcribe_audio(
                audio
            )

            print(f"\n[YOU] {question}")

            if not question:
                print(
                    "[STT] No speech recognized."
                )
                continue

            if question.lower().strip(" .!?") in (
                "exit",
                "quit",
                "stop",
                "goodbye"
            ):
                print("\n[SYSTEM] Goodbye!")
                break

            # STEP 3: Claude AI
            print("\n[CLAUDE] Thinking...")

            answer, ai_time = ask_ai(question)

            print(f"\n[CLAUDE] {answer}")

            # STEP 4: Generate speech
            tts_time = generate_speech(
                answer
            )

            # STEP 5: Play and monitor
            (
                interrupted,
                pending_audio,
                playback_time
            ) = play_audio()

            # STEP 6: Display results
            show_performance(
                stt_time,
                ai_time,
                tts_time,
                playback_time,
                interrupted
            )

            # STEP 7: Log interaction
            save_performance(
                question,
                stt_time,
                ai_time,
                tts_time,
                playback_time,
                interrupted,
                "success"
            )

        except KeyboardInterrupt:

            print(
                "\n[SYSTEM] Stopped by user."
            )
            break

        except Exception as error:

            print("\n[ERROR]", error)
            print("[SYSTEM] Recovering...")

            pending_audio = None

            save_performance(
                question,
                stt_time,
                ai_time,
                tts_time,
                playback_time,
                interrupted,
                "failed",
                str(error)
            )

            time.sleep(1)

    print("\n[SYSTEM] Application closed.")


# ============================================
# START APPLICATION
# ============================================

if __name__ == "__main__":

    try:
        main()

    except KeyboardInterrupt:
        print(
            "\n[SYSTEM] Application stopped."
        )
