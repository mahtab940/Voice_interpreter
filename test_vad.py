
import numpy as np
import sounddevice as sd
import torch

from scipy.signal import resample_poly
from silero_vad import load_silero_vad

MIC_DEVICE = 7
MIC_SAMPLE_RATE = 44100
VAD_SAMPLE_RATE = 16000

FRAME_SIZE = 512
MIC_BLOCK_SIZE = 14112

SPEECH_THRESHOLD = 0.35
SILENCE_SECONDS = 1.2

torch.set_num_threads(1)

print("[SYSTEM] Loading Silero VAD...")
model = load_silero_vad()
model.reset_states()

print("[SYSTEM] Silero ready.")
print("[MIC] Listening on device 7...")
print("[TEST] Speak clearly into your microphone.")
print("[TEST] Press Ctrl+C to stop.")

speaking = False
silence_time = 0.0

try:
    with sd.InputStream(
        device=MIC_DEVICE,
        samplerate=MIC_SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=MIC_BLOCK_SIZE
    ) as stream:

        while True:
            audio, overflow = stream.read(MIC_BLOCK_SIZE)

            if overflow:
                print("[WARNING] Audio overflow")

            audio = audio.flatten()

            rms = float(np.sqrt(np.mean(audio ** 2)))
            peak = float(np.max(np.abs(audio)))

            audio_16k = resample_poly(
                audio,
                VAD_SAMPLE_RATE,
                MIC_SAMPLE_RATE
            ).astype(np.float32)

            probabilities = []

            for i in range(0, len(audio_16k), FRAME_SIZE):
                frame = audio_16k[i:i + FRAME_SIZE]

                if len(frame) != FRAME_SIZE:
                    continue

                with torch.no_grad():
                    probability = float(
                        model(
                            torch.from_numpy(frame),
                            VAD_SAMPLE_RATE
                        ).item()
                    )

                probabilities.append(probability)

                if probability >= SPEECH_THRESHOLD:
                    silence_time = 0.0

                    if not speaking:
                        speaking = True
                        print("\n[VAD] Speech detected!")

                elif speaking:
                    silence_time += FRAME_SIZE / VAD_SAMPLE_RATE

                    if silence_time >= SILENCE_SECONDS:
                        print("\n[VAD] End of speech detected.")
                        speaking = False
                        silence_time = 0.0

            max_probability = max(
                probabilities, default=0.0
            )

            print(
                f"\r[MIC] RMS: {rms:.5f} | "
                f"Peak: {peak:.5f} | "
                f"Speech probability: {max_probability:.3f}",
                end="",
                flush=True
            )

except KeyboardInterrupt:
    print("\n[SYSTEM] Test stopped.")
