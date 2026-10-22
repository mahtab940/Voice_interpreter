
import sounddevice as sd
import numpy as np
from scipy.signal import resample_poly
from faster_whisper import WhisperModel

# Microphone settings
DEVICE = 7
SAMPLE_RATE = 44100
WHISPER_RATE = 16000
DURATION = 5

print("Loading Whisper model...")

model = WhisperModel(
    "base.en",
    device="cpu",
    compute_type="int8"
)

print("\nSpeak English now...")
print("Recording for 5 seconds...")

audio = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="float32",
    device=DEVICE
)

sd.wait()

print("Recording completed.")
print("Converting speech to text...")

# Convert audio to 16 kHz for Whisper
audio = audio.flatten()

audio_16k = resample_poly(
    audio,
    WHISPER_RATE,
    SAMPLE_RATE
).astype(np.float32)

# Speech recognition
segments, info = model.transcribe(
    audio_16k,
    language="en",
    vad_filter=True,
    beam_size=5
)

text = " ".join(
    segment.text.strip()
    for segment in segments
)

print("\n==========================")
print("Recognized Speech:", text)
print("==========================")
