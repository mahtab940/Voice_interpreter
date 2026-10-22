import sounddevice as sd
import soundfile as sf

DEVICE = 7
SAMPLE_RATE = 44100
DURATION = 5

print("Recording for 5 seconds...")

audio = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="float32",
    device=DEVICE
)

sd.wait()

sf.write("microphone_test.wav", audio, SAMPLE_RATE)

print("Recording completed!")
print("Saved as microphone_test.wav")
