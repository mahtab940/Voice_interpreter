
import csv
import statistics
from pathlib import Path

path = Path("results/performance_silero.csv")

with path.open(newline="") as f:
    rows = list(csv.DictReader(f))

successful = [
    row for row in rows
    if row.get("status") == "success"
]

metrics = {
    "Whisper STT": "stt_seconds",
    "Claude AI": "ai_seconds",
    "Piper TTS": "tts_generation_seconds",
    "Processing total": "processing_total",
    "Playback duration": "playback_seconds"
}

print("\nASSIGNMENT 11 PERFORMANCE SUMMARY")
print("=" * 45)

print("Total interactions:", len(rows))
print("Successful:", len(successful))
print("Failed:", len(rows) - len(successful))

for name, column in metrics.items():

    values = [
        float(row[column])
        for row in successful
        if row.get(column)
    ]

    if not values:
        continue

    print(f"\n{name}")
    print(f"Average: {statistics.mean(values):.3f}s")
    print(f"Minimum: {min(values):.3f}s")
    print(f"Maximum: {max(values):.3f}s")

interruptions = sum(
    row.get("interrupted", "").lower() == "true"
    for row in successful
)

print("\nDetected interruptions:", interruptions)
print("=" * 45)
