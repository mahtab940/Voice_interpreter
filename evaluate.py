
import csv
import statistics
from pathlib import Path
from jiwer import wer

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

PERFORMANCE_FILE = (
    RESULTS_DIR / "performance_advanced.csv"
)

TRANSCRIPTION_FILE = (
    RESULTS_DIR / "transcription_tests.csv"
)

REPORT_FILE = RESULTS_DIR / "evaluation_report.txt"


def evaluate_performance():
    if not PERFORMANCE_FILE.exists():
        return "No performance data available."

    with PERFORMANCE_FILE.open(
        newline="", encoding="utf-8"
    ) as f:
        rows = list(csv.DictReader(f))

    successful = [
        row for row in rows
        if row["status"] == "success"
    ]

    failed = [
        row for row in rows
        if row["status"] == "failed"
    ]

    lines = [
        "PERFORMANCE EVALUATION",
        "=" * 40,
        f"Total interactions: {len(rows)}",
        f"Successful: {len(successful)}",
        f"Failed: {len(failed)}",
    ]

    if successful:
        metrics = [
            ("STT", "stt_latency"),
            ("Claude", "ai_latency"),
            ("TTS", "tts_generation_latency"),
            ("Processing total", "processing_total")
        ]

        for name, key in metrics:
            values = [
                float(row[key])
                for row in successful
                if row.get(key)
            ]

            if values:
                lines.append(
                    f"{name} average: "
                    f"{statistics.mean(values):.3f}s"
                )
                lines.append(
                    f"{name} maximum: "
                    f"{max(values):.3f}s"
                )

        interrupted = sum(
            str(row.get("interrupted", "")).lower()
            == "true"
            for row in successful
        )

        lines.append(
            f"Detected interruptions: {interrupted}"
        )

    return "\n".join(lines)


def evaluate_transcriptions():
    if not TRANSCRIPTION_FILE.exists():
        return "No transcription test data available."

    with TRANSCRIPTION_FILE.open(
        newline="", encoding="utf-8"
    ) as f:
        rows = list(csv.DictReader(f))

    lines = [
        "TRANSCRIPTION ACCURACY",
        "=" * 40
    ]

    scores = []

    for row in rows:
        expected = row["expected"].lower().strip()
        recognized = row["recognized"].lower().strip()

        if not expected:
            continue

        score = wer(expected, recognized)
        scores.append(score)

        lines.append(
            f"Test {len(scores)}: "
            f"WER = {score * 100:.2f}%"
        )

    if scores:
        lines.append(
            f"Average test WER: "
            f"{statistics.mean(scores) * 100:.2f}%"
        )
    else:
        lines.append("No valid test sentences found.")

    return "\n".join(lines)


def main():
    print("\nEvaluating Claude AI Voice Tutor...\n")

    performance = evaluate_performance()
    transcription = evaluate_transcriptions()

    report = (
        "CLAUDE AI VOICE TUTOR\n"
        "ASSIGNMENT 11 EVALUATION\n\n"
        + performance
        + "\n\n"
        + transcription
        + "\n"
    )

    print(report)

    REPORT_FILE.write_text(
        report,
        encoding="utf-8"
    )

    print(f"Report saved: {REPORT_FILE}")


if __name__ == "__main__":
    main()
