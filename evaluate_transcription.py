
import csv
from collections import defaultdict
from jiwer import wer

data = defaultdict(lambda: {"expected": [], "recognized": []})

with open("results/transcription_tests.csv", newline="") as file:
    for row in csv.DictReader(file):
        expected = row["expected"].strip().lower()
        recognized = row["recognized"].strip().lower()
        environment = row["environment"].strip().lower()

        if not expected or not recognized:
            continue

        data[environment]["expected"].append(expected)
        data[environment]["recognized"].append(recognized)

for environment, values in data.items():
    score = wer(
        values["expected"],
        values["recognized"]
    )

    print(
        f"{environment}: "
        f"WER = {score * 100:.2f}% "
        f"({len(values['expected'])} tests)"
    )
