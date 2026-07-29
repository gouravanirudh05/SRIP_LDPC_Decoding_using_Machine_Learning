"""Extract training and final-test metrics from an ECCT logging.txt file.

Usage:
    python analyze_ecct_log.py path/to/logging.txt

The CSV files are written beside the supplied log file.
"""

from __future__ import annotations

import csv
import math
import re
import sys
from pathlib import Path


PAPER_NEG_LOG_BER_N6_D128 = {4: 6.13, 5: 8.71, 6: 12.10}


def parse_float(value: str) -> float:
    return float(value.replace("−", "-"))


def parse_metric_line(line: str, metric: str) -> dict[int, float]:
    match = re.search(rf"Test {metric}\s+(.+)$", line)
    if not match:
        return {}
    return {
        int(snr): parse_float(value)
        for snr, value in re.findall(r"(\d+)\s*:\s*([0-9.eE+−-]+)", match.group(1))
    }


def main(log_path: Path) -> None:
    lines = log_path.read_text(errors="replace").splitlines()
    output_dir = log_path.parent

    train_rows: dict[int, dict[str, float | int]] = {}
    train_pattern = re.compile(
        r"Training epoch (\d+), Batch 1000/1000: LR=([0-9.eE+-]+), "
        r"Loss=([0-9.eE+-]+) BER=([0-9.eE+-]+) FER=([0-9.eE+-]+)"
    )
    time_pattern = re.compile(r"Epoch (\d+) Train Time ([0-9.eE+-]+)s")

    for line in lines:
        match = train_pattern.search(line)
        if match:
            epoch = int(match.group(1))
            train_rows[epoch] = {
                "epoch": epoch,
                "learning_rate": parse_float(match.group(2)),
                "loss": parse_float(match.group(3)),
                "ber": parse_float(match.group(4)),
                "fer": parse_float(match.group(5)),
            }
        match = time_pattern.search(line)
        if match and int(match.group(1)) in train_rows:
            train_rows[int(match.group(1))]["train_time_seconds"] = parse_float(match.group(2))

    training_csv = output_dir / "training_metrics.csv"
    best_loss = float("inf")
    for epoch in sorted(train_rows):
        loss = float(train_rows[epoch]["loss"])
        train_rows[epoch]["is_best_loss"] = loss < best_loss
        best_loss = min(best_loss, loss)
    with training_csv.open("w", newline="") as handle:
        fields = [
            "epoch", "learning_rate", "loss", "ber", "fer",
            "train_time_seconds", "is_best_loss",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for epoch in sorted(train_rows):
            row = train_rows[epoch]
            writer.writerow({field: row.get(field, "") for field in fields})

    test_metrics = {metric: {} for metric in ("Loss", "FER", "BER")}
    sample_counts: list[float] = []
    for line in lines:
        for metric in test_metrics:
            parsed = parse_metric_line(line, metric)
            if parsed:
                test_metrics[metric] = parsed
        match = re.search(r"# of testing samples:\s*\[([^]]+)\]", line)
        if match:
            sample_counts = [parse_float(value) for value in match.group(1).split(",")]

    test_csv = output_dir / "test_metrics.csv"
    fields = [
        "ebn0_db", "loss", "fer", "ber", "negative_log_ber",
        "testing_samples", "observed_frame_errors", "paper_negative_log_ber",
        "difference_vs_paper", "ber_ratio_vs_paper",
    ]
    with test_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, ebn0 in enumerate(sorted(test_metrics["BER"])):
            ber = test_metrics["BER"][ebn0]
            paper_value = PAPER_NEG_LOG_BER_N6_D128[ebn0]
            sample_count = sample_counts[index] if index < len(sample_counts) else ""
            writer.writerow({
                "ebn0_db": ebn0,
                "loss": test_metrics["Loss"].get(ebn0, ""),
                "fer": test_metrics["FER"].get(ebn0, ""),
                "ber": ber,
                "negative_log_ber": -math.log(ber),
                "testing_samples": sample_count,
                "observed_frame_errors": (
                    test_metrics["FER"].get(ebn0, 0) * sample_count
                    if sample_count != "" else ""
                ),
                "paper_negative_log_ber": paper_value,
                "difference_vs_paper": -math.log(ber) - paper_value,
                "ber_ratio_vs_paper": ber / math.exp(-paper_value),
            })

    print(f"Wrote {training_csv}")
    print(f"Wrote {test_csv}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python analyze_ecct_log.py path/to/logging.txt")
    main(Path(sys.argv[1]))
