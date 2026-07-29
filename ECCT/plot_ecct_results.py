"""Plot ECCT training curves and LDPC(49,24) BER comparison.

Run from the ECCT directory after generating CSV files:
    python plot_ecct_results.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


BASE = Path(__file__).resolve().parent
TRAINING_CSV = BASE / "LDPC__Code_n_49_k_24__27_07_2026_04_24_37" / "training_metrics.csv"
TEST_CSV = BASE / "LDPC__Code_n_49_k_24__27_07_2026_04_24_37" / "test_metrics.csv"
OUTPUT_DIR = BASE / "plots"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def plot_training(rows: list[dict[str, str]]) -> None:
    epochs = [int(row["epoch"]) for row in rows]
    metrics = [("loss", "Training loss"), ("ber", "Training BER"), ("fer", "Training FER")]
    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    for axis, (key, label) in zip(axes, metrics):
        axis.semilogy(epochs, [float(row[key]) for row in rows])
        axis.set_ylabel(label)
        axis.grid(True, which="both", alpha=0.3)
    axes[-1].set_xlabel("Epoch")
    fig.suptitle("ECCT training curves: LDPC(49,24)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "training_curves.png", dpi=220)
    plt.close(fig)


def plot_test(rows: list[dict[str, str]]) -> None:
    ebn0 = [float(row["ebn0_db"]) for row in rows]
    observed_ber = [float(row["ber"]) for row in rows]
    paper_ber = [10 ** 0 * __import__("math").exp(-float(row["paper_negative_log_ber"])) for row in rows]
    observed_neglog = [float(row["negative_log_ber"]) for row in rows]
    paper_neglog = [float(row["paper_negative_log_ber"]) for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].semilogy(ebn0, observed_ber, "o-", label="This run")
    axes[0].semilogy(ebn0, paper_ber, "s--", label="Paper ECCT, N=6, d=128")
    axes[0].set_xlabel("Eb/N0 (dB)")
    axes[0].set_ylabel("BER")
    axes[0].set_title("BER comparison")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend()

    axes[1].plot(ebn0, observed_neglog, "o-", label="This run")
    axes[1].plot(ebn0, paper_neglog, "s--", label="Paper ECCT, N=6, d=128")
    axes[1].set_xlabel("Eb/N0 (dB)")
    axes[1].set_ylabel("-ln(BER), higher is better")
    axes[1].set_title("Negative log-BER comparison")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "ber_comparison.png", dpi=220)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    training_rows = read_csv(TRAINING_CSV)
    test_rows = read_csv(TEST_CSV)
    plot_training(training_rows)
    plot_test(test_rows)
    print(f"Wrote plots to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
