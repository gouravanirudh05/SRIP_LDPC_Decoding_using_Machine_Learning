"""Create paper-style ECCT/AECCT comparison plots from comparison/data/*.csv.

Run:
    python3 comparison/build_comparison_csv.py
    python3 comparison/plot_paper_style.py

Figures are written to comparison/plots_paper_style/.
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE / "plots_paper_style"
PAPER_NEG_LN = {4: 6.13, 5: 8.71, 6: 12.10}
COLORS = {
    "ECCT (FP32)": "#1769aa",
    "AECCT phase-1 (FP32)": "#2ca25f",
    "AECCT final (ternary W / int8 A)": "#e66101",
}
MARKERS = {
    "ECCT (FP32)": "o",
    "AECCT phase-1 (FP32)": "^",
    "AECCT final (ternary W / int8 A)": "s",
}


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def grouped(rows, key):
    out = {}
    for row in rows:
        out.setdefault(row[key], []).append(row)
    for rows2 in out.values():
        rows2.sort(key=lambda r: float(r["ebno_db"]))
    return out


def style(ax):
    ax.grid(True, which="major", alpha=0.35)
    ax.grid(True, which="minor", alpha=0.15)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_metric(test_rows, metric, ylabel, filename, log_y=True):
    fig, ax = plt.subplots(figsize=(7.5, 5.3))
    for model, rows in grouped(test_rows, "model").items():
        x = [float(r["ebno_db"]) for r in rows]
        y = [float(r[metric]) for r in rows]
        ax.plot(x, y, marker=MARKERS.get(model, "o"), linewidth=2, markersize=7,
                label=model, color=COLORS.get(model, "black"))
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel(r"$E_b/N_0$ (dB)")
    ax.set_ylabel(ylabel)
    ax.set_xticks([4, 5, 6])
    style(ax)
    ax.legend()
    ax.set_title("LDPC(49,24): ECCT vs AECCT")
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=220)
    plt.close(fig)


def plot_neg_ln(test_rows):
    fig, ax = plt.subplots(figsize=(7.5, 5.3))
    for model, rows in grouped(test_rows, "model").items():
        x = [float(r["ebno_db"]) for r in rows]
        y = [float(r["neg_ln_ber"]) for r in rows]
        ax.plot(x, y, marker=MARKERS.get(model, "o"), linewidth=2, markersize=7,
                label=model, color=COLORS.get(model, "black"))
    ax.plot(list(PAPER_NEG_LN), list(PAPER_NEG_LN.values()), "k--", marker="x",
            linewidth=1.5, label="ECCT paper")
    ax.set_xlabel(r"$E_b/N_0$ (dB)")
    ax.set_ylabel(r"$-\ln(\mathrm{BER})$ (higher is better)")
    ax.set_xticks([4, 5, 6])
    style(ax)
    ax.legend()
    ax.set_title("LDPC(49,24): paper-style quality metric")
    fig.tight_layout()
    fig.savefig(OUT / "neg_ln_ber.png", dpi=220)
    plt.close(fig)


def plot_training():
    ecct = read_csv(DATA / "ecct_training.csv")
    aecct = read_csv(DATA / "aecct_training.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].plot([int(r["epoch"]) for r in ecct], [float(r["train_loss"]) for r in ecct], label="ECCT", color=COLORS["ECCT (FP32)"])
    for phase, label, color in [("1", "AECCT phase 1 (FP32)", COLORS["AECCT phase-1 (FP32)"]), ("2", "AECCT phase 2 (QAT)", COLORS["AECCT final (ternary W / int8 A)"])]:
        rows = [r for r in aecct if r["phase"] == phase]
        axes[0].plot([int(r["epoch"]) for r in rows], [float(r["train_loss"]) for r in rows], label=label, color=color)
    axes[0].set_xlabel("Logged epoch")
    axes[0].set_ylabel("Training loss")
    axes[0].set_title("Training loss by logged epoch")
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=8)

    axes[1].plot([int(r["cum_updates"]) / 1000 for r in ecct], [float(r["train_loss"]) for r in ecct], label="ECCT", color=COLORS["ECCT (FP32)"])
    for phase, label, color in [("1", "AECCT phase 1", COLORS["AECCT phase-1 (FP32)"]), ("2", "AECCT phase 2", COLORS["AECCT final (ternary W / int8 A)"])]:
        rows = [r for r in aecct if r["phase"] == phase]
        axes[1].plot([int(r["cum_updates"]) / 1000 for r in rows], [float(r["train_loss"]) for r in rows], label=label, color=color)
    axes[1].set_xlabel("Cumulative optimizer updates (thousands)")
    axes[1].set_ylabel("Training loss")
    axes[1].set_title("Training loss by update budget")
    axes[1].set_yscale("log")
    for ax in axes:
        style(ax)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "training_loss.png", dpi=220)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    tests = read_csv(DATA / "test_results.csv")
    plot_metric(tests, "ber", "BER (lower is better)", "ber_vs_ebno.png")
    plot_metric(tests, "fer", "FER (lower is better)", "fer_vs_ebno.png")
    plot_neg_ln(tests)
    plot_training()
    print(f"Wrote plots to {OUT}")


if __name__ == "__main__":
    main()
