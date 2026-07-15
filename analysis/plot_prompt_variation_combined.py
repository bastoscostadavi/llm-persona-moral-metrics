#!/usr/bin/env python3
"""Consolidated prompt-paraphrase figures: both models on one shared x-axis.

Two figures (one for robustness R, one for susceptibility S). Within each figure
both models are drawn on the *same* x-axis, grouped by model (5 canonical
variants v0..v4 per model). Each model's v0 baseline is dashed and its
across-variant spread shaded, so the tight within-model clustering is visible
while the shared axis makes the between-model comparison direct.

Writes ``prompt_variation_robustness_combined`` /
``prompt_variation_susceptibility_combined`` as PDF (and PNG to results/plots).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RESULTS_DIR = REPO_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
PAPER_FIGURES_DIR = REPO_ROOT / "paper" / "figures"

MODELS = ["gpt-4.1-mini", "claude-haiku-4-5"]
GAP = 1.6  # blank rows between the two model groups


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PLOTS_DIR)
    parser.add_argument("--copy-to-paper", action="store_true",
                        help="Also copy the PDFs into paper/figures/.")
    return parser.parse_args()


def make_figure(metric: str, uncertainty: str, xlabel: str, output_stem: str,
                output_dir: Path, copy_to_paper: bool) -> None:
    from model_registry import label_for_model, model_output_stem, plot_color_for_model

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    trans = blended_transform_factory(ax.transAxes, ax.transData)

    cursor = 0.0
    max_val = 0.0
    for model in MODELS:
        stem = model_output_stem(model)
        frame = pd.read_csv(RESULTS_DIR / f"prompt_variation_metrics_{stem}.csv").reset_index(drop=True)
        color = plot_color_for_model(model)

        ys = [cursor + i for i in range(len(frame))]
        values = frame[metric].to_numpy(dtype=float)
        errors = frame[uncertainty].to_numpy(dtype=float)
        max_val = max(max_val, float((values + errors).max()))

        baseline = float(frame.loc[frame["variant"] == "v0", metric].iloc[0])
        lo, hi = float(values.min()), float(values.max())
        spread_pct = 100.0 * (hi - lo) / baseline if baseline else 0.0

        y_top, y_bot = min(ys) - 0.5, max(ys) + 0.5
        ax.fill_betweenx([y_top, y_bot], lo, hi, color="#BFBFBF", alpha=0.20, zorder=0)
        ax.plot([baseline, baseline], [y_top, y_bot], color="#333333",
                linestyle="--", linewidth=1.1, zorder=1)

        ax.barh(ys, values, xerr=errors, color=color, alpha=0.92, zorder=2,
                error_kw={"elinewidth": 1.0, "capsize": 3.0, "ecolor": "#333333"})
        for y, (val, err) in zip(ys, zip(values, errors)):
            ax.text(val + err + max_val * 0.012, y, f"{val:.3f}",
                    va="center", ha="left", fontsize=8.5, color="#222222")

        # bold model name + spread, to the left of the v-labels
        center = sum(ys) / len(ys)
        ax.text(-0.14, center, f"{label_for_model(model)}\n(spread {spread_pct:.0f}%)",
                transform=trans, rotation=90, ha="center", va="center",
                fontsize=11, fontweight="bold")
        cursor = max(ys) + 1 + GAP

    n_rows = int(cursor)
    yticks, yticklabels = [], []
    c = 0.0
    for model in MODELS:
        stem = model_output_stem(model)
        m = len(pd.read_csv(RESULTS_DIR / f"prompt_variation_metrics_{stem}.csv"))
        for i in range(m):
            yticks.append(c + i)
            yticklabels.append(f"v{i}")
        c = c + m - 1 + 1 + GAP
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=10)
    ax.set_ylim(-0.8, n_rows - GAP + 0.3)
    ax.invert_yaxis()
    ax.set_xlim(0, max_val * 1.16)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.tick_params(axis="x", labelsize=10)
    ax.grid(True, axis="x", alpha=0.25, zorder=0)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{output_stem}.png"
    pdf_path = output_dir / f"{output_stem}.pdf"
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {png_path} and {pdf_path}")
    if copy_to_paper:
        PAPER_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        (PAPER_FIGURES_DIR / pdf_path.name).write_bytes(pdf_path.read_bytes())
        print(f"Copied {PAPER_FIGURES_DIR / pdf_path.name}")


def main() -> None:
    args = parse_args()
    make_figure("robustness", "robustness_uncertainty", "Moral Robustness  R",
                "prompt_variation_robustness_combined", args.output_dir, args.copy_to_paper)
    make_figure("susceptibility", "susceptibility_uncertainty", "Moral Susceptibility  S",
                "prompt_variation_susceptibility_combined", args.output_dir, args.copy_to_paper)


if __name__ == "__main__":
    main()
