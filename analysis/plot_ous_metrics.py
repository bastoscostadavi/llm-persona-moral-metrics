#!/usr/bin/env python3
"""Benchmark bar plots for OUS moral robustness and susceptibility.

Mirrors ``analysis/plot_metrics.py`` (same horizontal-bar style as the paper's
``robustness_temp01.pdf`` / ``susceptibility_temp01.pdf``) but reads the OUS
metrics and writes ``ous_robustness_temp01`` / ``ous_susceptibility_temp01`` as
both PNG and PDF.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model_registry import label_for_model, plot_color_for_model

RESULTS_DIR = REPO_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
DEFAULT_METRICS_CSV = RESULTS_DIR / "ous_persona_moral_metrics.csv"
PAPER_FIGURES_DIR = REPO_ROOT / "paper" / "figures"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Temperature slice to plot (default: 0.1).")
    parser.add_argument("--output-dir", type=Path, default=PLOTS_DIR)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS_CSV)
    parser.add_argument("--copy-to-paper", action="store_true",
                        help="Also copy the PDFs into paper/figures/.")
    return parser.parse_args()


def _plot_bar(frame: pd.DataFrame, metric: str, uncertainty: str, xlabel: str,
              output_stem: Path, copy_to_paper: bool) -> None:
    plot_frame = frame.sort_values(metric, ascending=False).reset_index(drop=True)
    labels = [label_for_model(model) for model in plot_frame["model"]]
    colors = [plot_color_for_model(model) for model in plot_frame["model"]]

    fig_height = max(2.6, 0.55 * len(plot_frame) + 1.2)
    fig, ax = plt.subplots(figsize=(8.5, fig_height))
    ax.barh(
        labels,
        plot_frame[metric],
        xerr=plot_frame[uncertainty],
        color=colors,
        alpha=0.9,
        error_kw={"elinewidth": 1.0, "capsize": 2.5, "ecolor": "#333333"},
    )
    ax.invert_yaxis()
    ax.set_xlabel(xlabel, fontsize=13)
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {png_path} and {pdf_path}")

    if copy_to_paper:
        PAPER_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        paper_pdf = PAPER_FIGURES_DIR / pdf_path.name
        paper_pdf.write_bytes(pdf_path.read_bytes())
        print(f"Copied {paper_pdf}")


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.metrics)
    frame = frame[frame["temperature"].round(10) == round(args.temperature, 10)].copy()
    if frame.empty:
        raise RuntimeError(f"No OUS metrics found at temperature {args.temperature} in {args.metrics}")

    temp_tag = f"{int(round(args.temperature * 10)):02d}"
    _plot_bar(frame, "robustness", "robustness_uncertainty", "Moral Robustness",
              args.output_dir / f"ous_robustness_temp{temp_tag}", args.copy_to_paper)
    _plot_bar(frame, "susceptibility", "susceptibility_uncertainty", "Moral Susceptibility",
              args.output_dir / f"ous_susceptibility_temp{temp_tag}", args.copy_to_paper)
    print(f"Wrote OUS benchmark plots for T={args.temperature} to {args.output_dir}")


if __name__ == "__main__":
    main()
