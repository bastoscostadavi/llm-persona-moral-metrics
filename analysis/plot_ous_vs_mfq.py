#!/usr/bin/env python3
"""Side-by-side MFQ vs OUS benchmark bars for the shared model set.

For each metric (robustness, susceptibility) produces a two-panel figure:
left = MFQ, right = OUS, same four models on the same rows (ordered by the MFQ
value) so the reader can directly read off whether the ranking reproduces on
the second instrument. Each panel keeps its own x-axis, since the two
instruments are on different absolute scales; the comparison of interest is the
ordering / pattern, not absolute magnitude.

Outputs (PNG + PDF): ous_vs_mfq_robustness_temp01, ous_vs_mfq_susceptibility_temp01
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
MFQ_METRICS_CSV = RESULTS_DIR / "persona_moral_metrics.csv"
OUS_METRICS_CSV = RESULTS_DIR / "ous_persona_moral_metrics.csv"
PAPER_FIGURES_DIR = REPO_ROOT / "paper" / "figures"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--output-dir", type=Path, default=PLOTS_DIR)
    parser.add_argument("--mfq-metrics", type=Path, default=MFQ_METRICS_CSV)
    parser.add_argument("--ous-metrics", type=Path, default=OUS_METRICS_CSV)
    parser.add_argument("--copy-to-paper", action="store_true",
                        help="Also copy the PDFs into paper/figures/.")
    return parser.parse_args()


def _slice(csv_path: Path, temperature: float) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    frame = frame[frame["temperature"].round(10) == round(temperature, 10)].copy()
    return frame.set_index("model")


def _panel(ax: plt.Axes, frame: pd.DataFrame, order: list[str], metric: str,
           uncertainty: str, title: str, show_ylabels: bool) -> None:
    # Place order[0] at the top row without invert_yaxis (which would be applied
    # twice on a shared y-axis and cancel out).
    n = len(order)
    y = [n - 1 - i for i in range(n)]
    values = [frame.loc[m, metric] for m in order]
    errs = [frame.loc[m, uncertainty] for m in order]
    colors = [plot_color_for_model(m) for m in order]
    ax.barh(y, values, xerr=errs, color=colors, alpha=0.9,
            error_kw={"elinewidth": 1.0, "capsize": 2.5, "ecolor": "#333333"})
    ax.set_yticks(y)
    if show_ylabels:
        ax.set_yticklabels([label_for_model(m) for m in order], fontsize=12)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_title(title, fontsize=13, pad=6)
    ax.tick_params(axis="x", labelsize=11)
    ax.grid(True, axis="x", alpha=0.25)


def _comparison_figure(mfq: pd.DataFrame, ous: pd.DataFrame, metric: str,
                       uncertainty: str, xlabel: str, output_stem: Path,
                       copy_to_paper: bool) -> None:
    # Shared row order: sort by the MFQ value (descending) and keep it identical
    # across both panels so ranking changes are visible as row reorderings.
    order = list(mfq[metric].sort_values(ascending=False).index)

    fig_height = max(3.0, 0.62 * len(order) + 1.0)
    fig, (ax_mfq, ax_ous) = plt.subplots(
        1, 2, figsize=(11.0, fig_height), sharey=True,
        gridspec_kw={"wspace": 0.08}, constrained_layout=True,
    )
    _panel(ax_mfq, mfq, order, metric, uncertainty, "MFQ", show_ylabels=True)
    _panel(ax_ous, ous, order, metric, uncertainty, "OUS", show_ylabels=False)
    fig.supxlabel(xlabel, fontsize=13)

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
    mfq = _slice(args.mfq_metrics, args.temperature)
    ous = _slice(args.ous_metrics, args.temperature)
    # Compare every model that has both an MFQ and an OUS measurement.
    shared = [m for m in ous.index if m in mfq.index]
    if not shared:
        raise RuntimeError(f"No models with both MFQ and OUS metrics at T={args.temperature}")
    mfq = mfq.loc[shared]
    ous = ous.loc[shared]
    print(f"Comparing {len(shared)} shared models: {', '.join(shared)}")

    temp_tag = f"{int(round(args.temperature * 10)):02d}"
    _comparison_figure(mfq, ous, "robustness", "robustness_uncertainty",
                       "Moral Robustness", args.output_dir / f"ous_vs_mfq_robustness_temp{temp_tag}",
                       args.copy_to_paper)
    _comparison_figure(mfq, ous, "susceptibility", "susceptibility_uncertainty",
                       "Moral Susceptibility", args.output_dir / f"ous_vs_mfq_susceptibility_temp{temp_tag}",
                       args.copy_to_paper)
    print(f"Wrote MFQ-vs-OUS comparison plots for T={args.temperature} to {args.output_dir}")


if __name__ == "__main__":
    main()
