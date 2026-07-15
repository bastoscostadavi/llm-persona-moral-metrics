#!/usr/bin/env python3
"""R and S bar plots across MFQ prompt paraphrases (Deliverable 5).

Two horizontal-bar figures (robustness, susceptibility) for a single model under
the five prompt variants defined in ``prompt_variations.py``. Bars stay in
canonical v0..v4 order (not sorted by value) so prompt-robustness is read
directly: the baseline v0 is marked with a dashed reference line and the spread
across all variants is shaded, making it obvious how little the wording moves
each metric.

Reads ``results/prompt_variation_metrics.csv`` and writes
``prompt_variation_robustness`` / ``prompt_variation_susceptibility`` as PNG+PDF.
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

RESULTS_DIR = REPO_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
PAPER_FIGURES_DIR = REPO_ROOT / "paper" / "figures"

BASELINE_COLOR = "#2C6E8F"   # v0 baseline
VARIANT_COLOR = "#74C69D"    # v1..v4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt-4.1-mini", help="Model key (default: gpt-4.1-mini).")
    parser.add_argument("--metrics", type=Path, default=None,
                        help="Metrics CSV (default: results/prompt_variation_metrics_<stem>.csv).")
    parser.add_argument("--output-dir", type=Path, default=PLOTS_DIR)
    parser.add_argument("--model-label", default=None,
                        help="Display label (default: from model registry).")
    parser.add_argument("--copy-to-paper", action="store_true",
                        help="Also copy the PDFs into paper/figures/.")
    return parser.parse_args()


def _plot_bar(frame: pd.DataFrame, metric: str, uncertainty: str, xlabel: str,
              title: str, output_stem: Path, copy_to_paper: bool) -> None:
    # Keep canonical v0..v4 order but draw top-to-bottom.
    plot_frame = frame.reset_index(drop=True)
    labels = list(plot_frame["variant_label"])
    values = plot_frame[metric].to_numpy(dtype=float)
    errors = plot_frame[uncertainty].to_numpy(dtype=float)
    colors = [BASELINE_COLOR if v == "v0" else VARIANT_COLOR for v in plot_frame["variant"]]

    baseline = float(plot_frame.loc[plot_frame["variant"] == "v0", metric].iloc[0]) \
        if (plot_frame["variant"] == "v0").any() else float(values.mean())
    lo, hi = float(values.min()), float(values.max())
    spread_pct = 100.0 * (hi - lo) / baseline if baseline else 0.0

    fig, ax = plt.subplots(figsize=(7.6, 3.4))

    # Shaded band spanning the full range across variants + dashed baseline line.
    ax.axvspan(lo, hi, color="#BFBFBF", alpha=0.20, zorder=0)
    ax.axvline(baseline, color=BASELINE_COLOR, linestyle="--", linewidth=1.2, zorder=1)

    ax.barh(labels, values, xerr=errors, color=colors, alpha=0.92, zorder=2,
            error_kw={"elinewidth": 1.0, "capsize": 3.0, "ecolor": "#333333"})
    ax.invert_yaxis()

    for y, (val, err) in enumerate(zip(values, errors)):
        ax.text(val + err + (hi - lo) * 0.04 + baseline * 0.005, y, f"{val:.3f}",
                va="center", ha="left", fontsize=9, color="#222222")

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_title(f"{title}\n(spread across prompts = {spread_pct:.1f}% of baseline)", fontsize=11)
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=11)
    ax.grid(True, axis="x", alpha=0.25, zorder=0)
    ax.margins(x=0.18)
    # Baseline label sits just above the top bar, to the right of the dashed line.
    ax.set_ylim(len(labels) - 0.5, -1.1)
    ax.annotate(f"v0 baseline = {baseline:.3f}", xy=(baseline, -0.65),
                xytext=(4, 0), textcoords="offset points", ha="left", va="center",
                fontsize=8, color=BASELINE_COLOR)
    fig.tight_layout()

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    # NB: do not use Path.with_suffix here -- model stems (e.g. "gpt-4.1-mini")
    # contain dots, which with_suffix would mangle. Append extensions literally.
    png_path = output_stem.parent / f"{output_stem.name}.png"
    pdf_path = output_stem.parent / f"{output_stem.name}.pdf"
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
    from model_registry import label_for_model, model_output_stem

    stem = model_output_stem(args.model)
    metrics_path = args.metrics or (RESULTS_DIR / f"prompt_variation_metrics_{stem}.csv")
    model_label = args.model_label or label_for_model(args.model)

    frame = pd.read_csv(metrics_path)
    if frame.empty:
        raise RuntimeError(f"No prompt-variation metrics found in {metrics_path}")

    _plot_bar(frame, "robustness", "robustness_uncertainty", "Moral Robustness  R",
              f"{model_label}: robustness under prompt paraphrase",
              args.output_dir / f"prompt_variation_robustness_{stem}", args.copy_to_paper)
    _plot_bar(frame, "susceptibility", "susceptibility_uncertainty", "Moral Susceptibility  S",
              f"{model_label}: susceptibility under prompt paraphrase",
              args.output_dir / f"prompt_variation_susceptibility_{stem}", args.copy_to_paper)
    print("Wrote prompt-variation R/S plots.")


if __name__ == "__main__":
    main()
