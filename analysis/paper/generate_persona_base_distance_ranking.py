#!/usr/bin/env python3
"""Deliverable 2: enumerate the 100 personas by their distance from the baseline.

For each (model, persona) we measure how far the persona pulls the model away
from its no-persona `self` baseline:

    d_{m,p} = sqrt( sum_q (mu_{m,p,q} - mu_{m,self,q})^2 )   (Euclidean over 30 items)

where mu_{m,p,q} is the persona's mean rating (avg of 10 attempts). We then
average this distance across the 15 benchmark models to get one "average
variation from base" per persona, sort the personas by that value, and plot:

    x = persona rank (1, 2, 3, ...)
    y = average distance from baseline (mean over models; whisker = +/-1 SD)

This makes it easy to read off the group of personas that shift the models most
(left end of the plot). Persona IDs for the top movers are annotated; the full
mapping is in the CSV.

Outputs:
- paper/figures/persona_base_distance_ranking.pdf
- results/persona_base_distance_ranking.csv       (aggregated, one row per persona)
- results/persona_base_distance_by_model.csv       (per model-persona rows)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from persona_analysis_common import (  # noqa: E402
    MODELS,
    REPO_ROOT,
    load_persona_remap,
    persona_base_distances,
)

N_ANNOTATE = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figure",
        type=Path,
        default=REPO_ROOT / "paper" / "figures" / "persona_base_distance_ranking",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=REPO_ROOT / "results" / "persona_base_distance_ranking.csv",
    )
    parser.add_argument(
        "--by-model-csv",
        type=Path,
        default=REPO_ROOT / "results" / "persona_base_distance_by_model.csv",
    )
    parser.add_argument("--annotate", type=int, default=N_ANNOTATE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    by_model = pd.concat([persona_base_distances(m) for m in MODELS], ignore_index=True)

    agg = (
        by_model.groupby("persona_id")["distance_from_self"]
        .agg(avg_distance="mean", std_distance="std", n_models="count")
        .reset_index()
    )
    remap = load_persona_remap()
    agg["new_id"] = agg["persona_id"].map(remap)
    # ascending order: smallest shift (p0) on the left, largest (p99) on the right
    agg = agg.sort_values("new_id").reset_index(drop=True)

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    by_model.to_csv(args.by_model_csv, index=False)
    agg.to_csv(args.csv, index=False)
    print(f"Wrote {len(agg)} aggregated persona rows to {args.csv}")
    print("\nTop persona movers (new_id [old_id]: avg distance):")
    for _, row in agg.sort_values("avg_distance", ascending=False).head(args.annotate).iterrows():
        print(f"  persona p{int(row['new_id']):3d} [old p{int(row['persona_id'])}]: "
              f"{row['avg_distance']:.2f} +/- {row['std_distance']:.2f}  (n={int(row['n_models'])})")

    _make_figure(agg, args.figure, args.annotate)


def _make_figure(agg: pd.DataFrame, out: Path, annotate: int) -> None:
    fig, ax = plt.subplots(figsize=(11.0, 5.2))
    x = agg["new_id"].to_numpy()
    y = agg["avg_distance"].to_numpy()
    err = np.nan_to_num(agg["std_distance"].to_numpy(), nan=0.0)

    ax.errorbar(
        x, y, yerr=err, fmt="o", markersize=4.5, color="#2C5F8A",
        ecolor="#B0BEC5", elinewidth=0.9, capsize=0, alpha=0.9, zorder=3,
    )

    ax.set_xlabel("Persona id (ordered by average distance from baseline)", fontsize=12)
    ax.set_ylabel(
        r"Average distance from no-persona baseline"
        "\n"
        r"$\langle d_p\rangle_{\mathrm{models}},\ "
        r"d_{m,p}=\sqrt{\sum_q (\mu_{m,p,q}-\mu_{m,\mathrm{self},q})^2}$",
        fontsize=11,
    )
    ax.set_xlim(-1, len(agg))
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.25, zorder=0)
    ax.set_title(
        "Personas ranked by how far they move the models from baseline\n"
        "(each dot: one persona, averaged over 15 models; whisker: $\\pm$1 SD across models)",
        fontsize=11,
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote figure to {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
