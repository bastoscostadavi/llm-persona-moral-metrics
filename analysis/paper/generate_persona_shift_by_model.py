#!/usr/bin/env python3
"""Per-model persona shift-from-baseline, consolidated two ways.

For every model and persona we compute the shift from the no-persona baseline,
    d_{m,p} = sqrt( sum_q (mu_{m,p,q} - mu_{m,self,q})^2 ),
and plot it against the (unsorted) persona id 0..99. The 15 per-model plots are
consolidated into a single figure two ways:

- ``grid``    : 3x5 small multiples, one stem panel per model, shared y-axis.
- ``heatmap`` : one models x personas image, colour = shift. Aligned columns make
                personas that move *every* model show up as vertical stripes.

Outputs:
- paper/figures/persona_shift_by_model_grid.pdf
- paper/figures/persona_shift_by_model_heatmap.pdf
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
    FAMILY_COLORS,
    FAMILY_MAP,
    FAMILY_ORDER,
    MODELS,
    REPO_ROOT,
    label,
    load_persona_remap,
    persona_base_distances,
)

N_PERSONAS = 100


def _models_by_family() -> list[str]:
    return sorted(MODELS, key=lambda m: (FAMILY_ORDER.index(FAMILY_MAP[m]), m))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure-dir", type=Path, default=REPO_ROOT / "paper" / "figures")
    parser.add_argument("--which", choices=["grid", "heatmap", "both"], default="heatmap")
    return parser.parse_args()


def _load_all() -> pd.DataFrame:
    data = pd.concat([persona_base_distances(m) for m in MODELS], ignore_index=True)
    remap = load_persona_remap()
    data["persona_id"] = data["persona_id"].map(remap)  # relabel to ascending-shift ids
    return data


def _grid_figure(data: pd.DataFrame, out: Path) -> None:
    models = _models_by_family()
    ymax = float(data["distance_from_self"].max()) * 1.08
    fig, axes = plt.subplots(3, 5, figsize=(15.0, 7.2), sharex=True, sharey=True)
    for ax, model in zip(axes.flat, models):
        sub = data[data["model"] == model]
        color = FAMILY_COLORS[FAMILY_MAP[model]]
        x = sub["persona_id"].to_numpy()
        y = sub["distance_from_self"].to_numpy()
        ax.vlines(x, 0, y, color=color, alpha=0.45, linewidth=0.7, zorder=2)
        ax.plot(x, y, "o", ms=2.4, color=color, alpha=0.9, zorder=3)
        ax.set_title(label(model), fontsize=10)
        ax.set_xlim(-1, N_PERSONAS)
        ax.set_ylim(0, ymax)
        ax.grid(True, alpha=0.2, zorder=0)
    for ax in axes[-1, :]:
        ax.set_xlabel("Persona id", fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("Shift from baseline", fontsize=9)
    fig.suptitle(
        "Per-persona shift from the no-persona baseline, by model "
        r"($d_{m,p}=\sqrt{\sum_q (\mu_{m,p,q}-\mu_{m,\mathrm{self},q})^2}$)",
        fontsize=12, y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out.with_suffix('.pdf')}")


def _heatmap_figure(data: pd.DataFrame, out: Path) -> None:
    models = _models_by_family()
    matrix = np.full((len(models), N_PERSONAS), np.nan)
    for i, model in enumerate(models):
        sub = data[data["model"] == model]
        matrix[i, sub["persona_id"].to_numpy()] = sub["distance_from_self"].to_numpy()

    fig, ax = plt.subplots(figsize=(13.0, 5.2))
    cmap = plt.get_cmap("magma_r").copy()
    cmap.set_bad("#DDDDDD")
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, interpolation="nearest")

    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([label(m) for m in models], fontsize=9)
    # color each model tick label by family
    for tick, model in zip(ax.get_yticklabels(), models):
        tick.set_color(FAMILY_COLORS[FAMILY_MAP[model]])
    ax.set_xticks(range(0, N_PERSONAS, 5))
    ax.set_xticklabels(range(0, N_PERSONAS, 5), fontsize=8)
    ax.set_xlabel("Persona id", fontsize=11)
    # family separators
    boundaries = []
    for i in range(1, len(models)):
        if FAMILY_MAP[models[i]] != FAMILY_MAP[models[i - 1]]:
            boundaries.append(i - 0.5)
    for b in boundaries:
        ax.axhline(b, color="white", linewidth=1.6)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label("Shift from baseline  $d_{m,p}$", fontsize=10)
    ax.set_title(
        "Per-persona shift from the no-persona baseline (models × personas).\n"
        "Vertical stripes = personas that move every model; gray = dropped (parsing failures).",
        fontsize=11,
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out.with_suffix('.pdf')}")


def main() -> None:
    args = parse_args()
    data = _load_all()
    if args.which in ("grid", "both"):
        _grid_figure(data, args.figure_dir / "persona_shift_by_model_grid")
    if args.which in ("heatmap", "both"):
        _heatmap_figure(data, args.figure_dir / "persona_shift_by_model_heatmap")


if __name__ == "__main__":
    main()
