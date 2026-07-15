#!/usr/bin/env python3
"""Build the persona id remap: original id -> new id ordered by ascending shift.

New ids enumerate the personas from the smallest to the largest model-averaged
distance from the no-persona baseline (so the strongest mover gets the highest
id). Raw data files keep their original persona ids; this mapping bridges them to
the renumbered appendix and figures.

Output: results/persona_id_remap.csv with columns
    old_id, new_id, avg_distance, n_models, persona_text
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from persona_analysis_common import MODELS, REPO_ROOT, persona_base_distances  # noqa: E402

OUTPUT = REPO_ROOT / "results" / "persona_id_remap.csv"
PERSONAS_JSON = REPO_ROOT / "personas.json"


def build_remap() -> pd.DataFrame:
    by_model = pd.concat([persona_base_distances(m) for m in MODELS], ignore_index=True)
    agg = (
        by_model.groupby("persona_id")["distance_from_self"]
        .agg(avg_distance="mean", n_models="count")
        .reset_index()
        .rename(columns={"persona_id": "old_id"})
        .sort_values("avg_distance", ascending=True)  # smallest shift -> new id 0
        .reset_index(drop=True)
    )
    agg["new_id"] = agg.index.astype(int)

    personas = json.loads(PERSONAS_JSON.read_text(encoding="utf-8"))
    agg["persona_text"] = agg["old_id"].map(lambda i: personas[int(i)])
    return agg[["old_id", "new_id", "avg_distance", "n_models", "persona_text"]]


def main() -> None:
    remap = build_remap()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    remap.to_csv(OUTPUT, index=False)
    print(f"Wrote {len(remap)} rows to {OUTPUT}")
    print("\nTop movers (highest new ids):")
    for _, r in remap.tail(10)[::-1].iterrows():
        print(f"  old p{int(r.old_id):2d} -> new p{int(r.new_id):2d}  "
              f"d={r.avg_distance:.2f}  {r.persona_text[:60]}")
    print("\nRemap for personas cited in the failures table:")
    for old in (66, 94):
        row = remap[remap.old_id == old].iloc[0]
        print(f"  old p{old} -> new p{int(row.new_id)}")


if __name__ == "__main__":
    main()
