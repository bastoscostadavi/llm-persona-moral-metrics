#!/usr/bin/env python3
"""Compute R and S for the MFQ prompt-paraphrase variants (Deliverable 5).

Reads the baseline (main-paper) run and the four paraphrase reruns for one
model and computes moral robustness (R) and susceptibility (S) for each prompt
variant. All variants are evaluated on the *same* intersection of valid
personas and questions so the only thing that differs across bars is the prompt
wording -- this is what makes it a clean prompt-robustness comparison.

Estimator (per-cell summaries, persona + rerun bootstrap, R = 1/sigma-bar) is
imported verbatim from ``compute_metrics`` so the numbers are directly
comparable to the main-paper Fig. 3.

Inputs:
    v0 (baseline): data/sampling/<stem>_tempXX.csv        (reused verbatim)
    v1..v4:        data/sampling_prompt_variations/<stem>-vK_tempXX.csv

Outputs:
    results/prompt_variation_metrics.csv
    results/prompt_variation_metrics_per_foundation.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from compute_metrics import (
    FOUNDATION_BY_QUESTION,
    FOUNDATION_ORDER,
    _metrics_from_summary,
    _seed_from_parts,
    _summary_frame_for_path,
    personas_with_valid_stats,
)
from model_registry import model_output_stem, temperature_tag
from prompt_variations import VARIANTS

RESULTS_DIR = REPO_ROOT / "results"
BASELINE_DIR = REPO_ROOT / "data" / "sampling"
VARIATIONS_DIR = REPO_ROOT / "data" / "sampling_prompt_variations"
DEFAULT_SUMMARY_CACHE = RESULTS_DIR / "sampling_summary_cache_prompt_variations"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt-4.1-mini", help="Model key (default: gpt-4.1-mini).")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--summary-cache-dir", type=Path, default=DEFAULT_SUMMARY_CACHE)
    parser.add_argument("--output", type=Path, default=None,
                        help="Overall metrics CSV (default: results/prompt_variation_metrics_<stem>.csv).")
    parser.add_argument("--foundation-output", type=Path, default=None,
                        help="Per-foundation metrics CSV (default: results/prompt_variation_metrics_per_foundation_<stem>.csv).")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--response-bootstrap-samples", type=int, default=400)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--v0-from-variations", action="store_true",
                        help="Use the freshly-collected v0 in data/sampling_prompt_variations/ instead of the "
                             "main-paper baseline in data/sampling/. Needed when v0 had to be recollected with "
                             "different settings (e.g. Claude Haiku's larger token budget).")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _variant_path(stem: str, variant_key: str, temperature: float, v0_from_variations: bool = False) -> Path:
    tag = temperature_tag(temperature)
    if variant_key == "v0" and not v0_from_variations:
        return BASELINE_DIR / f"{stem}_{tag}.csv"
    return VARIATIONS_DIR / f"{stem}-{variant_key}_{tag}.csv"


def _row_from_metrics(variant_key, label, metrics, n_personas, n_questions,
                      personas_json, questions_json, source_file, foundation=None):
    (
        uncertainty, uncertainty_se, susceptibility, susceptibility_se,
        uncertainty_persona_se, uncertainty_response_se,
        susceptibility_persona_se, susceptibility_response_se,
        min_runs, median_runs, max_runs,
    ) = metrics
    row = {"variant": variant_key, "variant_label": label}
    if foundation is not None:
        row["foundation"] = foundation
    row.update({
        "uncertainty": uncertainty,
        "uncertainty_uncertainty": uncertainty_se,
        "robustness": 1.0 / uncertainty,
        "robustness_uncertainty": uncertainty_se / (uncertainty ** 2) if uncertainty_se > 0 else 0.0,
        "susceptibility": susceptibility,
        "susceptibility_uncertainty": susceptibility_se,
        "uncertainty_persona_uncertainty": uncertainty_persona_se,
        "uncertainty_response_uncertainty": uncertainty_response_se,
        "susceptibility_persona_uncertainty": susceptibility_persona_se,
        "susceptibility_response_uncertainty": susceptibility_response_se,
        "personas": n_personas,
        "questions": n_questions,
        "min_runs_per_cell": min_runs,
        "median_runs_per_cell": median_runs,
        "max_runs_per_cell": max_runs,
        "retained_persona_ids_json": personas_json,
        "retained_question_ids_json": questions_json,
        "source_file": source_file,
    })
    return row


def main() -> None:
    args = parse_args()
    stem = model_output_stem(args.model)
    overall_out = args.output or (RESULTS_DIR / f"prompt_variation_metrics_{stem}.csv")
    foundation_out = args.foundation_output or (RESULTS_DIR / f"prompt_variation_metrics_per_foundation_{stem}.csv")

    # Load every variant's summary frame and its set of valid personas/questions.
    variant_runs = []
    for variant in VARIANTS:
        path = _variant_path(stem, variant.key, args.temperature, args.v0_from_variations)
        if not path.exists():
            print(f"WARNING: missing data for {variant.key}: {path} -- skipping.", file=sys.stderr)
            continue
        frame = _summary_frame_for_path(path, args.summary_cache_dir)
        question_ids = sorted(int(q) for q in frame["question_id"].astype(int).unique())
        _, valid_personas = personas_with_valid_stats(frame, question_ids)
        if not valid_personas:
            print(f"WARNING: no valid personas for {variant.key} -- skipping.", file=sys.stderr)
            continue
        variant_runs.append({
            "variant": variant,
            "path": path,
            "frame": frame,
            "question_ids": set(question_ids),
            "valid_personas": set(int(p) for p in valid_personas),
        })

    if len(variant_runs) < 2:
        raise RuntimeError("Need at least two variants with data to compare.")

    # Common persona/question set across ALL variants -> only the prompt differs.
    common_questions = set.intersection(*(r["question_ids"] for r in variant_runs))
    common_personas = set.intersection(*(r["valid_personas"] for r in variant_runs))
    retained_question_ids = sorted(int(q) for q in common_questions)
    retained_persona_ids = sorted(int(p) for p in common_personas)
    if not retained_question_ids or not retained_persona_ids:
        raise RuntimeError("Empty intersection of personas/questions across variants.")

    print(f"Comparing {len(variant_runs)} variants on "
          f"{len(retained_persona_ids)} shared personas x {len(retained_question_ids)} questions.")

    foundation_questions = {
        foundation: [q for q in retained_question_ids if FOUNDATION_BY_QUESTION.get(q) == foundation]
        for foundation in FOUNDATION_ORDER
    }
    personas_json = json.dumps(retained_persona_ids)
    questions_json = json.dumps(retained_question_ids)

    overall_rows = []
    foundation_rows = []

    for run in variant_runs:
        variant = run["variant"]
        raw_frame = pd.read_csv(run["path"])
        try:
            source_file = str(run["path"].relative_to(REPO_ROOT))
        except ValueError:
            source_file = str(run["path"])

        rng = np.random.default_rng(_seed_from_parts(variant.key, args.temperature, args.seed, "overall"))
        metrics = _metrics_from_summary(
            run["frame"], raw_frame, retained_question_ids, retained_persona_ids,
            args.bootstrap_samples, args.response_bootstrap_samples, rng,
        )
        overall_rows.append(_row_from_metrics(
            variant.key, variant.label, metrics, len(retained_persona_ids),
            len(retained_question_ids), personas_json, questions_json, source_file,
        ))
        if args.verbose:
            print(f"  {variant.key}: R={overall_rows[-1]['robustness']:.3f} "
                  f"S={overall_rows[-1]['susceptibility']:.3f}")

        for foundation in FOUNDATION_ORDER:
            qids = foundation_questions[foundation]
            if not qids:
                continue
            f_rng = np.random.default_rng(_seed_from_parts(variant.key, args.temperature, foundation, args.seed, "found"))
            f_metrics = _metrics_from_summary(
                run["frame"], raw_frame, qids, retained_persona_ids,
                args.bootstrap_samples, args.response_bootstrap_samples, f_rng,
            )
            foundation_rows.append(_row_from_metrics(
                variant.key, variant.label, f_metrics, len(retained_persona_ids),
                len(qids), personas_json, json.dumps(qids), source_file, foundation=foundation,
            ))

    # Preserve the canonical v0..v4 order.
    order = {v.key: i for i, v in enumerate(VARIANTS)}
    overall_df = pd.DataFrame(overall_rows)
    overall_df = overall_df.sort_values("variant", key=lambda s: s.map(order)).reset_index(drop=True)
    overall_out.parent.mkdir(parents=True, exist_ok=True)
    overall_df.to_csv(overall_out, index=False)
    print(f"Wrote {len(overall_df)} overall rows to {overall_out}")

    if foundation_rows:
        foundation_df = pd.DataFrame(foundation_rows)
        foundation_df = foundation_df.sort_values(
            ["variant", "foundation"], key=lambda s: s.map(order) if s.name == "variant" else s
        ).reset_index(drop=True)
        foundation_out.parent.mkdir(parents=True, exist_ok=True)
        foundation_df.to_csv(foundation_out, index=False)
        print(f"Wrote {len(foundation_df)} foundation rows to {foundation_out}")


if __name__ == "__main__":
    main()
