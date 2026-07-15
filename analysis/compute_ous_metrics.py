#!/usr/bin/env python3
"""Compute sampled OUS metrics from raw OUS sampling files.

Parallel to ``analysis/compute_metrics.py`` but for the Oxford Utilitarianism
Scale. Reads ``data/sampling_ous/*_temp*.csv``, bootstraps over personas and
repeated runs, and writes:

- ``results/ous_persona_moral_metrics.csv``            (overall R, S by model/temperature)
- ``results/ous_persona_moral_metrics_per_subscale.csv`` (IH / IB decomposition)

The heavy lifting (per-cell summaries, persona/rerun bootstrap, alignment) is
reused verbatim from ``compute_metrics`` so the two instruments share exactly
the same estimator; only the item set and the subscale decomposition differ.
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

from ous_questions import SUBSCALE_BY_QUESTION, SUBSCALE_ORDER
from compute_metrics import (
    TEMP_FILE_PATTERN,
    _alignment_for_runs,
    _metrics_from_summary,
    _seed_from_parts,
    _summary_frame_for_path,
    personas_with_valid_stats,
)


OUS_SAMPLING_DIR = REPO_ROOT / "data" / "sampling_ous"
RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_SUMMARY_CACHE = RESULTS_DIR / "sampling_summary_cache_ous"
DEFAULT_OVERALL_OUT = RESULTS_DIR / "ous_persona_moral_metrics.csv"
DEFAULT_SUBSCALE_OUT = RESULTS_DIR / "ous_persona_moral_metrics_per_subscale.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=OUS_SAMPLING_DIR,
                        help="Directory containing raw OUS sampling CSVs (default: data/sampling_ous).")
    parser.add_argument("--summary-cache-dir", type=Path, default=DEFAULT_SUMMARY_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OVERALL_OUT)
    parser.add_argument("--subscale-output", type=Path, default=DEFAULT_SUBSCALE_OUT)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--response-bootstrap-samples", type=int, default=400)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--min-attempted-personas", type=int, default=90,
                        help="Skip models whose raw file covers fewer than this many personas "
                             "(guards against including still-running / partial collections).")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def candidate_sampling_files(data_dir: Path) -> dict[str, list[tuple[float, Path]]]:
    models: dict[str, list[tuple[float, Path]]] = {}
    for path in sorted(data_dir.glob("*_temp*.csv")):
        match = TEMP_FILE_PATTERN.match(path.stem)
        if not match:
            continue
        models.setdefault(match.group("model"), []).append((int(match.group("temp")) / 10.0, path))
    return models


def _row_from_metrics(model: str, temperature: float, metrics: tuple, n_personas: int,
                      n_questions: int, personas_json: str, questions_json: str,
                      source_file: str, subscale: str | None = None) -> dict[str, object]:
    (
        uncertainty, uncertainty_se, susceptibility, susceptibility_se,
        uncertainty_persona_se, uncertainty_response_se,
        susceptibility_persona_se, susceptibility_response_se,
        min_runs, median_runs, max_runs,
    ) = metrics
    row: dict[str, object] = {
        "model": model,
        "temperature": temperature,
    }
    if subscale is not None:
        row["subscale"] = subscale
    row.update(
        {
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
        }
    )
    return row


def main() -> None:
    args = parse_args()
    model_files = candidate_sampling_files(args.data_dir)
    if not model_files:
        raise RuntimeError(f"No *_temp*.csv files found in {args.data_dir}")

    overall_rows: list[dict[str, object]] = []
    subscale_rows: list[dict[str, object]] = []

    for model, temp_files in sorted(model_files.items()):
        runs: list[dict[str, object]] = []
        for temperature, path in sorted(temp_files):
            frame = _summary_frame_for_path(path, args.summary_cache_dir)
            attempted_personas = int(frame["persona_id"].nunique())
            if attempted_personas < args.min_attempted_personas:
                if args.verbose:
                    print(f"Skipping {model} @ T={temperature}: only {attempted_personas} personas "
                          f"attempted (< {args.min_attempted_personas}); likely still running.", file=sys.stderr)
                continue
            question_ids = sorted(int(qid) for qid in frame["question_id"].astype(int).unique())
            _, valid_personas = personas_with_valid_stats(frame, question_ids)
            if not valid_personas:
                continue
            runs.append({
                "temperature": temperature,
                "path": path,
                "frame": frame,
                "question_ids": question_ids,
                "valid_personas": sorted(valid_personas),
            })
        if not runs:
            continue

        # OUS has only 9 items; relax the alignment persona floor accordingly.
        retained_question_ids, retained_persona_ids = _alignment_for_runs(runs, min_personas=50)
        if not retained_question_ids or not retained_persona_ids:
            if args.verbose:
                print(f"Skipping {model}: no shared personas/questions across temperatures.", file=sys.stderr)
            continue

        if args.verbose:
            print(f"{model}: {len(retained_persona_ids)} personas, {len(retained_question_ids)} questions "
                  f"across {len(runs)} temperatures", file=sys.stderr)

        subscale_questions = {
            subscale: [qid for qid in retained_question_ids if SUBSCALE_BY_QUESTION.get(qid) == subscale]
            for subscale in SUBSCALE_ORDER
        }
        personas_json = json.dumps(retained_persona_ids)
        questions_json = json.dumps(retained_question_ids)

        for run in sorted(runs, key=lambda item: float(item["temperature"])):
            temperature = float(run["temperature"])
            raw_frame = pd.read_csv(run["path"])
            try:
                source_file = str(Path(run["path"]).relative_to(REPO_ROOT))
            except ValueError:
                source_file = str(run["path"])

            rng = np.random.default_rng(_seed_from_parts(model, temperature, args.seed, "overall"))
            metrics = _metrics_from_summary(
                run["frame"], raw_frame, retained_question_ids, retained_persona_ids,
                args.bootstrap_samples, args.response_bootstrap_samples, rng,
            )
            overall_rows.append(_row_from_metrics(
                model, temperature, metrics, len(retained_persona_ids),
                len(retained_question_ids), personas_json, questions_json, source_file,
            ))

            for subscale in SUBSCALE_ORDER:
                qids = subscale_questions[subscale]
                if not qids:
                    continue
                sub_rng = np.random.default_rng(_seed_from_parts(model, temperature, subscale, args.seed, "subscale"))
                sub_metrics = _metrics_from_summary(
                    run["frame"], raw_frame, qids, retained_persona_ids,
                    args.bootstrap_samples, args.response_bootstrap_samples, sub_rng,
                )
                subscale_rows.append(_row_from_metrics(
                    model, temperature, sub_metrics, len(retained_persona_ids),
                    len(qids), personas_json, json.dumps(qids), source_file, subscale=subscale,
                ))

    if not overall_rows:
        raise RuntimeError("No sampled OUS metrics were computed.")

    overall_df = pd.DataFrame(overall_rows).sort_values(["model", "temperature"]).reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    overall_df.to_csv(args.output, index=False)
    print(f"Wrote {len(overall_df)} overall OUS metric rows to {args.output}")

    if subscale_rows:
        subscale_df = (
            pd.DataFrame(subscale_rows)
            .sort_values(["model", "temperature", "subscale"])
            .reset_index(drop=True)
        )
        args.subscale_output.parent.mkdir(parents=True, exist_ok=True)
        subscale_df.to_csv(args.subscale_output, index=False)
        print(f"Wrote {len(subscale_df)} subscale OUS metric rows to {args.subscale_output}")


if __name__ == "__main__":
    main()
