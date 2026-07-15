#!/usr/bin/env python3
"""Config-driven Oxford Utilitarianism Scale (OUS) sampling collector.

Mirrors ``run_mfq_sampling.py`` exactly (persona conditioning, sampling
budget, resume behaviour, CSV schema) but swaps the MFQ items for the 9-item
OUS (see ``ous_questions``). Output is written to ``data/sampling_ous`` so it
never collides with the MFQ runs, and downstream metrics/plots are computed by
``analysis/compute_ous_metrics.py`` and ``analysis/plot_ous_metrics.py``.

Usage:
    python run_ous_sampling.py --model gpt-4.1-mini --temperature 0.1 --n 10 --p 100
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from llm_interface import get_llm_response
from ous_questions import iter_questions
from model_registry import (
    DATA_DIR,
    benchmark_defaults,
    ensure_data_dirs,
    model_config,
    model_output_stem,
    prompt_for_model_selection,
    request_kwargs_for_model,
    temperature_tag,
)

# Reuse the instrument-agnostic helpers from the MFQ collector so the resume
# logic, CSV schema, and parsing stay identical across the two instruments.
from run_mfq_sampling import (
    FIELDNAMES,
    create_persona_prompt,
    extract_rating,
    load_personas,
    _load_existing_sampling_rows,
    _write_sampling_rows,
)

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


OUS_SAMPLING_DIR = DATA_DIR / "sampling_ous"


def resolve_ous_output_path(identifier_or_model: str | dict[str, Any], temperature: float) -> Path:
    return OUS_SAMPLING_DIR / f"{model_output_stem(identifier_or_model)}_{temperature_tag(temperature)}.csv"


def run_ous_sampling(
    personas: List[str],
    model_type: str,
    model_name: str,
    n: int = 10,
    csv_writer: Optional[csv.DictWriter] = None,
    csv_file=None,
    existing_valid_slots: Optional[Set[Tuple[int, int, int]]] = None,
    slot_failures: Optional[Dict[Tuple[int, int, int], int]] = None,
    row_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    **model_kwargs,
) -> Tuple[int, int]:
    if csv_writer is None and row_callback is None:
        raise ValueError("run_ous_sampling requires a csv_writer unless row_callback is provided")

    questions = list(iter_questions())
    personas_processed = 0
    responses_written = 0
    existing_valid_slots = existing_valid_slots or set()
    slot_failures = slot_failures or {}

    print(f"Running OUS experiment with {len(personas)} personas using {model_type}:{model_name}")

    for persona_id, persona in enumerate(personas):
        persona_text = str(persona)
        print(f"\nProgress: {persona_id + 1}/{len(personas)} - {persona_text[:50]}...")
        personas_processed += 1

        for question in questions:
            prompt = create_persona_prompt(persona_text, question.prompt)
            for run_index in range(1, n + 1):
                slot_key = (persona_id, question.id, run_index)
                if slot_key in existing_valid_slots:
                    continue

                response = get_llm_response(model_type, model_name, prompt, **model_kwargs)
                rating = extract_rating(response)
                response_text = response.strip() if isinstance(response, str) else str(response)

                prior_failures = slot_failures.get(slot_key, 0)
                failures = prior_failures + (1 if rating < 0 else 0)
                row = {
                    "persona_id": persona_id,
                    "question_id": question.id,
                    "run_index": run_index,
                    "rating": rating,
                    "failures": failures,
                    "response": response_text,
                    "collected_at": datetime.now().isoformat(),
                }

                if csv_writer is not None:
                    csv_writer.writerow(row)
                    responses_written += 1
                    if csv_file is not None:
                        csv_file.flush()
                else:
                    responses_written += 1

                slot_failures[slot_key] = failures

                if row_callback is not None:
                    row_callback(dict(row))
                if rating >= 0:
                    existing_valid_slots.add(slot_key)

    return personas_processed, responses_written


def parse_args() -> argparse.Namespace:
    defaults = benchmark_defaults()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None, help="Model key from config/models.yaml. Defaults to interactive selection.")
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(defaults.get("temperature", 0.1)),
        help="Sampling temperature in 0.1 increments.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=int(defaults.get("n", 10)),
        help="Number of repeated answers per persona-question cell.",
    )
    parser.add_argument(
        "--p",
        type=int,
        default=int(defaults.get("p", 100)),
        help="Number of personas to include.",
    )
    parser.add_argument(
        "--personas-file",
        type=Path,
        default=Path(defaults.get("personas_file", "personas.json")),
        help="Persona JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output CSV path. Defaults to data/sampling_ous/<model>_tempXX.csv.",
    )
    parser.add_argument("--limit", type=int, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    ensure_data_dirs()
    OUS_SAMPLING_DIR.mkdir(parents=True, exist_ok=True)
    args = parse_args()
    selected_model = (
        model_config(args.model, capability="sampling") if args.model else prompt_for_model_selection("sampling")
    )
    model_type = str(selected_model["provider"])
    model_name = str(selected_model["model_name"])
    model_kwargs = {"temperature": args.temperature, "max_tokens": 1}
    model_kwargs.update(request_kwargs_for_model(selected_model))
    print(f"Selected model: {selected_model['label']} ({model_type}:{model_name})")

    persona_limit = args.limit if args.limit is not None else args.p
    try:
        personas = load_personas(args.personas_file, persona_limit)
    except FileNotFoundError:
        print(f"Error: Could not find personas file: {args.personas_file}")
        return

    print(f"Loaded {len(personas)} personas")
    output_path = args.output or resolve_ous_output_path(selected_model, args.temperature)

    file_exists, existing_valid_slots, slot_failures, rows_by_key, had_missing_failures = _load_existing_sampling_rows(output_path)
    if existing_valid_slots:
        print(f"Found {len(existing_valid_slots)} valid existing slots. Only missing or invalid entries will be run.")

    if file_exists:
        def handle_new_row(row: Dict[str, Any]) -> None:
            key = (row["persona_id"], row["question_id"], row["run_index"])
            rows_by_key[key] = row
            slot_failures[key] = row.get("failures", 0)
            _write_sampling_rows(output_path, rows_by_key)

        personas_processed, responses_written = run_ous_sampling(
            personas,
            model_type,
            model_name,
            n=args.n,
            existing_valid_slots=set(existing_valid_slots),
            slot_failures=slot_failures,
            row_callback=handle_new_row,
            **model_kwargs,
        )
        if responses_written == 0 and had_missing_failures and rows_by_key:
            _write_sampling_rows(output_path, rows_by_key)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
            writer.writeheader()
            personas_processed, responses_written = run_ous_sampling(
                personas,
                model_type,
                model_name,
                n=args.n,
                csv_writer=writer,
                csv_file=csv_file,
                slot_failures=slot_failures,
                **model_kwargs,
            )

    if file_exists and responses_written == 0:
        print("\nNo new runs were required; all slots were already filled with valid ratings.")

    print(
        f"\nExperiment completed. Processed {personas_processed} personas and logged {responses_written} responses to {output_path}."
    )


if __name__ == "__main__":
    main()
