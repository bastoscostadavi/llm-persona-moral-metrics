#!/usr/bin/env python3
"""Collect persona-conditioned MFQ responses under a paraphrased prompt template.

Deliverable 5 (reviewer R2.3, prompt-robustness): rerun the MFQ sampling for a
single fast model (``gpt-4.1-mini`` by default) under several semantically
equivalent but reworded prompt templates, defined in ``prompt_variations.py``.

One invocation collects exactly one variant so the four new variants can be run
as independent parallel processes. API calls within a run are issued
concurrently (``--workers``) because each cell is a single-token completion and
the run is otherwise dominated by request latency.

Output: ``data/sampling_prompt_variations/<stem>-<variant>_tempXX.csv`` with the
same schema and resume semantics as ``run_mfq_sampling.py``. The ``v0`` baseline
is NOT collected here -- it reuses the existing main-paper run verbatim.

Usage:
    python run_mfq_prompt_variations.py --model gpt-4.1-mini --variant v1 \
        --temperature 0.1 --n 10 --p 100 --workers 16
"""

from __future__ import annotations

import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from llm_interface import get_llm_response
from mfq_questions import iter_questions
from model_registry import (
    DATA_DIR,
    benchmark_defaults,
    ensure_data_dirs,
    model_config,
    model_output_stem,
    request_kwargs_for_model,
    temperature_tag,
)
from prompt_variations import VARIANT_BY_KEY
from run_mfq_sampling import (
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


PROMPT_VARIATIONS_DIR = DATA_DIR / "sampling_prompt_variations"


def resolve_output_path(stem: str, variant_key: str, temperature: float) -> Path:
    return PROMPT_VARIATIONS_DIR / f"{stem}-{variant_key}_{temperature_tag(temperature)}.csv"


def parse_args() -> argparse.Namespace:
    defaults = benchmark_defaults()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt-4.1-mini", help="Model key from config/models.yaml.")
    parser.add_argument("--variant", required=True, choices=sorted(VARIANT_BY_KEY), help="Prompt variant key to run.")
    parser.add_argument("--temperature", type=float, default=float(defaults.get("temperature", 0.1)))
    parser.add_argument("--n", type=int, default=int(defaults.get("n", 10)))
    parser.add_argument("--p", type=int, default=int(defaults.get("p", 100)))
    parser.add_argument("--personas-file", type=Path, default=Path(defaults.get("personas_file", "personas.json")))
    parser.add_argument("--workers", type=int, default=16, help="Concurrent API requests.")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Base completion token budget (default: model's request_kwargs, else 1).")
    parser.add_argument("--passes", type=int, default=1,
                        help="Number of retry passes at the base token budget. Each pass only re-attempts "
                             "slots that still lack a valid rating; at T>0 stochastic failures resolve across "
                             "passes, keeping the vast majority of cells on the minimal digit-only budget.")
    parser.add_argument("--escalate-max-tokens", default="",
                        help="Comma-separated token budgets to try, in order, for cells still failing after the "
                             "base passes (e.g. '8,16'). Only the stubborn residual (e.g. Claude Haiku personas "
                             "that deterministically prepend a markdown heading) is collected at these budgets.")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _run_pass(
    *, variant, personas, questions, n, model_type, model_name, model_kwargs,
    output_path: Path, workers: int, pass_label: str,
) -> int:
    """One collection pass: re-attempt every slot lacking a valid rating.

    Returns the number of slots still invalid after this pass.
    """
    _, existing_valid_slots, slot_failures, rows_by_key, _ = _load_existing_sampling_rows(output_path)

    todo: List[Tuple[int, int, int, str]] = []
    for persona_id, persona in enumerate(personas):
        persona_text = str(persona)
        for question in questions:
            prompt = variant.build(persona_text, question)
            for run_index in range(1, n + 1):
                slot_key = (persona_id, question.id, run_index)
                if slot_key in existing_valid_slots:
                    continue
                todo.append((persona_id, question.id, run_index, prompt))

    total_slots = len(personas) * len(questions) * n
    if not todo:
        print(f"[{pass_label}] all {total_slots} slots valid; nothing to do.")
        return 0
    print(f"[{pass_label}] {len(todo)} slots to (re)attempt with {workers} workers "
          f"(max_tokens={model_kwargs.get('max_tokens')}).")

    lock = threading.Lock()
    completed = 0
    checkpoint_every = max(200, len(todo) // 20)

    def worker(task: Tuple[int, int, int, str]) -> Dict[str, Any]:
        persona_id, question_id, run_index, prompt = task
        response = get_llm_response(model_type, model_name, prompt, **model_kwargs)
        rating = extract_rating(response)
        response_text = response.strip() if isinstance(response, str) else str(response)
        return {
            "persona_id": persona_id,
            "question_id": question_id,
            "run_index": run_index,
            "rating": rating,
            "response": response_text,
            "collected_at": datetime.now().isoformat(),
        }

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(worker, task) for task in todo]
        for future in as_completed(futures):
            row = future.result()
            slot_key = (row["persona_id"], row["question_id"], row["run_index"])
            with lock:
                prior = slot_failures.get(slot_key, 0)
                row["failures"] = prior + (1 if row["rating"] < 0 else 0)
                slot_failures[slot_key] = row["failures"]
                rows_by_key[slot_key] = row
                completed += 1
                if completed % checkpoint_every == 0:
                    _write_sampling_rows(output_path, rows_by_key)

    _write_sampling_rows(output_path, rows_by_key)
    remaining = sum(1 for r in rows_by_key.values() if int(r["rating"]) < 0)
    valid = len(rows_by_key) - remaining
    print(f"[{pass_label}] wrote {len(rows_by_key)} rows ({valid} valid, {remaining} still invalid).")
    return remaining


def main() -> None:
    ensure_data_dirs()
    PROMPT_VARIATIONS_DIR.mkdir(parents=True, exist_ok=True)
    args = parse_args()

    variant = VARIANT_BY_KEY[args.variant]
    selected_model = model_config(args.model, capability="sampling")
    model_type = str(selected_model["provider"])
    model_name = str(selected_model["model_name"])
    stem = model_output_stem(selected_model)

    base_kwargs = {"temperature": args.temperature, "max_tokens": 1}
    base_kwargs.update(request_kwargs_for_model(selected_model))
    if args.max_tokens is not None:
        base_kwargs["max_tokens"] = args.max_tokens
    base_max_tokens = base_kwargs["max_tokens"]

    escalation = [int(t) for t in str(args.escalate_max_tokens).split(",") if t.strip()]

    personas = load_personas(args.personas_file, args.p)
    questions = list(iter_questions())
    output_path = args.output or resolve_output_path(stem, variant.key, args.temperature)

    print(f"Model:   {selected_model['label']} ({model_type}:{model_name})")
    print(f"Variant: {variant.key} -- {variant.label}")
    print(f"Budget:  {len(personas)} personas x {len(questions)} items x {args.n} runs")
    print(f"Plan:    {args.passes} pass(es) @ max_tokens={base_max_tokens}"
          + (f", then escalate through {escalation} for residual failures" if escalation else "")
          + f"\nOutput:  {output_path}")

    common = dict(variant=variant, personas=personas, questions=questions, n=args.n,
                  model_type=model_type, model_name=model_name, output_path=output_path,
                  workers=args.workers)

    # Base passes at the minimal token budget.
    remaining = None
    for i in range(1, args.passes + 1):
        remaining = _run_pass(model_kwargs=dict(base_kwargs), pass_label=f"base {i}/{args.passes} (mt={base_max_tokens})", **common)
        if remaining == 0:
            break

    # Escalate the token budget only for the stubborn residual.
    for mt in escalation:
        if remaining == 0:
            break
        esc_kwargs = dict(base_kwargs)
        esc_kwargs["max_tokens"] = mt
        remaining = _run_pass(model_kwargs=esc_kwargs, pass_label=f"escalate (mt={mt})", **common)

    print(f"\nDone. {remaining} slot(s) still invalid after all passes -> {output_path}.")


if __name__ == "__main__":
    main()
