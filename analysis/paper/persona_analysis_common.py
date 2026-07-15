#!/usr/bin/env python3
"""Shared helpers for persona-set robustness analyses (rebuttal deliverables).

Loads raw T=0.1 sampling files, builds per-(persona, question) summaries, and
exposes the R/S metric computation on arbitrary persona/question subsets. Used by
`generate_persona_subsample_stability.py` (D1) and
`generate_persona_distance_scatter.py` (D2).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model_registry import label_for_model  # noqa: E402

SAMPLING_DIR = REPO_ROOT / "data" / "sampling"

# The 15 benchmark models (T=0.1), grouped into the six provider families used in
# the paper. Order controls plotting/legend order.
FAMILY_MAP = {
    "claude-haiku-4-5": "Claude",
    "claude-sonnet-4-5": "Claude",
    "deepseek-v3": "DeepSeek",
    "deepseek-v3.1": "DeepSeek",
    "gemini-2.5-flash": "Gemini",
    "gemini-2.5-flash-lite": "Gemini",
    "gpt-4.1": "GPT",
    "gpt-4.1-mini": "GPT",
    "gpt-4.1-nano": "GPT",
    "gpt-4o": "GPT",
    "gpt-4o-mini": "GPT",
    "grok-4": "Grok",
    "grok-4-fast": "Grok",
    "llama-4-maverick": "Llama",
    "llama-4-scout": "Llama",
}

MODELS = list(FAMILY_MAP.keys())

FAMILY_ORDER = ["Claude", "DeepSeek", "Gemini", "GPT", "Grok", "Llama"]

FAMILY_COLORS = {
    "Claude": "#E67E22",   # orange
    "DeepSeek": "#6A4C93", # purple
    "Gemini": "#C9A227",   # gold
    "GPT": "#2E8B57",      # sea green
    "Grok": "#C0392B",     # red
    "Llama": "#2980B9",    # blue
}


def _sampling_path(model: str, self_baseline: bool = False) -> Path:
    suffix = "_temp01_self.csv" if self_baseline else "_temp01.csv"
    return SAMPLING_DIR / f"{model}{suffix}"


def _clean_ratings(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    return df[df["rating"].notna() & (df["rating"] != -1)]


def load_summary(model: str) -> tuple[pd.DataFrame, list[int], list[int]]:
    """Return (summary, valid_personas, question_ids) for a model's T=0.1 run.

    ``summary`` has one row per (persona_id, question_id) with columns
    ``average_score`` and ``standard_deviation`` (sample SD, 0 when a single
    valid rating). ``valid_personas`` are personas with a valid entry for every
    question (matching the paper's per-model persona retention).
    """
    df = pd.read_csv(_sampling_path(model))
    valid = _clean_ratings(df)
    question_ids = sorted(int(q) for q in df["question_id"].unique())
    stats = (
        valid.groupby(["persona_id", "question_id"], as_index=False)["rating"].agg(
            average_score="mean",
            standard_deviation=lambda s: 0.0 if len(s) <= 1 else float(s.std(ddof=1)),
        )
    )
    # personas complete across all questions
    counts = stats.groupby("persona_id")["question_id"].nunique()
    valid_personas = sorted(int(p) for p, c in counts.items() if c == len(question_ids))
    return stats, valid_personas, question_ids


def load_self_baseline(model: str) -> dict[int, float]:
    """Return {question_id: mean rating} for the no-persona `self` run."""
    df = pd.read_csv(_sampling_path(model, self_baseline=True))
    valid = _clean_ratings(df)
    means = valid.groupby("question_id")["rating"].mean()
    return {int(q): float(m) for q, m in means.items()}


def compute_R_S(
    summary: pd.DataFrame,
    personas: list[int],
    question_ids: list[int],
) -> tuple[float, float, float]:
    """Compute (R, S, sigma_bar) on the given persona/question subset."""
    sub = summary[
        summary["persona_id"].isin(personas) & summary["question_id"].isin(question_ids)
    ]
    std_pivot = sub.pivot(index="persona_id", columns="question_id", values="standard_deviation")
    avg_pivot = sub.pivot(index="persona_id", columns="question_id", values="average_score")
    sigma_bar = float(std_pivot.to_numpy(dtype=float).mean())
    robustness = float("inf") if sigma_bar == 0 else 1.0 / sigma_bar
    # susceptibility: per-question SD across persona means, averaged over questions
    per_question_tau = avg_pivot.std(axis=0, ddof=1)
    susceptibility = float(per_question_tau.mean())
    return robustness, susceptibility, sigma_bar


def persona_base_distances(model: str) -> pd.DataFrame:
    """Per-persona Euclidean distance of the mean MFQ profile from the model's
    no-persona `self` baseline, over the 30 questions.

    Returns one row per valid persona with columns ``model``, ``family``,
    ``persona_id`` and ``distance_from_self``.
    """
    summary, valid_personas, question_ids = load_summary(model)
    self_means = load_self_baseline(model)
    q_common = [q for q in question_ids if q in self_means]
    self_vec = np.array([self_means[q] for q in q_common])

    sub = summary[
        summary["persona_id"].isin(valid_personas) & summary["question_id"].isin(q_common)
    ]
    avg_pivot = sub.pivot(index="persona_id", columns="question_id", values="average_score")[q_common]
    diffs = avg_pivot.to_numpy() - self_vec[None, :]
    distance = np.sqrt((diffs ** 2).sum(axis=1))
    return pd.DataFrame(
        {
            "model": model,
            "family": family_of(model),
            "persona_id": avg_pivot.index.astype(int),
            "distance_from_self": distance,
        }
    )


def load_persona_remap() -> dict[int, int]:
    """Return {old_id: new_id}, where new ids order personas by ascending shift.

    Requires ``results/persona_id_remap.csv`` (produced by
    ``generate_persona_id_remap.py``).
    """
    path = REPO_ROOT / "results" / "persona_id_remap.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; run analysis/paper/generate_persona_id_remap.py first."
        )
    remap = pd.read_csv(path)
    return {int(o): int(n) for o, n in zip(remap["old_id"], remap["new_id"])}


def family_of(model: str) -> str:
    return FAMILY_MAP[model]


def label(model: str) -> str:
    return label_for_model(model)
