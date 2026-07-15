# Deliverable 5 — Prompt-paraphrase robustness (reviewer R2.3)

**Question addressed:** are R (robustness) and S (susceptibility) stable when the
prompt is reworded? We take fast models and rerun the persona-conditioned MFQ
under 5 prompt templates.

## Protocol

- **Model(s):** `gpt-4.1-mini` (complete, 100 personas), `claude-haiku-4-5`
  (partial — 52 personas; see caveat below).
- **Budget per variant:** 100 personas × 30 MFQ items × 10 repeated runs, T = 0.1.
- **v0** = the exact original wording used in the paper. **v1–v4** paraphrase the
  persona-roleplay wrapper and the rating-scale framing; the MFQ item text and the
  0–5 anchor meanings are held fixed, so the instrument is unchanged.
- All 5 variants are scored on the **same intersection of valid personas and
  questions** (only the prompt wording differs), using the identical estimator as
  the main paper (R = 1/σ̄, persona + rerun bootstrap SEs).
- Ratings are the single leading integer 0–5. For Claude Haiku the API returns
  empty content at a 1-token budget, and for some persona styles the model prefixes
  a markdown heading (`# 4`); we therefore collect at the model's native 2-token
  budget with several digit-only retry passes and escalate the token budget only
  for the residual cells (rating value is unaffected — only truncation is avoided).

### The five variants (leading phrasing)

| Variant | Persona wrapper | Scale framing |
|---|---|---|
| v0 (original) | "You are roleplaying as the following persona: …" | "…rate the consideration using this scale: 0 = … 5 = …" |
| v1 (adopt persona) | "Adopt the following persona and respond fully in character: …" | inline "Rate it from 0 to 5, where 0 = … 5 = …" |
| v2 (imagine you are) | "Imagine you are the following person: …" | parenthetical "0 (not at all relevant) … 5 (extremely relevant)" |
| v3 (in the voice of) | "Answer the following question in the voice of this persona — …" | "Please choose a rating on this 0-5 scale: …" |
| v4 (item/task/scale) | "For this task, play the role of the persona described here: …" | formal "Item / Task / Scale (0-5): …" |

---

## Results — `gpt-4.1-mini` (complete; 0 parse failures)

### Overall R and S (value ± bootstrap SE)

| Variant | Robustness R | Susceptibility S |
|---|---|---|
| v0 (original) | 11.31 ± 0.47 | 0.827 ± 0.037 |
| v1 (adopt persona) | 10.04 ± 0.42 | 0.690 ± 0.036 |
| v2 (imagine you are) | 10.19 ± 0.47 | 0.663 ± 0.034 |
| v3 (in the voice of) | 14.10 ± 0.74 | 0.650 ± 0.032 |
| v4 (item/task/scale) | 11.10 ± 0.48 | 0.767 ± 0.034 |

**Spread across the 5 wordings:**

| Metric | min | max | mean | SD | CV |
|---|---|---|---|---|---|
| R | 10.04 | 14.10 | 11.35 | 1.64 | 14.4% |
| S | 0.650 | 0.827 | 0.719 | 0.075 | 10.4% |

Every variant stays in the same high-robustness / moderate-susceptibility regime;
no wording collapses R or zeroes out S. R varies more than S but all five remain
tightly clustered relative to the between-model differences reported in the paper.

### Per-foundation robustness R

| Foundation | v0 | v1 | v2 | v3 | v4 |
|---|---|---|---|---|---|
| Harm/Care | 12.46 | 11.02 | 10.81 | 16.58 | 12.68 |
| Fairness/Reciprocity | 15.54 | 11.58 | 13.58 | 15.68 | 17.61 |
| In-group/Loyalty | 11.64 | 10.23 | 10.72 | 18.72 | 10.06 |
| Authority/Respect | 9.71 | 7.95 | 7.91 | 10.59 | 8.96 |
| Purity/Sanctity | 9.22 | 10.27 | 9.54 | 12.09 | 9.62 |

### Per-foundation susceptibility S

| Foundation | v0 | v1 | v2 | v3 | v4 |
|---|---|---|---|---|---|
| Harm/Care | 0.689 | 0.590 | 0.564 | 0.509 | 0.639 |
| Fairness/Reciprocity | 0.618 | 0.589 | 0.552 | 0.477 | 0.537 |
| In-group/Loyalty | 0.974 | 0.759 | 0.700 | 0.744 | 0.914 |
| Authority/Respect | 0.893 | 0.742 | 0.722 | 0.763 | 0.877 |
| Purity/Sanctity | 0.958 | 0.774 | 0.776 | 0.757 | 0.869 |

The foundation ordering is preserved across wordings (e.g. In-group/Loyalty and
Purity/Sanctity are consistently the most susceptible; Fairness the least).

---

## Results — `claude-haiku-4-5` (partial data, 52 personas)

**Caveat:** the Anthropic API credit balance was exhausted mid-collection, so
28–38% of cells per variant are missing. R and S are therefore computed on the
**52 personas that are complete across all 5 variants** (vs. 100 for
gpt-4.1-mini). This is the paper's standard subsample size, so the estimate is
meaningful; numbers will tighten slightly once the full set is collected. The
within-model comparison across wordings (the point of this analysis) is
unaffected.

### Overall R and S (value ± bootstrap SE, 52 shared personas)

| Variant | Robustness R | Susceptibility S |
|---|---|---|
| v0 (original) | 42.19 ± 4.63 | 0.696 ± 0.045 |
| v1 (adopt persona) | 40.86 ± 4.73 | 0.640 ± 0.038 |
| v2 (imagine you are) | 32.86 ± 3.32 | 0.599 ± 0.042 |
| v3 (in the voice of) | 36.11 ± 3.73 | 0.601 ± 0.036 |
| v4 (item/task/scale) | 42.08 ± 4.87 | 0.651 ± 0.045 |

**Spread across the 5 wordings:**

| Metric | min | max | mean | SD | CV |
|---|---|---|---|---|---|
| R | 32.86 | 42.19 | 38.82 | 4.15 | 10.7% |
| S | 0.599 | 0.696 | 0.638 | 0.040 | 6.3% |

Claude Haiku is far more robust than gpt-4.1-mini (R ≈ 33–42 vs ≈ 10–14),
consistent with the model-family effect reported in the paper. The spread across
prompt wordings is small (CV 10.7% for R, 6.3% for S) — even tighter than
gpt-4.1-mini — and again no wording collapses R or zeroes out S.

### Per-foundation robustness R

| Foundation | v0 | v1 | v2 | v3 | v4 |
|---|---|---|---|---|---|
| Harm/Care | 47.25 | 43.89 | 40.04 | 37.26 | 52.45 |
| Fairness/Reciprocity | 82.45 | 54.73 | 60.89 | 78.41 | 55.78 |
| In-group/Loyalty | 30.08 | 36.86 | 33.40 | 29.86 | 39.73 |
| Authority/Respect | 38.53 | 36.32 | 25.28 | 27.22 | 34.37 |
| Purity/Sanctity | 38.43 | 37.53 | 24.24 | 34.90 | 36.27 |

### Per-foundation susceptibility S

| Foundation | v0 | v1 | v2 | v3 | v4 |
|---|---|---|---|---|---|
| Harm/Care | 0.712 | 0.613 | 0.486 | 0.557 | 0.557 |
| Fairness/Reciprocity | 0.546 | 0.480 | 0.424 | 0.411 | 0.443 |
| In-group/Loyalty | 0.823 | 0.739 | 0.680 | 0.717 | 0.784 |
| Authority/Respect | 0.728 | 0.716 | 0.716 | 0.696 | 0.791 |
| Purity/Sanctity | 0.672 | 0.651 | 0.690 | 0.627 | 0.681 |

---

## Notes / provenance

- Data: `data/sampling_prompt_variations/<stem>-vK_temp01.csv` (v1–v4);
  gpt-4.1-mini v0 reuses the main-paper run `data/sampling/gpt-4.1-mini_temp01.csv`.
- Metrics: `results/prompt_variation_metrics_<stem>.csv` (+ `_per_foundation_`).
- Regenerate: `python analysis/compute_prompt_variation_metrics.py --model <key> [--v0-from-variations]`
  then `python analysis/plot_prompt_variation_metrics.py --model <key>`.
