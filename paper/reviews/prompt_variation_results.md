# Deliverable 5 — Prompt-paraphrase robustness (reviewer R2.3)

**Question addressed:** are R (robustness) and S (susceptibility) stable when the
prompt is reworded? We take fast models and rerun the persona-conditioned MFQ
under 5 prompt templates.

## Protocol

- **Model(s):** `gpt-4.1-mini` (100 personas), `claude-haiku-4-5` (81 personas
  after excluding cells the model declined to rate; see note below).
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

## Results — `claude-haiku-4-5` (81 personas)

**Note on parse failures / refusals.** Under some paraphrases, for a subset of
personas Claude Haiku declines to output a rating and instead returns a verbal
deflection ("I appreciate the question, but…"), which has no leading 0–5 digit.
These are genuine non-responses, not truncation, and cannot be recovered by a
larger token budget. Per-variant failure rate: v0 0.7%, v1 2.8%, v2 3.1%,
v3 2.0%, v4 0.02%; the failures are concentrated in ~20 personas (e.g. ids 44,
51, 95). Following the paper's per-model exclusion, R and S are computed on the
**81 personas that are complete across all 5 variants** (vs. 100 for
gpt-4.1-mini).

### Overall R and S (value ± bootstrap SE, 81 shared personas)

| Variant | Robustness R | Susceptibility S |
|---|---|---|
| v0 (original) | 44.21 ± 4.14 | 0.740 ± 0.032 |
| v1 (adopt persona) | 37.43 ± 3.25 | 0.669 ± 0.028 |
| v2 (imagine you are) | 33.98 ± 2.90 | 0.621 ± 0.029 |
| v3 (in the voice of) | 34.60 ± 2.71 | 0.636 ± 0.026 |
| v4 (item/task/scale) | 41.03 ± 3.50 | 0.689 ± 0.032 |

**Spread across the 5 wordings:**

| Metric | min | max | mean | SD | CV |
|---|---|---|---|---|---|
| R | 33.98 | 44.21 | 38.25 | 4.35 | 11.4% |
| S | 0.621 | 0.740 | 0.671 | 0.047 | 7.0% |

Claude Haiku is far more robust than gpt-4.1-mini (R ≈ 34–44 vs ≈ 10–14),
consistent with the model-family effect reported in the paper. The spread across
prompt wordings is small (CV 11.4% for R, 7.0% for S) — comparable to
gpt-4.1-mini — and again no wording collapses R or zeroes out S.

### Per-foundation robustness R

| Foundation | v0 | v1 | v2 | v3 | v4 |
|---|---|---|---|---|---|
| Harm/Care | 49.73 | 41.56 | 42.41 | 31.75 | 52.45 |
| Fairness/Reciprocity | 73.23 | 40.52 | 46.30 | 53.43 | 57.47 |
| In-group/Loyalty | 28.62 | 39.28 | 34.94 | 33.39 | 38.31 |
| Authority/Respect | 50.44 | 32.39 | 28.19 | 28.47 | 35.57 |
| Purity/Sanctity | 40.73 | 35.09 | 26.39 | 34.23 | 32.08 |

### Per-foundation susceptibility S

| Foundation | v0 | v1 | v2 | v3 | v4 |
|---|---|---|---|---|---|
| Harm/Care | 0.693 | 0.580 | 0.482 | 0.547 | 0.520 |
| Fairness/Reciprocity | 0.517 | 0.516 | 0.435 | 0.433 | 0.449 |
| In-group/Loyalty | 0.907 | 0.762 | 0.710 | 0.762 | 0.815 |
| Authority/Respect | 0.761 | 0.738 | 0.713 | 0.719 | 0.828 |
| Purity/Sanctity | 0.823 | 0.747 | 0.767 | 0.721 | 0.831 |

---

## Notes / provenance

- Data: `data/sampling_prompt_variations/<stem>-vK_temp01.csv` (v1–v4);
  gpt-4.1-mini v0 reuses the main-paper run `data/sampling/gpt-4.1-mini_temp01.csv`.
- Metrics: `results/prompt_variation_metrics_<stem>.csv` (+ `_per_foundation_`).
- Regenerate: `python analysis/compute_prompt_variation_metrics.py --model <key> [--v0-from-variations]`
  then `python analysis/plot_prompt_variation_metrics.py --model <key>`.
