# Second moral instrument: Oxford Utilitarianism Scale (OUS)

**Addresses:** Reviewer 2 ("the analysis is performed on only one moral benchmark"; "expand beyond the MFQ") and Reviewer 3 ("the benchmark depends heavily on MFQ ... it is not obvious that robustness and susceptibility on MFQ generalize").

## What we did

We re-ran the benchmark on a **second, independent moral instrument** — the **Oxford Utilitarianism Scale** (OUS; Kahane et al., 2018, *Psychological Review*). The OUS is grounded in a *different* moral theory (consequentialism / utilitarianism) than the MFQ's Moral Foundations Theory, and is organized into two validated subscales: **Instrumental Harm (IH)**, items 2/4/6/8, and **Impartial Beneficence (IB)**, items 1/3/5/7/9.

The collection protocol is **identical to the MFQ runs** — same persona set (100 personas), same 0–5 agreement response format, same repeated sampling (`n = 10` per persona–item cell), same temperature (`T = 0.1`), and the *same* robustness/susceptibility estimator (persona- and rerun-bootstrap). **Only the item content differs**, so the two instruments are directly comparable.

We report **six models** that are already in the MFQ benchmark, chosen so that the **rankings can be checked for reproduction** on the new instrument (same providers as in the paper): Claude Haiku 4.5, DeepSeek V3, Gemini 2.5 Flash Lite, GPT-4.1 Mini, GPT-4.1 Nano, and Llama 4 Scout.

## Result 1 — Overall robustness and susceptibility (MFQ vs. OUS)

Values are mean ± standard error at `T = 0.1`.

| Model | R (MFQ) | R (OUS) | S (MFQ) | S (OUS) |
|---|---|---|---|---|
| Claude Haiku 4.5 | 92.04 ± 10.72 | **65.18 ± 13.95** | 0.747 ± 0.028 | 0.506 ± 0.016 |
| Gemini 2.5 Flash Lite | 27.73 ± 2.10 | 37.70 ± 5.97 | 0.809 ± 0.032 | 0.634 ± 0.021 |
| GPT-4.1 Nano | 12.28 ± 0.77 | 8.91 ± 0.62 | 0.723 ± 0.047 | 0.765 ± 0.039 |
| GPT-4.1 Mini | 11.31 ± 0.46 | 7.86 ± 0.49 | 0.827 ± 0.037 | 0.629 ± 0.023 |
| Llama 4 Scout | 4.59 ± 0.16 | 7.88 ± 0.56 | 0.667 ± 0.028 | 0.391 ± 0.020 |
| DeepSeek V3 | 3.27 ± 0.07 | 2.49 ± 0.08 | 0.698 ± 0.034 | 0.441 ± 0.019 |

## Result 2 — The main finding reproduces

**Robustness (R):** the paper's central pattern reproduces almost exactly. The rank correlation across the six models is **Spearman ρ = 0.943 (p = 0.005)**. Robustness spans more than an order of magnitude and is strongly separated by model — Claude Haiku 4.5 is the most robust and DeepSeek V3 the least on *both* instruments, with Gemini 2.5 Flash Lite second and GPT-4.1 Nano third on both.

| Metric | MFQ ranking (high → low) | OUS ranking (high → low) | Spearman ρ |
|---|---|---|---|
| Robustness | Haiku > Gemini-FL > Nano > Mini > Scout > DeepSeek | Haiku > Gemini-FL > Nano > Scout > Mini > DeepSeek | **0.943** (p=0.005) |
| Susceptibility | Mini > Gemini-FL > Haiku > Nano > DeepSeek > Scout | Nano > Gemini-FL > Mini > Haiku > DeepSeek > Scout | 0.600 (p=0.21) |

The only reordering in robustness is a Mini↔Scout swap, and those two models are statistically indistinguishable (OUS R = 7.86 ± 0.49 vs. 7.88 ± 0.56). The large, well-separated gaps — Claude at the top, DeepSeek at the bottom — are stable across both instruments.

**Susceptibility (S):** as in the paper, S varies over a **much narrower range** and is **less strongly structured** than R. On both instruments S stays within ≈ 0.4–0.8, the correlation is positive but weaker and not significant at n = 6 (ρ = 0.60, p = 0.21), and the extreme models are consistent (Gemini 2.5 Flash Lite high, Llama 4 Scout lowest on both). This matches the paper's claim that susceptibility is not primarily a family-level property, in contrast to robustness.

**Takeaway:** the two metrics behave the same way on a moral instrument built on an entirely different moral theory — robustness is a large, stable, strongly-separating quantity (ρ = 0.94), while susceptibility is compressed and only weakly ordered — supporting the generality of the metrics beyond the MFQ.

## Result 3 — Subscale (IH / IB) decomposition on OUS

The OUS subscales play the role that the five foundations play for the MFQ.

| Model | R (Instrumental Harm) | R (Impartial Beneficence) | S (IH) | S (IB) |
|---|---|---|---|---|
| Gemini 2.5 Flash Lite | 94.40 | 25.46 | 0.605 | 0.658 |
| Claude Haiku 4.5 | 56.10 | 74.88 | 0.498 | 0.512 |
| Llama 4 Scout | 21.21 | 5.24 | 0.232 | 0.519 |
| GPT-4.1 Nano | 6.57 | 12.46 | 0.857 | 0.691 |
| GPT-4.1 Mini | 7.20 | 8.48 | 0.669 | 0.597 |
| DeepSeek V3 | 1.80 | 3.58 | 0.531 | 0.369 |

The decomposition is informative on the new instrument and surfaces model-specific structure: some models are far more robust on Instrumental-Harm items than on Impartial-Beneficence items (Gemini 2.5 Flash Lite, Llama 4 Scout), while others show the opposite (Claude Haiku, GPT-4.1 Nano, DeepSeek V3).

## Result 4 — Data quality and per-model persona retention

This directly addresses **Reviewer 3's** request to *"report the number of retained personas per model"* and to be explicit about per-model exclusion.

| Model | Personas attempted | Personas retained | Parse failures (of 9000) |
|---|---|---|---|
| GPT-4.1 Mini | 100 | 100 | 0 (0.00%) |
| Llama 4 Scout | 100 | 100 | 0 (0.00%) |
| DeepSeek V3 | 100 | 100 | 0 (0.00%) |
| GPT-4.1 Nano | 100 | 99 | 14 (0.16%) |
| Gemini 2.5 Flash Lite | 100 | 99 | 26 (0.29%) |
| Claude Haiku 4.5 | 100 | 92 | 173 (1.92%) |

Each collection was re-run (resume passes) to backfill transient failures; Llama 4 Scout, for example, went from 702 → 0 failures across three passes.

**A substantive observation about the residual failures.** The remaining failures for Claude Haiku (173) and Gemini 2.5 Flash Lite (26) are **not** parsing noise — they are **refusals concentrated on the Instrumental-Harm items** (e.g., for Claude Haiku, item 6 "torture an innocent person to prevent a bomb": 71 failures; items 2 and 8: 40 each). Instead of returning a numeric agreement rating, the model declines and steps out of the persona (e.g., *"I appreciate the question, but I need to step out of the persona here … the moral weight of this topic is too …"*). These are deterministic at `T = 0.1` and cannot be recovered by re-running or by allowing longer outputs (longer generations either still refuse or introduce spurious digits from the reasoning text). We therefore keep the protocol fixed and let the pipeline drop the affected cells, exactly as for the MFQ. This refusal behavior is itself consistent with the paper's interpretation that **post-training shapes moral response behavior**: the models that refuse to numerically endorse instrumental harm are the more safety-tuned ones.

## Reproducibility

All OUS code and data mirror the MFQ pipeline:

- Instrument: `ous_questions.py` (9 items, IH/IB subscales)
- Collection: `run_ous_sampling.py` → `data/sampling_ous/<model>_temp01.csv`
- Metrics: `analysis/compute_ous_metrics.py` → `results/ous_persona_moral_metrics.csv`, `results/ous_persona_moral_metrics_per_subscale.csv`
- Plots: `analysis/plot_ous_metrics.py`, `analysis/plot_ous_vs_mfq.py`
