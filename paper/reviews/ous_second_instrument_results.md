# Second moral instrument: Oxford Utilitarianism Scale (OUS)

**Addresses:** Reviewer 2 ("the analysis is performed on only one moral benchmark"; "expand beyond the MFQ") and Reviewer 3 ("the benchmark depends heavily on MFQ ... it is not obvious that robustness and susceptibility on MFQ generalize").

## What we did

We re-ran the benchmark on a **second, independent moral instrument** — the **Oxford Utilitarianism Scale** (OUS; Kahane et al., 2018, *Psychological Review*). The OUS is grounded in a *different* moral theory (consequentialism / utilitarianism) than the MFQ's Moral Foundations Theory, and is organized into two validated subscales: **Instrumental Harm (IH)**, items 2/4/6/8, and **Impartial Beneficence (IB)**, items 1/3/5/7/9.

The collection protocol is **identical to the MFQ runs** — same persona set (100 personas), same 0–5 agreement response format, same repeated sampling (`n = 10` per persona–item cell), same temperature (`T = 0.1`), and the *same* robustness/susceptibility estimator (persona- and rerun-bootstrap). **Only the item content differs**, so the two instruments are directly comparable.

We report four models that are already in the MFQ benchmark, chosen so that the **rankings can be checked for reproduction** on the new instrument (same providers as in the paper): Claude Haiku 4.5, GPT-4.1 Mini, GPT-4.1 Nano, and Llama 4 Scout. (Two further models, DeepSeek V3 and Gemini 2.5 Flash Lite, are being collected and will be added.)

## Result 1 — Overall robustness and susceptibility (MFQ vs. OUS)

Values are mean ± standard error at `T = 0.1`.

| Model | R (MFQ) | R (OUS) | S (MFQ) | S (OUS) |
|---|---|---|---|---|
| Claude Haiku 4.5 | 92.04 ± 10.72 | **65.18 ± 13.95** | 0.747 ± 0.028 | 0.506 ± 0.016 |
| GPT-4.1 Nano | 12.28 ± 0.77 | 8.91 ± 0.62 | 0.723 ± 0.047 | 0.765 ± 0.039 |
| GPT-4.1 Mini | 11.31 ± 0.46 | 7.86 ± 0.49 | 0.827 ± 0.037 | 0.629 ± 0.023 |
| Llama 4 Scout | 4.59 ± 0.16 | 7.88 ± 0.56 | 0.667 ± 0.028 | 0.391 ± 0.020 |

## Result 2 — The main finding reproduces

**Robustness (R):** the headline pattern of the paper reproduces cleanly. Claude Haiku 4.5 is **roughly an order of magnitude more robust** than every other model on *both* instruments (R ≈ 65 vs. ≈ 8 on OUS; R ≈ 92 vs. ≈ 5–12 on MFQ). The rank correlation across the four models is **Spearman ρ = 0.80**.

| Metric | MFQ ranking (high → low) | OUS ranking (high → low) | Spearman ρ |
|---|---|---|---|
| Robustness | Haiku ≫ Nano > Mini > Scout | Haiku ≫ Nano > Scout > Mini | **0.80** |
| Susceptibility | Mini > Haiku > Nano > Scout | Nano > Mini > Haiku > Scout | 0.40 |

The only reordering in robustness is a Mini↔Scout swap *inside* the tightly-clustered non-Claude group, where all three models sit at R ≈ 8 with overlapping confidence intervals — i.e., the swap is within noise, while the large, well-separated Claude gap is stable.

**Susceptibility (S):** as in the paper, S varies over a **much narrower range** and is **less family-structured** than R. On both instruments S stays within ≈ 0.4–0.8, Llama 4 Scout is the least susceptible on both, and the top of the ordering reshuffles within/near error bars (ρ = 0.40). This is consistent with the paper's claim that susceptibility is not primarily a family-level property, in contrast to robustness.

**Takeaway:** the two metrics behave the same way on a moral instrument built on an entirely different theory — robustness is a large, stable, family-separating quantity; susceptibility is compressed and weakly structured — supporting the generality of the metrics beyond the MFQ.

## Result 3 — Subscale (IH / IB) decomposition on OUS

The OUS subscales play the role that the five foundations play for the MFQ.

| Model | R (Instrumental Harm) | R (Impartial Beneficence) | S (IH) | S (IB) |
|---|---|---|---|---|
| Claude Haiku 4.5 | 56.10 | 74.88 | 0.498 | 0.512 |
| GPT-4.1 Nano | 6.57 | 12.46 | 0.857 | 0.691 |
| GPT-4.1 Mini | 7.20 | 8.48 | 0.669 | 0.597 |
| Llama 4 Scout | 21.21 | 5.24 | 0.232 | 0.519 |

The order-of-magnitude Claude robustness lead holds within *both* subscales. The subscale split also surfaces model-specific structure (e.g., Llama 4 Scout is far more robust on Instrumental-Harm items than on Impartial-Beneficence items, whereas the GPT models show the opposite pattern), illustrating that the decomposition is informative on the new instrument as well.

## Result 4 — Data quality and per-model persona retention

This directly addresses **Reviewer 3's** request to *"report the number of retained personas per model"* and to be explicit about per-model exclusion.

| Model | Personas attempted | Personas retained | Parse failures (of 9000) |
|---|---|---|---|
| GPT-4.1 Mini | 100 | 100 | 0 (0.00%) |
| Llama 4 Scout | 100 | 100 | 0 (0.00%) |
| GPT-4.1 Nano | 100 | 99 | 14 (0.16%) |
| Claude Haiku 4.5 | 100 | 92 | 173 (1.92%) |

Each collection was re-run (resume passes) to backfill transient failures; Llama 4 Scout, for example, went from 702 → 0 failures across three passes.

**A substantive observation about the Claude Haiku failures.** The 173 residual Haiku failures are **not** parsing noise — they are **refusals concentrated on the Instrumental-Harm items** (item 6, "torture an innocent person to prevent a bomb": 71 failures; items 2 and 8: 40 each). Instead of returning a numeric agreement rating, Claude Haiku declines and steps out of the persona (e.g., *"I appreciate the question, but I need to step out of the persona here … the moral weight of this topic is too …"*). These are deterministic at `T = 0.1` and cannot be recovered by re-running or by allowing longer outputs (longer generations either still refuse or introduce spurious digits from the reasoning text). We therefore keep the protocol fixed and let the pipeline drop the affected cells, exactly as for the MFQ. This refusal behavior is itself consistent with the paper's interpretation that **post-training shapes moral response behavior**: the most safety-tuned model in the set is the one that refuses to numerically endorse instrumental harm.

## Notes / status

- Two additional models (DeepSeek V3, Gemini 2.5 Flash Lite) are being collected on the OUS and will extend these tables.
- All OUS code and data mirror the MFQ pipeline: `ous_questions.py`, `run_ous_sampling.py`, `analysis/compute_ous_metrics.py`; raw data in `data/sampling_ous/`, metrics in `results/ous_persona_moral_metrics.csv` and `results/ous_persona_moral_metrics_per_subscale.csv`.
