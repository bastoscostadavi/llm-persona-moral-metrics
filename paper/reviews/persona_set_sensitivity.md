# Persona-set sensitivity (tables for OpenReview)

Moral robustness (R) and susceptibility (S) are **aggregate** measures over the
persona set. Two concerns were raised: that the values might depend on the
particular 100 personas drawn (R2), and that per-model parsing-failure exclusion
leaves different models evaluated on slightly different persona sets (R3). We check
both directly. All numbers use the T=0.1 sampling data (100 personas × 30 MFQ
items × 10 repetitions per model).

---

## 1. Resampling the persona set (random 50-persona halves)

The error bars in Fig. 3 are **persona-bootstrap standard errors** and therefore
already quantify sensitivity to the persona sample. To make the set-size question
explicit, we drew 200 random 50-persona halves and recomputed both metrics.

**Table A. Full 100-persona value vs. 5–95% band over 200 random 50-persona subsamples.**

| Model | R (full) | R 5–95% (50-subset) | S (full) | S 5–95% (50-subset) |
|---|---|---|---|---|
| Claude Sonnet 4.5 | 107.7 | [90.6, 131.0] | 0.751 | [0.693, 0.798] |
| Claude Haiku 4.5 | 89.9 | [77.4, 106.4] | 0.747 | [0.690, 0.783] |
| Gemini 2.5 Flash Lite | 27.7 | [25.0, 31.2] | 0.809 | [0.747, 0.852] |
| GPT-4.1 | 14.5 | [13.3, 15.9] | 0.821 | [0.751, 0.873] |
| GPT-4o Mini | 12.9 | [11.9, 13.9] | 0.663 | [0.596, 0.713] |
| GPT-4.1 Nano | 12.3 | [11.1, 13.7] | 0.723 | [0.649, 0.787] |
| GPT-4.1 Mini | 11.3 | [10.6, 12.1] | 0.827 | [0.753, 0.888] |
| Gemini 2.5 Flash | 10.0 | [9.1, 10.9] | 1.043 | [0.962, 1.113] |
| GPT-4o | 9.8 | [9.2, 10.6] | 0.793 | [0.729, 0.845] |
| Llama 4 Scout | 4.6 | [4.3, 4.8] | 0.667 | [0.616, 0.707] |
| Llama 4 Maverick | 4.4 | [4.2, 4.6] | 0.706 | [0.669, 0.739] |
| DeepSeek V3.1 | 4.1 | [3.8, 4.4] | 0.794 | [0.729, 0.841] |
| Grok 4 Fast | 3.3 | [3.2, 3.5] | 0.915 | [0.837, 0.975] |
| Grok 4 | 3.3 | [3.2, 3.5] | 0.773 | [0.716, 0.821] |
| DeepSeek V3 | 3.3 | [3.2, 3.4] | 0.698 | [0.643, 0.747] |

- **Ranking preserved** under resampling: Spearman ρ = **0.989** for R (5th pct 0.979)
  and **0.901** for S (5th pct 0.804), over the 200 subsamples.
- **Median across-subsample coefficient of variation** ≈ **4%** for both metrics.

**Table B. The subsample spread just re-expresses the persona-bootstrap SE already in the paper**
(ratio ≈ 1.0), because halving the sample (√2 inflation) is offset by the
finite-population correction √(50/99).

| Model | S | persona-bootstrap SE (paper) | implied SD (50-subset band) | ratio |
|---|---|---|---|---|
| Claude Sonnet 4.5 | 0.751 | 0.0298 | 0.0320 | 1.07 |
| Gemini 2.5 Flash | 1.043 | 0.0436 | 0.0460 | 1.06 |
| GPT-4.1 | 0.821 | 0.0366 | 0.0372 | 1.01 |
| Grok 4 Fast | 0.915 | 0.0387 | 0.0417 | 1.08 |
| DeepSeek V3 | 0.698 | 0.0343 | 0.0317 | 0.92 |

---

## 2. Common persona set valid across all models (R3: comparability)

Parsing-failure exclusion is applied per model, so retained-persona counts differ
slightly (94–100 of 100). To rule out that this drives the rankings, we recompute R
and S for every model on the **92 personas that are valid across all 15 models**
and compare to each model's own retained set.

**Table C. R and S on each model's own retained set vs. the common 92-persona set.**

| Model | retained n | R (own set) | R (shared) | S (own set) | S (shared) |
|---|---|---|---|---|---|
| Claude Sonnet 4.5 | 100 | 107.7 | 108.4 | 0.751 | 0.747 |
| Claude Haiku 4.5 | 94 | 89.9 | 89.4 | 0.747 | 0.741 |
| Gemini 2.5 Flash Lite | 97 | 27.7 | 28.4 | 0.809 | 0.795 |
| GPT-4.1 | 100 | 14.5 | 14.8 | 0.821 | 0.815 |
| GPT-4o Mini | 100 | 12.9 | 13.6 | 0.663 | 0.638 |
| GPT-4.1 Nano | 100 | 12.3 | 12.5 | 0.723 | 0.678 |
| GPT-4.1 Mini | 100 | 11.3 | 11.7 | 0.827 | 0.810 |
| Gemini 2.5 Flash | 100 | 10.0 | 10.3 | 1.043 | 1.014 |
| GPT-4o | 100 | 9.8 | 10.0 | 0.793 | 0.780 |
| Llama 4 Scout | 100 | 4.6 | 4.7 | 0.667 | 0.641 |
| Llama 4 Maverick | 100 | 4.4 | 4.5 | 0.706 | 0.693 |
| DeepSeek V3.1 | 100 | 4.1 | 4.3 | 0.794 | 0.769 |
| Grok 4 Fast | 100 | 3.3 | 3.5 | 0.915 | 0.885 |
| Grok 4 | 100 | 3.3 | 3.4 | 0.773 | 0.759 |
| DeepSeek V3 | 100 | 3.3 | 3.3 | 0.698 | 0.691 |

- **R ranking is identical** on the shared set (Spearman = **1.000**; max |ΔR| = 0.74).
- **S ranking is essentially unchanged** (Spearman = **0.982**; max |ΔS| = 0.045; only
  adjacent near-ties reorder).

**Takeaway.** Neither the specific 100-persona draw nor the per-model persona
filtering drives the reported metrics or their model rankings; the values are
stable to both, consistent with the (small) persona-bootstrap standard errors
already reported.

---

## 3. Which personas drive the shift, and their moral salience (R3)

R3 noted that the personas differ in moral salience and asked for either
controlled personas or "an analysis of which personas actually drive MFQ shifts."
We provide the latter directly. For each persona we measure its shift from the
model's no-persona baseline, `d = sqrt( sum_q (mu_persona,q - mu_self,q)^2 )` over
the 30 MFQ items (for scale, a uniform 1-point shift on all items gives
`d = sqrt(30) ≈ 5.5`), and average it across the 15 models. The two figures below
summarize the result.

**Ranking of personas by shift (Fig. `persona_base_distance_ranking`).** The average
shift varies smoothly and substantially across the set, from `d ≈ 3.9` for the
least-moving persona to `d ≈ 10.8` for the most-moving one, with no small clique of
outliers dominating — so the susceptibility signal is broad-based rather than
carried by a few personas. The largest shifts are concentrated in morally and
ideologically committed personas (religion, tradition, authority, in-group
loyalty): the top movers include "a traditionalist who firmly believes Christmas
should be celebrated only in December," "a deeply religious family member," and
"a skeptical pastor," while the smallest shifts come from neutral academic and
professional personas (e.g. cognitive-neuroscience professors, a public-policy
student). This is the pattern MFQ construct validity would predict, and it makes
the "moral salience" of the set explicit rather than assumed.

Crucially, a persona's shift is **not** reliably predictable from an a-priori
judgment of moral salience, which is precisely why an empirical ranking is needed
rather than hand-labeling. Among R3's own cited examples, the intuitive ordering
only partly holds:

| Persona (as cited by R3) | R3's expectation | New id (0=least, 99=most) | Avg. shift `d` |
|---|---|---|---|
| Traditionalist about Christmas | morally loaded | p99 | 10.84 |
| Injured factory worker | morally loaded | p69 | 6.35 |
| Fair-trade official | morally loaded | p43 | 5.37 |
| Space-mission analyst | weakly moralized | p85 | 7.55 |
| Himalayan flora tour guide | weakly moralized | p54 | 5.82 |

The traditionalist is indeed the strongest mover, but the "weakly moralized"
space-mission analyst produces an above-median shift while the "morally loaded"
fair-trade official sits below the median. The data-driven ranking, not intuition,
is the reliable way to identify the personas that move MFQ responses.

**Shift across models (Fig. `persona_shift_by_model_heatmap`).** Persona
conditioning moves every one of the 15 models by a substantial amount (per-model
mean `d` between roughly 4.5 and 8.4, i.e. ~0.8–1.5 points per item on average), so
the effect is not confined to particular providers or decoding behaviors. Ordering
the columns by average shift produces a clear low-to-high gradient shared across
models: the personas that move the models most tend to do so across families, even
though models also differ in their persona-specific responses (this residual
model-specific variation is expected and is what moral susceptibility, as an
aggregate, is designed to summarize). Gray cells mark the per-model personas
dropped for parsing failures.
