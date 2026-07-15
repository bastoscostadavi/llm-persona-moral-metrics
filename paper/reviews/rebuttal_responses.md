# Rebuttal — draft responses

Working draft. Points marked **[TODO]** depend on deliverables still in progress
(additional moral instrument, prompt-paraphrase rerun, raw-SD/log-scale reporting,
BFI persona control, language softening, Kwon et al. comparison, model citations,
scale normalization). Points marked **[DONE]** are ready to paste.

Common evidence referenced below:
- **Persona-set sensitivity.** The error bars in Fig. 3 are persona-bootstrap
  standard errors, so they already quantify how much R and S depend on the
  particular personas drawn (S SE ≈ 0.03–0.04; R relative error a few percent).
  Resampling 200 random 50-persona halves and recomputing both metrics leaves the
  model ranking essentially unchanged (Spearman ρ = 0.99 for R, 0.90 for S), with a
  median across-subsample coefficient of variation ≈ 4%.
- **Which personas drive the shift.** A new appendix figure ranks all 100 personas
  by their average distance from the no-persona baseline (averaged over the 15
  models).

---

## Reviewer 1 (Overall 2.5 — Borderline)

**R1.1 — Metrics not calibrated to a common range.** R and S deliberately measure
different quantities (within- vs. across-persona variation) and are not intended as
a single comparable score. We will additionally report a scale-free version
normalized by the 0–5 rating range so both are expressed as fractions of full
scale. **[TODO]**

**R1.2 — Conclusions read as conjecture.** We will soften the pre-/post-training
language throughout to frame these as empirically supported hypotheses rather than
mechanistic or causal claims (see R3.4). **[TODO]**

**R1.3 — Personas may be too terse / source not peer-reviewed.** The personas are
drawn from the large-scale Persona Hub set of \citet{ge2025scalingsyntheticdatacreation},
now widely used. Empirically they are far from inert: the new persona-distance
figure shows they induce substantial and heterogeneous shifts away from the
no-persona baseline. **[DONE — cite new figure]**

**R1.4 — Missing related work (Smith-Vaniz et al. 2025; Wang et al. 2025).** We will
add and discuss both. **[TODO]**

---

## Reviewer 2 (Overall 3 — Findings)

**R2.1 — Only 100 randomly drawn personas; no justification for the count.** **[DONE]**
> The error bars in Fig. 3 are persona-bootstrap standard errors and therefore
> already quantify how much R and S depend on the specific personas sampled; they
> are small (S standard error ≈ 0.03–0.04; a few percent relative error for R). To
> address the set-size question directly, we resampled 200 random 50-persona halves
> of the set and recomputed both metrics: the induced model ranking is essentially
> unchanged (Spearman ρ = 0.99 for robustness and 0.90 for susceptibility) and the
> median across-subsample coefficient of variation is ≈ 4%. The reported values are
> thus not an artifact of the particular 100-persona draw.

**R2.2 — Only one moral instrument (MFQ).** We are reproducing R and S on a second
moral instrument for four models to test whether the qualitative pattern holds.
**[TODO]**

**R2.3 — Paraphrasing/prompt-robustness not studied.** We rerun the fastest model
with rephrased prompt templates and report R and S for each variant. **[TODO]**

**R2.4 — Relation to Kwon et al., "Dropouts in Confidence" (AAAI 2026).** We will add
a comparison: shared observations (family-level effects, logit-based estimation) and
differences (our explicit separation of within- vs. across-persona variation and the
persona-conditioned MFQ design). **[TODO]**

**R2.5 — Missing references for the evaluated models.** We will add citations for all
benchmarked models. **[TODO]**

**R2.6 — Ordinal structure of MFQ ratings (4 is closer to 5 than to 0).** Both metrics
already operate on the numeric 0–5 Likert scale and treat rating differences
metrically (means and standard deviations), so ordinal proximity is respected. We
will additionally report the scale-free normalization by the 0–5 range. **[TODO]**

---

## Reviewer 3 (Overall 2 — Resubmit)

**R3.1 — Construct validity / uneven moral salience / which personas drive MFQ
shifts.** **[DONE — new appendix figure]**
> We add an appendix analysis that enumerates all 100 personas by their average
> distance from the model's no-persona baseline. The shift is distributed across the
> set rather than driven by a small clique of personas (the distance curve declines
> smoothly with no cliff), and the strongest movers are consistent across the 15
> models. This makes explicit which personas are most morally diagnostic and shows
> the susceptibility signal is not an artifact of a few outlier personas.

**R3.2 — Robustness metric fragility (reciprocal amplifies small variance; report raw
SD, log scale, non-moral control).** We will (i) report the raw average within-persona
standard deviation σ̄ alongside R = 1/σ̄, (ii) present R on a log scale, and (iii) add
a non-moral control using the BFI with the same persona set to disentangle a generic
persona-susceptibility from a specifically moral one. **[TODO]**

**R3.3 — Parsing failures and per-model persona exclusion → comparability.** We will
report the number of retained personas per model and a sensitivity analysis computed
on the common intersection of valid personas across models. The 50-persona
subsample result above already indicates the ranking is stable to differences in
which personas are included. **[TODO — add retained-count table + intersection check]**

**R3.4 — Causal overreach on pre- vs. post-training.** We will soften phrasing such as
"robustness is primarily determined by post-training" to "is consistent with" /
"suggests the hypothesis that" in the abstract, discussion, and conclusion. **[TODO]**

**R3.5 — MFQ-specific "moral behavior" wording is too broad.** We will consistently
distinguish "moral judgments expressed in survey responses" from broader "moral
behavior" and narrow the corresponding claims. **[TODO]**
