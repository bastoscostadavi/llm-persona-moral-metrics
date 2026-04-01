Official Review of Submission18582 by Reviewer BnRP
Official Reviewby Reviewer BnRP13 Mar 2026, 04:12 (modified: 24 Mar 2026, 10:44)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer BnRPRevisions
Summary:
This paper investigates how persona role-play affects the moral judgments expressed by large language models. The authors combine the Moral Foundations Questionnaire (MFQ), a standard instrument from moral psychology measuring five moral foundations, with 100 persona descriptions to elicit moral ratings from 15 frontier LLMs across 6 model families. Each model responds to each persona–question pair 10 times at fixed temperature, yielding two variance-based metrics: moral robustness (consistency of responses within a persona across repeated sampling) and moral susceptibility (variation of responses across different personas). The main empirical findings are that model family is the primary factor explaining robustness differences, with Claude models substantially more robust than all others, while susceptibility shows a within-family size trend where larger variants tend to be more susceptible. The paper also reports baseline moral foundation profiles for each model without persona conditioning and average profiles per persona across models, and observes a positive correlation between robustness and susceptibility at both the model and family level.

Strengths And Weaknesses:
Strengths:

The paper tests 15 frontier models across 6 families (Claude, DeepSeek, Gemini, GPT-4, Grok, Llama) with both large and small variants within families.

People could easily reproduce the paper because it provides fixed temperature, single-question prompting, and consistent prompt templates across all models.

Weaknesses:

The contributions seems quite narrow for this paper. The paper’s contribution is basically a standardized measurement protocol (MFQ × personas × repetitions) plus two variance-based summary statistics, applied at scale. There's no new modeling, no intervention, no mechanistic insight into why models differ, and no actionable takeaway for alignment beyond "Claude is more consistent." The conceptual framing of "robustness vs. susceptibility" sounds novel but operationally it's just within-group vs. between-group variance. To me this may be closer to a workshop paper than a full conference contribution.

There is not too much to comment on the paper based on its contribution. One point is that the experiment fixes temperature to be 0.1 but doesn’t vary it, so I am not sure if this robustness behavior is just applicable at T=0.1 or any other temperatures.

Soundness: 2: fair
Presentation: 3: good
Significance: 2: fair
Originality: 2: fair
Key Questions For Authors:
You have per-persona, per-foundation, per-model scores. Have you checked whether personas with obviously relevant traits (religious backgrounds, authority figures, caregivers) shift the expected foundations?

The MFQ was validated on humans who have embodied moral experiences. When a model outputs "4" for a Purity/Sanctity question, what does that mean? Are you treating it as a measurement of the model's values, its prediction of what the persona would say, or something else?

Limitations:
Yes

Overall Recommendation: 2: Reject: For instance, a paper with technical flaws, weak evaluation, inadequate reproducibility, incompletely addressed ethical considerations, or writing so poor that it is not possible to understand its key claims.
Confidence: 4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.
Compliance With LLM Reviewing Policy: Affirmed.
Code Of Conduct Acknowledgement: Affirmed.