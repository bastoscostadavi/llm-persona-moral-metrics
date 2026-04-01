Official Review of Submission18582 by Reviewer z5bY
Official Reviewby Reviewer z5bY05 Mar 2026, 21:08 (modified: 24 Mar 2026, 10:44)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer z5bYRevisions
Summary:
This paper proposes a framework to assess the moral susceptibility and robustness when using persona role-play in LLMs. The paper introduces Moral robustness to assess within -persona variability and moral susceptibility to measure across-persona variability. Using these two metrics, multiple model families, including Gemini and GPT-4 models were analyzed. The main contribution of this paper is the extensive experiments conducted on recent LLMs.

Strengths And Weaknesses:

Presentation

The extensive experiments on recent LLMs and their results are helpful for readers who are interested in the moral foundations of LLMs.
Originality

The attempt to formalize moral susceptibility and moral robustness to measure the within- and across-persona variability is new.

Soundness

 The metrics for both moral susceptibility and moral robustness yield relative scores rather than absolute ones. This relative nature can confuse readers and severely hinders practical application. Because the normalization factors rely on the average performance across all evaluated models, adding or removing an LLM from the evaluation pool mathematically changes the scores of all other models. This instability makes the proposed metrics difficult to interpret and limits their utility as a reliable, standalone benchmark.

 The mathematical formalization of moral susceptibility requires improvement. The metric does not measure the total across-persona susceptibility; instead, it calculates the mean of the variability within arbitrary groups. As noted in Appendix C, the 91 valid personas were partitioned into 7 disjoint groups of 13 personas. This arbitrary grouping naturally introduces sampling noise, as the specific combination of personas randomly assigned to a group affects variance, and consequently, the final susceptibility score.

 As noted in Line 238, and Appendix C, when models failed to answer in the exact numeric format after multiple retries, those personas were completely excluded from the analysis. This introduces a flaw to the benchmark, as these “failures” may actually reflect the models’ built-in safety guardrails or alignment tendencies reacting to sensitive or controversial personas. A more rigorous approach would require a careful analysis of these refusal behaviors rather than simply discarding them. Consequently, the current framework likely suffers from survivorship bias, evaluating only the personas that the models felt “safe” enough to rate.

Presentation

While the extensive experimental results are highly informative, Figures 6 and 7 are currently visually overwhelming and consume an excessive amount of space. It would be more effective to present only the core, aggregated results in the main text. This would free up valuable space to provide deeper, more meaningful analysis within the manuscript itself.
Soundness: 1: poor
Presentation: 1: poor
Significance: 1: poor
Originality: 3: good
Key Questions For Authors:
Q1. Why are the moral susceptibility and robustness metrics designed to be relative, depending entirely on the specific pool of evaluated LLMs? Could the authors justify why an absolute scoring metric was not used, and discuss how relative nature impacts the benchmark’s real-world usability?

Q2. How exactly are the personas divided into groups when calculating moral susceptibility? The manuscript states that 91 personas were divided into 7 groups of 13, but it is unclear whether this assignment is purely random, or algorithmic. Furthermore, have the authors investigated how sensitive the final susceptibility scores are to this specific group formation? It seems that random combinations of personas could induce noise into the final score.

Q3. Could the authors more explicitly clarify why measuring “moral susceptibility” and “moral robustness” is a critical problem? Specifically, what precise limitations in existing moral evaluation literatures does this framework solve?

Limitations:
yes

Overall Recommendation: 2: Reject: For instance, a paper with technical flaws, weak evaluation, inadequate reproducibility, incompletely addressed ethical considerations, or writing so poor that it is not possible to understand its key claims.
Confidence: 4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.
Compliance With LLM Reviewing Policy: Affirmed.
Code Of Conduct Acknowledgement: Affirmed.