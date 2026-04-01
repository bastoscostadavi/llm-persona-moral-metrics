Official Review of Submission18582 by Reviewer GA3z
Official Reviewby Reviewer GA3z07 Mar 2026, 14:43 (modified: 24 Mar 2026, 10:44)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer GA3zRevisions
Summary:
This paper studies how LLM moral judgments change under persona role-play. The authors combine persona prompting with the Moral Foundations Questionnaire (MFQ) and define two benchmark quantities: moral robustness, intended to capture within-persona stability under repeated sampling, and moral susceptibility, intended to capture across-persona sensitivity. The study evaluates 15 frontier models across 100 personas, 30 MFQ questions, and 10 repeated runs per persona-question pair, and reports both overall and foundation-specific analyses. The main findings are that model family explains much of the variation in robustness, larger models within a family tend to be more susceptible, and robustness and susceptibility are positively correlated overall.

Strengths And Weaknesses:
Strengths:

The paper tackles a timely and meaningful problem. Persona conditioning is increasingly common in LLM use, and studying how it affects moral responses is worthwhile. The decomposition into within-persona variability and across-persona variability is intuitive and gives the work a clear conceptual structure. The experimental sweep is also reasonably broad: the paper evaluates many contemporary models, uses 100 personas, and reports both aggregate and foundation-level results.
The empirical findings are interesting. In particular, the contrast between robustness being largely family-driven and susceptibility showing a within-family size trend is potentially useful for understanding how scaling and model lineage affect role-play behavior. The correlation analysis is also a nice addition, since it shows these two quantities are not trivially redundant.
Weaknesses: Generally speaking, the experiments are not enough for a benchmark paper. The current evaluation is conducted only on MFQ-style rating tasks, with 100 personas, 30 questions, and 10 repeated samplings, but it does not sufficiently test the robustness of the proposed benchmark itself. For example, there is no sensitivity analysis over prompt format, decoding temperature, persona selection, or the grouping strategy used to compute susceptibility. This is particularly important because the paper explicitly uses a fixed prompt template, a fixed temperature, and a group-based formulation for susceptibility, all of which could materially affect the reported scores.

Soundness: 4: excellent
Presentation: 2: fair
Significance: 2: fair
Originality: 2: fair
Key Questions For Authors:
For a benchmark paper, why is there no broader validation beyond MFQ-based ratings under a single prompting setup?
How robust are the benchmark conclusions to alternative prompt templates, temperatures, and output-parsing rules?
Can the authors demonstrate that the proposed metrics are stable under different persona subsets and grouping schemes?
What additional evidence would support the claim that this is a reliable benchmark rather than a task-specific evaluation protocol?
Limitations:
yes

Overall Recommendation: 2: Reject: For instance, a paper with technical flaws, weak evaluation, inadequate reproducibility, incompletely addressed ethical considerations, or writing so poor that it is not possible to understand its key claims.
Confidence: 4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.
Compliance With LLM Reviewing Policy: Affirmed.
Code Of Conduct Acknowledgement: Affirmed.