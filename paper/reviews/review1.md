Official Review of Submission18582 by Reviewer Z5VP
Official Reviewby Reviewer Z5VP18 Mar 2026, 22:22 (modified: 24 Mar 2026, 10:44)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer Z5VPRevisions
Summary:
The work aims to measure susceptibility, how much do models change their opinions based on different prompting personas and robustness, and how consistent are model responses when prompted with the same information. The paper claims that moral susceptibility is mildly correlated to model family and that there’s a trend within families in which the scaling of a model correlates clearly with the increase of susceptibility and that robustness is strongly driven by model family. They also claim that robustness and susceptibility are positively correlated when compared between families.

Strengths And Weaknesses:
Strengths

Model family is mildly correlated to the susceptibility claim is valid.
Paper is well written, other than a few typos, and easy to understand.
High number of data collected.
Weaknesses

The framework of Moral Foundations chosen to do the ethical analysis is contested within the psychological/behavioral community and there’s no acknowledgement of that in the paper. Here are two references but there are many more works contesting the framework:
Harper, Craig A., and Darren Rhodes. "Reanalysing the factor structure of the moral foundations questionnaire." British Journal of Social Psychology 60.4 (2021): 1303-1329.
Zakharin, Michael, and Timothy C. Bates. "Remapping the foundations of morality: Well-fitting structural model of the Moral Foundations Questionnaire." PloS one 16.10 (2021): e0258910.
Relevancy of table 1 and Figure 2? In table 1 all of the personas seem to be the persona that’s the most fitting to this prompt. I understand the paper proposes the concept of moral susceptibility in order to remove the confounding factors between the effects of prompting and the underlying susceptibility of the model. This table adds confusion, there’s no clarification about the confounding factors between prompting and MFQ results.
Here’s an example if one is measuring In-group/Loyalty then the persona that scored the best at this foundation was 75 which the prompt says “an ultrAslan fan, the hardcore fan group of Galatasaray SK”, which seems to me like the equivalent of pre-prompting the model to be better at being loyal, by specifically telling it you’re a fan of this group. Here’s another example in the Fairness/Reciprocity we have persona #27 as the highest scoring personality which has the prompt “A factory worker who is battling for compensation after being injured on the job due to negligence”.
A similar argument can be made for Fig 2.
The scale of size within families should clearly show an increase in susceptibility but that’s not clear from the data presented in Figure 5.
Both claude models show the same susceptibility score: 0.49
Gpt-4o and gpt-4.1-mini also have the same: 0.51
Grok-4 and grok-4-fast: it’s inversely correlated 0.51 for the first and 0.54 for the later
Other models vary +- 0.04, which is not significant enough to argue for a clear increase in susceptibility.(Error bars are +- 0.01)
Soundness: 2: fair
Presentation: 3: good
Significance: 2: fair
Originality: 2: fair
Key Questions For Authors:
Reported correlations in table 3 don't reach statistical significance. Especially the "Family" column. (Pearson correlation with 3 degrees of freedom at p < 0.05 is around 0.88)
Justification for why using the Moral Foundations framework and/or acknowledgement that’s not broadly recognized as a way of measuring morality. I lack the understanding of why this work is relevant.
Justification for Table 1 and Fig 2.
Justification for why figure 5 shows clear scaling = susceptibility?
Limitations:
Yes

Overall Recommendation: 2: Reject: For instance, a paper with technical flaws, weak evaluation, inadequate reproducibility, incompletely addressed ethical considerations, or writing so poor that it is not possible to understand its key claims.
Confidence: 4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.
Compliance With LLM Reviewing Policy: Affirmed.
Code Of Conduct Acknowledgement: Affirmed.