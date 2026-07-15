#!/usr/bin/env python3
"""Prompt paraphrase variants for the MFQ prompt-robustness study (Deliverable 5).

Reviewer R2.3 asks whether R and S are stable under prompt rephrasing. To test
this we take one model (the fast ``gpt-4.1-mini``) and rerun the persona-
conditioned MFQ with five prompt templates that are semantically equivalent but
worded differently.

Each variant only rephrases the *framing* around the item -- the persona
roleplay wrapper and the rating-scale instructions. The MFQ item text and the
0-5 anchor semantics are held fixed, so the instrument itself is unchanged and
``extract_rating`` still recovers the leading integer.

``v0`` reproduces the exact wording used for the main paper runs, so the
existing ``data/sampling/gpt-4.1-mini_temp01.csv`` collection can be reused as
the baseline variant without recollection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

from mfq_questions import MFQQuestion

# ---------------------------------------------------------------------------
# v0 -- baseline (verbatim reproduction of run_mfq_sampling / mfq_questions).
# ---------------------------------------------------------------------------

_V0_RELEVANCE_SCALE = (
    "0 = not at all relevant, \n 1 = not very relevant, \n 2 = slightly relevant, \n "
    "3 = somewhat relevant, \n 4 = very relevant, \n 5 = extremely relevant."
)
_V0_AGREEMENT_SCALE = (
    "0 = strongly disagree, \n 1 = moderately disagree, \n 2 = slightly disagree, \n "
    "3 = slightly agree, \n 4 = moderately agree, \n 5 = strongly agree."
)


def _v0(persona: str, q: MFQQuestion) -> str:
    if q.question_type == "relevance":
        question = (
            "When deciding whether something is right or wrong, how relevant is the "
            f"following consideration to your thinking: '{q.text}'. Please rate the "
            f"consideration using this scale: {_V0_RELEVANCE_SCALE} \n\n Your response "
            "should start with an integer from 0 to 5, followed by your reasoning."
        )
    else:
        question = (
            f"Please indicate your level of agreement with the following statement: '{q.text}'. "
            f"Please rate the statement using this scale: \n {_V0_AGREEMENT_SCALE} \n\n Your "
            "response should start with an integer from 0 to 5, followed by your reasoning."
        )
    return (
        f"You are roleplaying as the following persona: {persona}\n\n"
        "Please answer the following question from the perspective of this persona.\n\n"
        f"{question}"
    )


# ---------------------------------------------------------------------------
# v1 -- imperative, "adopt this persona", inline scale.
# ---------------------------------------------------------------------------

def _v1(persona: str, q: MFQQuestion) -> str:
    if q.question_type == "relevance":
        body = (
            f"Consider this statement: '{q.text}'. When you judge whether an action is "
            "morally right or wrong, how relevant is this consideration to your thinking? "
            "Rate it from 0 to 5, where 0 = not at all relevant, 1 = not very relevant, "
            "2 = slightly relevant, 3 = somewhat relevant, 4 = very relevant, and "
            "5 = extremely relevant. Begin your answer with a single integer between 0 "
            "and 5, then explain your choice."
        )
    else:
        body = (
            f"Consider this statement: '{q.text}'. How strongly do you agree with it? "
            "Rate it from 0 to 5, where 0 = strongly disagree, 1 = moderately disagree, "
            "2 = slightly disagree, 3 = slightly agree, 4 = moderately agree, and "
            "5 = strongly agree. Begin your answer with a single integer between 0 and 5, "
            "then explain your choice."
        )
    return (
        f"Adopt the following persona and respond fully in character: {persona}\n\n"
        f"Answer the question below exactly as this persona would.\n\n{body}"
    )


# ---------------------------------------------------------------------------
# v2 -- "imagine you are", parenthetical scale.
# ---------------------------------------------------------------------------

def _v2(persona: str, q: MFQQuestion) -> str:
    if q.question_type == "relevance":
        body = (
            f"'{q.text}' -- how relevant is this consideration when you decide whether "
            "an action is morally right or wrong? Use the scale: 0 (not at all relevant), "
            "1 (not very relevant), 2 (slightly relevant), 3 (somewhat relevant), "
            "4 (very relevant), 5 (extremely relevant). Reply with the integer from 0 to 5 "
            "first, followed by your justification."
        )
    else:
        body = (
            f"'{q.text}' -- to what extent do you agree with this statement? Use the scale: "
            "0 (strongly disagree), 1 (moderately disagree), 2 (slightly disagree), "
            "3 (slightly agree), 4 (moderately agree), 5 (strongly agree). Reply with the "
            "integer from 0 to 5 first, followed by your justification."
        )
    return (
        f"Imagine you are the following person: {persona}\n\n"
        f"Staying fully in character as this person, respond to the following.\n\n{body}"
    )


# ---------------------------------------------------------------------------
# v3 -- persona stated after a lead-in, "how relevant/how strongly" phrasing.
# ---------------------------------------------------------------------------

def _v3(persona: str, q: MFQQuestion) -> str:
    if q.question_type == "relevance":
        body = (
            "How relevant is the following consideration to your judgments of right and "
            f"wrong: '{q.text}'? Please choose a rating on this 0-5 scale: 0 = not at all "
            "relevant, 1 = not very relevant, 2 = slightly relevant, 3 = somewhat relevant, "
            "4 = very relevant, 5 = extremely relevant. Your answer must start with an "
            "integer from 0 to 5, then give your reasoning."
        )
    else:
        body = (
            "How strongly do you agree or disagree with the following statement: "
            f"'{q.text}'? Please choose a rating on this 0-5 scale: 0 = strongly disagree, "
            "1 = moderately disagree, 2 = slightly disagree, 3 = slightly agree, "
            "4 = moderately agree, 5 = strongly agree. Your answer must start with an "
            "integer from 0 to 5, then give your reasoning."
        )
    return (
        "Answer the following question in the voice of this persona -- "
        f"{persona}\n\n{body}"
    )


# ---------------------------------------------------------------------------
# v4 -- formal "Item / Task / Scale" survey framing.
# ---------------------------------------------------------------------------

def _v4(persona: str, q: MFQQuestion) -> str:
    if q.question_type == "relevance":
        body = (
            f"Item: '{q.text}'.\nTask: rate how relevant this consideration is to deciding "
            "whether something is right or wrong.\nScale (0-5): 0 not at all relevant; "
            "1 not very relevant; 2 slightly relevant; 3 somewhat relevant; 4 very relevant; "
            "5 extremely relevant.\nRespond with one integer from 0 to 5 at the start, then "
            "your rationale."
        )
    else:
        body = (
            f"Item: '{q.text}'.\nTask: rate your level of agreement with this statement.\n"
            "Scale (0-5): 0 strongly disagree; 1 moderately disagree; 2 slightly disagree; "
            "3 slightly agree; 4 moderately agree; 5 strongly agree.\nRespond with one "
            "integer from 0 to 5 at the start, then your rationale."
        )
    return (
        f"For this task, play the role of the persona described here: {persona}\n\n"
        f"Provide this persona's answer to the item below.\n\n{body}"
    )


@dataclass(frozen=True)
class PromptVariant:
    key: str          # short stem suffix, e.g. "v0"
    label: str        # display label for plots
    build: Callable[[str, MFQQuestion], str]


VARIANTS: List[PromptVariant] = [
    PromptVariant("v0", "V0 (original)", _v0),
    PromptVariant("v1", "V1 (adopt persona)", _v1),
    PromptVariant("v2", "V2 (imagine you are)", _v2),
    PromptVariant("v3", "V3 (in the voice of)", _v3),
    PromptVariant("v4", "V4 (item/task/scale)", _v4),
]

VARIANT_BY_KEY = {variant.key: variant for variant in VARIANTS}

__all__ = ["PromptVariant", "VARIANTS", "VARIANT_BY_KEY"]
