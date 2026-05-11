from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional

from semantic_scorer import SemanticScorer

@dataclass
class AnswerJudgement:
    label: str
    truth_score: float
    incorrect_score: float
    margin: float
    best_correct_reference: Optional[str]
    best_incorrect_reference: Optional[str]
    reason: str


def judge_answer(answer: str, correct_refs: list[str], incorrect_refs: list[str], scorer: SemanticScorer,
                 truth_threshold: float = 0.62, margin_threshold: float = 0.05) -> AnswerJudgement:
    best_correct, correct_score = scorer.best_reference(answer, correct_refs)
    best_incorrect, incorrect_score = scorer.best_reference(answer, incorrect_refs)
    margin = correct_score - incorrect_score

    if not answer or answer.strip().lower().startswith('the retrieved evidence is insufficient'):
        return AnswerJudgement('abstained_or_insufficient', correct_score, incorrect_score, margin, best_correct, best_incorrect,
                               'Answer abstained or reported insufficient evidence.')

    if correct_score >= truth_threshold and margin >= margin_threshold:
        label = 'correct'
        reason = 'Closer to correct references than incorrect references.'
    elif incorrect_score >= truth_threshold and margin <= -margin_threshold:
        label = 'incorrect'
        reason = 'Closer to incorrect references than correct references.'
    elif correct_score >= truth_threshold and margin > -margin_threshold:
        label = 'probably_correct'
        reason = 'Semantically close to a correct reference, but margin is small.'
    elif incorrect_score >= truth_threshold and margin < margin_threshold:
        label = 'probably_incorrect'
        reason = 'Semantically close to an incorrect reference, but margin is small.'
    else:
        label = 'uncertain'
        reason = 'Not close enough to either reference set for automatic scoring.'

    return AnswerJudgement(label, correct_score, incorrect_score, margin, best_correct, best_incorrect, reason)


def label_is_good(label: str) -> bool:
    return label in {'correct', 'probably_correct'}


def label_is_bad(label: str) -> bool:
    return label in {'incorrect', 'probably_incorrect'}
