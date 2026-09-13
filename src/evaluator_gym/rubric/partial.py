"""Partial credit — evidence F1 and tier-1 field accuracy."""

from __future__ import annotations

from typing import Any

from evaluator_gym.rubric.types import RewardComponent


def evidence_set_f1(gt_tags: set[str], pred_tags: set[str]) -> tuple[float, float, float, list[str], list[str]]:
    if not gt_tags and not pred_tags:
        return 1.0, 1.0, 1.0, [], []
    precision = len(gt_tags & pred_tags) / len(pred_tags) if pred_tags else 0.0
    recall = len(gt_tags & pred_tags) / len(gt_tags) if gt_tags else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    missing = sorted(gt_tags - pred_tags)
    extra = sorted(pred_tags - gt_tags)
    return f1, precision, recall, missing, extra


def score_evidence_f1(gt: dict[str, Any], parsed: dict[str, Any]) -> RewardComponent:
    gt_tags = set(gt.get("evidence_set") or [])
    pred_tags = set(parsed.get("evidence_set") or [])
    f1, precision, recall, missing, extra = evidence_set_f1(gt_tags, pred_tags)
    return RewardComponent(
        score=f1,
        clauses=("§6", "§13"),
        detail=f"precision={precision:.4f} recall={recall:.4f}",
        extra={"missing": missing, "extra": extra},
    )


def score_field_accuracy(
    gt: dict[str, Any],
    parsed: dict[str, Any],
    expected_keys: tuple[str, ...] | list[str],
) -> tuple[RewardComponent, bool]:
    keys = tuple(expected_keys)
    if not keys:
        return RewardComponent(score=0.0, clauses=("§3",), detail="no expected keys"), True
    correct = sum(1 for k in keys if str(parsed.get(k, "")) == str(gt.get(k, "")))
    any_wrong = correct < len(keys)
    accuracy = correct / len(keys)
    return (
        RewardComponent(
            score=accuracy,
            clauses=("§3",),
            detail=f"{correct}/{len(keys)} fields correct",
            extra={"correct": correct, "total": len(keys), "any_field_wrong": any_wrong},
        ),
        any_wrong,
    )
