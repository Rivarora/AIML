"""Conservative, explainable advisory prompts from validated inputs.

These rules are intentionally simple and transparent. They are prompted by
values that are often worth discussing with a student or adviser; they are not
model output, causal claims, predictions, or institutional decisions.
"""
from __future__ import annotations

from typing import Any

ADVISORY_RULES: tuple[dict[str, Any], ...] = (
    {
        "field": "attendance",
        "label": "Attendance",
        "operator": "<",
        "threshold": 75,
        "unit": "%",
        "message": "Consider reviewing attendance barriers and a realistic plan to re-engage with classes.",
    },
    {
        "field": "previous_scores",
        "label": "Previous scores",
        "operator": "<",
        "threshold": 60,
        "unit": "points",
        "message": "Consider targeted support for earlier concepts before building new material.",
    },
    {
        "field": "communication_skills",
        "label": "Communication skills",
        "operator": "<",
        "threshold": 6,
        "unit": "/ 10",
        "message": "Consider structured communication practice with feedback, such as presentations or mock interviews.",
    },
    {
        "field": "projects_completed",
        "label": "Projects completed",
        "operator": "<",
        "threshold": 2,
        "unit": "projects",
        "message": "Consider adding a small, completed practical project that can be discussed with evidence.",
    },
    {
        "field": "hours_studied",
        "label": "Hours studied per week",
        "operator": "<",
        "threshold": 10,
        "unit": "hours",
        "message": "Consider a regular study schedule rather than occasional long sessions.",
    },
)


def advisory_rules() -> list[dict[str, Any]]:
    """Return a JSON-safe copy of the transparent advisory thresholds."""
    return [dict(rule) for rule in ADVISORY_RULES]


def build_interventions(values: dict[str, Any]) -> list[str]:
    """Return advisory prompts for the supplied validated numeric values."""
    suggestions: list[str] = []
    for rule in ADVISORY_RULES:
        field = rule["field"]
        if field not in values:
            continue
        try:
            value = float(values[field])
        except (TypeError, ValueError):
            continue
        if value < float(rule["threshold"]):
            suggestions.append(str(rule["message"]))
    return suggestions
