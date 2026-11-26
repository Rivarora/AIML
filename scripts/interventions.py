"""Generate conservative, explainable advisory suggestions from validated inputs."""
from __future__ import annotations

from typing import Any


def build_interventions(values: dict[str, Any]) -> list[str]:
    suggestions: list[str] = []

    if "attendance" in values and float(values["attendance"]) < 75:
        suggestions.append("Consider improving attendance where possible.")
    if "previous_scores" in values and float(values["previous_scores"]) < 60:
        suggestions.append("Consider academic support for earlier concepts.")
    if "communication_skills" in values and float(values["communication_skills"]) < 6:
        suggestions.append("Consider communication practice and feedback.")
    if "projects_completed" in values and float(values["projects_completed"]) < 2:
        suggestions.append("Consider completing more practical projects.")
    if "hours_studied" in values and float(values["hours_studied"]) < 10:
        suggestions.append("Consider setting aside regular study time.")

    return suggestions
