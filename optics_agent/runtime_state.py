"""Factual draw feedback, independent of model decisions and optical grading."""
from __future__ import annotations


def draw_runtime_state(*, used: int, limit: int, same_prompt: bool, same_image: bool) -> dict:
    """Describe the enforced budget and exact consecutive duplicates to the actor.

    Duplicate flags compare the caption text and the returned file hash. They do
    not assess optical correctness, and never replace the actual image review.
    """
    remaining = limit - used
    instruction = "Review the returned pixels against the original request. "
    if remaining:
        instruction += (
            "If a material defect remains, change the relevant caption constraints "
            "before another draw; otherwise submit."
        )
    else:
        instruction += (
            "No draw calls remain. Give a final response stating any unresolved "
            "material defects; do not request another draw."
        )
    if same_prompt and same_image:
        instruction += (
            " The caption and image are unchanged from the previous draw. "
            "Repeating this action with the same seed will not make progress."
        )
    return {
        "schema_version": 1,
        "image_calls_used": used,
        "max_image_calls": limit,
        "remaining_image_calls": remaining,
        "draw_available": remaining > 0,
        "identical_prompt_to_previous": same_prompt,
        "identical_image_to_previous": same_image,
        "instruction": instruction,
    }
