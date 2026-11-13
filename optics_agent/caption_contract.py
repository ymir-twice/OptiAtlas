"""Generator-facing caption format shared by image annotation and Agent tools.

Sections describe the final pixels, not an optical reasoning trace. Observed
mode describes an existing training image; planned mode follows Agent reasoning.
"""
from __future__ import annotations

from collections.abc import Mapping


# Storage/cache identity only. Never insert this identifier into model prompts.
CAPTION_CONTRACT_VERSION = "optics_visible_results_v3"
SUPPORTED_CAPTION_CONTRACT_VERSIONS = ("optics_visible_results_v2", CAPTION_CONTRACT_VERSION)
GENERATOR_CAPTION_FIELDS = (
    ("style", "Style"),
    ("composition", "Composition"),
    ("materials_and_lighting", "Materials and lighting"),
    ("visible_optical_results", "Visible optical results"),
    ("detail_and_focus", "Detail and focus"),
)


def generator_caption_instructions(mode: str = "planned") -> str:
    """Return the same output format with task-appropriate grounding rules."""
    if mode not in {"observed", "planned"}:
        raise ValueError("caption mode must be observed or planned")
    lighting_basis = ("user constraints or justified properties of the intended scene"
                      if mode == "planned" else "visible evidence in the supplied image")
    common = f"""The generator caption is an action interface describing one complete image.
Write natural English using these five labels in this exact order:
Style: State the visual style and photographic treatment. Photorealism describes appearance,
not proof of camera origin or renderer provenance.
Composition: Describe the key two-dimensional layout needed to specify this image:
relevant object placements, relative sizes, spacing, overlaps, cropping and relationships.
Include identities, counts and colors where relevant. Use regional descriptions only when
helpful, not an exhaustive inventory of picture regions. Keep this section on the picture
plane; do not insert a three-dimensional reconstruction or duplicate scene section.
Materials and lighting: Describe surface texture, transparency, reflectivity, light sources
and lighting appearance or direction when supported by {lighting_basis}.
Visible optical results: Describe concrete observable outcomes and their reference objects
or regions: displacement, orientation, apparent size, repeated images, deformation, shadow
position and edge softness, color order, bright/dark bands, beam extent or blur as relevant.
A phenomenon name or optical law is not a substitute for the visible result.
Detail and focus: State which regions or objects are sharp or blurred and describe useful
small details and texture, without invented camera settings or generic quality slogans.
Keep all five sections mutually consistent. Usually 120-220 English words suffice; never
invent details to meet a word count. Each caption must stand alone, including after a
correction; do not refer to a previous image or prompt. Internal qualitative three-dimensional
scene and optical reasoning may inform the caption, but output only the resulting image
description, without reasoning steps, chain of thought, equations or analysis. Do not assert
exact hidden geometry, optical parameters or camera settings solely from two-dimensional
projections. Keep evidence tags, confidence scores and category checklists outside the caption.
Avoid generic claims such as 'physically accurate optics'. Do not add arrows, optical-path
diagrams, explanatory labels, split panels or collages unless requested; preserve ordinary
relevant scene text. Describe result directions only when supported, never from a universal
shortcut such as 'a submerged straw always shifts right'.
"""
    if mode == "observed":
        return common + """\nDescribe the supplied image using only what is visibly supported in it.
The actual pixels are authoritative; search labels, categories and a physically expected
effect are not evidence that it appears. This dataset requires realistic photographic
appearance. Internal qualitative reasoning may guide inspection but cannot establish hidden
geometry, causes, materials or measurements from appearance alone. Describe actual results
and focus limitations; do not improve, beautify, repair, relight or invent the image. A colored
band does not establish a wavelength or mechanism. Recheck every section against the image.
Remove unsupported claims when authoring a caption; flag them when independently verifying
one, and never mentally repair the image. Keep uncertain causes and metadata in separate
review evidence. Reject unsuitable
images or unclear target results; do not hallucinate missing effects. Only for rejected images,
a section may say the corresponding detail is not reliably visible.
"""
    return common + """\nDescribe the intended image from the original user request and justified optical reasoning.
The original request remains authoritative after every correction; a previous draw caption
never replaces it. Priority: explicit user constraints > derived visible optical requirements
> optional aesthetics. Before generation, reason internally about qualitative three-dimensional
scene geometry and optics to derive the visible results. Preserve explicit objects, materials,
lights, viewpoint, geometry and numerical parameters in the appropriate section; do not drop
them merely because they are not directly visible. Keep Composition on the picture plane and
express derived optical consequences as visible relationships, without the reasoning trace or
invented measurements. Default to realistic photography unless the user requests another style.
Do not claim the intended image has already been observed; include only justified effects.
"""


def caption_fields(format_version: str = CAPTION_CONTRACT_VERSION):
    """Read historical structure without relabeling it as the current format."""
    if format_version == CAPTION_CONTRACT_VERSION:
        return GENERATOR_CAPTION_FIELDS
    if format_version == "optics_visible_results_v2":
        from .caption_contract_v2 import GENERATOR_CAPTION_FIELDS as old_fields
        return old_fields
    raise ValueError("unsupported caption.format_version")


def validate_generator_caption(fields: Mapping[str, str], *,
                               format_version: str = CAPTION_CONTRACT_VERSION) -> dict[str, str]:
    """Validate structure only; image evidence must be checked by the verifier."""
    if not isinstance(fields, Mapping):
        raise ValueError("caption.structure must be an object")
    expected = caption_fields(format_version)
    if set(fields) != {key for key, _ in expected}:
        count = "five" if len(expected) == 5 else "six"
        raise ValueError("caption.structure must contain exactly the " + count + " generator-caption fields")
    cleaned = {}
    for key, _ in expected:
        value = fields[key]
        if not isinstance(value, str) or not value.strip():
            raise ValueError("caption.structure." + key + " must be a nonempty string")
        cleaned[key] = " ".join(value.split())
    return cleaned


def render_generator_caption(fields: Mapping[str, str], *,
                             format_version: str = CAPTION_CONTRACT_VERSION) -> str:
    """Deterministically render annotation fields as the Agent's caption text."""
    cleaned = validate_generator_caption(fields, format_version=format_version)
    return "\n".join(f"{label}: {cleaned[key]}" for key, label in caption_fields(format_version))
