# Frozen historical caption contract. Used only to read/project existing v2 records.
"""Generator-facing caption format shared by image annotation and Agent tools.

Sections describe the final pixels, not an optical reasoning trace. Observed
mode describes an existing training image; planned mode follows Agent reasoning.
"""
from __future__ import annotations

from collections.abc import Mapping


CAPTION_CONTRACT_VERSION = "optics_visible_results_v2"
GENERATOR_CAPTION_FIELDS = (
    ("style", "Style"),
    ("composition", "Composition"),
    ("scene", "Scene"),
    ("materials_and_lighting", "Materials and lighting"),
    ("visible_optical_results", "Visible optical results"),
    ("detail_and_focus", "Detail and focus"),
)


def generator_caption_instructions(mode: str = "planned") -> str:
    """Return the same output format with task-appropriate grounding rules."""
    if mode not in {"observed", "planned"}:
        raise ValueError("caption mode must be observed or planned")
    common = """Use the optics_visible_results_v2 generator-caption contract. Write a single English
dense caption in natural descriptive language, with these six labels in this exact order:
Style: Describe the visual style and visible photographic treatment, such as a natural outdoor
photograph, tabletop close-up or laboratory photograph. Photorealism describes appearance;
do not claim a known camera origin or claim a renderer.
Composition: State framing, viewpoint, placement and relative scale of key objects,
foreground/background relations, occlusions, and which side of a surface is being viewed.
Scene: Describe concrete objects, relevant counts, surroundings, colors and spatial relations.
Materials and lighting: Describe visible surface texture, transparency, reflectivity, lighting
appearance and contrast; name a light source or direction only when justified.
Visible optical results: Explicitly describe the final observable outcome in image space.
Identify the affected object or region and its relation to the original object or surrounding
scene: displacement, orientation, apparent size, repeated images, deformation, shadow position,
edge softness, color order, bright or dark bands, visible beam extent or blur, as applicable.
A phenomenon name alone (such as refraction, interference, or total internal reflection) is
never an adequate result. Do not use optical laws as a substitute for visible results.
Detail and focus: State sharpness, focus/depth relationships, relevant small detail,
texture and background treatment without generic quality slogans or invented camera settings.
Keep the six sections mutually consistent and describe one coherent image. Use concise complete
sentences, usually about 120-220 English words when the scene supports that much detail; do not
invent information to meet a word count. Return the caption itself, without reasoning steps,
equations, chain of thought, analysis, evidence tags, confidence scores or a category checklist.
Avoid generic claims such as 'physically accurate optics' and phrases such as 'the image shows'.
Do not add arrows, optical-path diagrams, explanatory labels, split panels or a collage unless
explicitly requested by the user. Preserve ordinary relevant text or markings in the scene.
Example of a visible result, not a universal rule: 'At the waterline, the submerged straw
segment appears offset to the right of the segment above it.' The direction must come from
this image or established geometry; never copy the example automatically.
"""
    if mode == "observed":
        return common + """\nOBSERVED-IMAGE MODE: This dataset requires realistic photographic appearance. Ground every
section in the supplied image. Search labels and category hints are unverified. Describe what
is actually visible, including real focus limitations; do not improve, beautify, repair,
relight or invent a scene. Distinguish appearances from causes: 'a curved colored band' can
be visible without proving a refractive index, wavelength, hidden light location, exact lens
model, focal length, aperture, material composition or mechanism. Keep causal explanations
and uncertain metadata outside the generator caption. Omit details that cannot be established
and record uncertainty in review evidence. Reject unsuitable images or unclear target results
in content review; do not hallucinate a result to fill a section. For a rejected image only,
a section may briefly state that the corresponding detail is not reliably visible.
"""
    return common + """\nPLANNED-SCENE MODE: Default to realistic photographic appearance for optical image tasks;
an explicit user request for a different style takes precedence and must remain explicit.
Use user constraints and the Agent's established scene/optical reasoning to infer the visible
result to render. Preserve those constraints and resolve layout/viewpoint consistently.
Translate reasoning into concrete visible relationships; do not forward the reasoning trace
or merely request a named phenomenon. Do not assert unseen numerical optical parameters or
irrelevant camera settings. If a parameter is explicitly specified, its visible consequences
must agree with it; do not invent additional measurements. Describe the intended image rather
than claiming it was observed in a reference photo. Include only effects justified by the scene.
"""


def validate_generator_caption(fields: Mapping[str, str]) -> dict[str, str]:
    """Validate structure only; image evidence must be checked by the verifier."""
    if not isinstance(fields, Mapping):
        raise ValueError("caption.structure must be an object")
    if set(fields) != {key for key, _ in GENERATOR_CAPTION_FIELDS}:
        raise ValueError("caption.structure must contain exactly the six generator-caption fields")
    cleaned = {}
    for key, _ in GENERATOR_CAPTION_FIELDS:
        value = fields[key]
        if not isinstance(value, str) or not value.strip():
            raise ValueError("caption.structure." + key + " must be a nonempty string")
        cleaned[key] = " ".join(value.split())
    return cleaned


def render_generator_caption(fields: Mapping[str, str]) -> str:
    """Deterministically render annotation fields as the Agent's caption text."""
    cleaned = validate_generator_caption(fields)
    return "\n".join(f"{label}: {cleaned[key]}" for key, label in GENERATOR_CAPTION_FIELDS)
