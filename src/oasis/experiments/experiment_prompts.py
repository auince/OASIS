from __future__ import annotations

import re
from collections.abc import Mapping

from ..prompts import INSTRUCTION, build_prompt

SCORE_INSTRUCTION = (
    "Assess the image on exactly one quality dimension. Return only the single "
    "learned score token and no explanation."
)
OFFICIAL_MOS_DEFINITION = {
    "name": "overall_quality",
    "description": (
        "Evaluate the image's overall perceived quality as a holistic human mean opinion score. "
        "Consider how the full image works as a viewing experience, including the combined effect "
        "of exposure, color rendering, tonal separation, visible noise or artifacts, detail clarity, "
        "and any other clearly visible strengths or defects. A higher score means stronger overall "
        "perceived quality. Do not compute a mechanical average of the five named dimensions: "
        "weigh evidence by severity, spatial extent, scene relevance, and impact on the whole image. "
        "Broad or content-obscuring problems should matter more than minor local imperfections."
    ),
    "positive_evidence": (
        "an image with high overall perceived quality, coherent global presentation, clearly "
        "readable content, and visible strengths that outweigh its defects"
    ),
    "negative_evidence": (
        "an image with low overall perceived quality because broad, severe, or content-obscuring "
        "weaknesses substantially harm the viewing experience"
    ),
    "score_guidance": (
        "Dimension-specific score meaning:\n"
        "For SPAQ MOS, the 0-100 score represents the image's overall perceived quality as a holistic human opinion score.\n"
        "A higher score means the image is more visually successful and satisfying overall after considering the full image and the combined impact of visible strengths and defects.\n"
        "A lower score means visible weaknesses substantially reduce the image's overall viewing quality.\n"
        "Judge the final holistic impression rather than mechanically averaging individual attributes, and let broad or severe issues matter more than small isolated imperfections."
    ),
}
OFFICIAL_MOS_PROMPT = build_prompt(OFFICIAL_MOS_DEFINITION)
GENERIC_QUALITY_PROMPT = (
    INSTRUCTION + "\n\nDimension: overall_quality\nDefinition: 分析该图片质量"
)
VARIANTS = (
    "full_prompt", "definition_only", "name_only", "shared_mos", "image_only", "generic_quality"
)
_FIELDS = (
    ("Dimension", "name"), ("Definition", "description"),
    ("Positive evidence", "positive_evidence"),
    ("Negative evidence", "negative_evidence"), ("Scoring guidance", "score_guidance"),
)
_MARKER = re.compile(r"^(Dimension|Definition|Positive evidence|Negative evidence|Scoring guidance):[ \t]*", re.MULTILINE)


def parse_full_prompt(prompt: str) -> dict[str, str]:

    if not isinstance(prompt, str) or not prompt.startswith(INSTRUCTION + "\n\n"):
        raise ValueError("Full prompt must use the exact paper instruction and five named fields")
    body = prompt[len(INSTRUCTION) + 2:]
    matches = list(_MARKER.finditer(body))
    if [match.group(1) for match in matches] != [label for label, _ in _FIELDS] or matches[0].start() != 0:
        raise ValueError("Full prompt must contain each of Dimension, Definition, Positive evidence, "
                         "Negative evidence and Scoring guidance exactly once, in order")
    result = {}
    for index, (match, (_, key)) in enumerate(zip(matches, _FIELDS)):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        value = body[match.end():end].strip()
        if not value:
            raise ValueError(f"Full prompt field {match.group(1)} must not be empty")
        result[key] = value
    return result


def extract_definition(row: Mapping) -> dict[str, str]:

    value = row.get("definition")
    if isinstance(value, Mapping):
        result = dict(value)
        for target, legacy in (("name", "dimension"), ("positive_evidence", "positive_prompt"),
                               ("negative_evidence", "negative_prompt")):
            if target not in result and legacy in result:
                result[target] = result[legacy]
        if "name" not in result:
            result["name"] = row.get("dimension", row.get("task"))
        build_prompt(result)
        if "prompt" in row and parse_full_prompt(row["prompt"]) != {key: result[key].strip() for _, key in _FIELDS}:
            raise ValueError("Structured definition and supplied full prompt disagree")
        return {key: result[key] for _, key in _FIELDS}
    if "definition" in row:
        raise ValueError("definition must contain the five structured criterion fields")
    return parse_full_prompt(row.get("prompt"))


def build_variant_prompt(row: Mapping, variant: str) -> str:

    if variant not in VARIANTS:
        raise ValueError(f"Unknown input variant: {variant}")
    if variant == "image_only":
        return ""
    if variant == "shared_mos":
        return OFFICIAL_MOS_PROMPT
    if variant == "generic_quality":
        return GENERIC_QUALITY_PROMPT
    definition = extract_definition(row)
    if variant == "full_prompt":
        return row["prompt"] if "prompt" in row else build_prompt(definition)
    if variant == "definition_only":
        return SCORE_INSTRUCTION + "\n\nDefinition: " + definition["description"]
    return SCORE_INSTRUCTION + "\n\nDimension: " + definition["name"]
