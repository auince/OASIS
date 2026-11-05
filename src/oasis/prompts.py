INSTRUCTION = (
    "Assess the image on exactly one quality dimension. Follow the supplied "
    "definition rather than guessing from the dimension name. Return only the "
    "single learned score token and no explanation."
)


def build_prompt(definition):
    fields = (
        ("name", "Dimension"), ("description", "Definition"),
        ("positive_evidence", "Positive evidence"),
        ("negative_evidence", "Negative evidence"),
        ("score_guidance", "Scoring guidance"),
    )
    if not isinstance(definition, dict):
        raise ValueError("definition must be an object")
    for key, _ in fields:
        if not isinstance(definition.get(key), str) or not definition[key].strip():
            raise ValueError(f"definition.{key} must be a non-empty string")
    return INSTRUCTION + "\n\n" + "\n".join(
        f"{label}: {definition[key]}" for key, label in fields
    )
