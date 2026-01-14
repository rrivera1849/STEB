from typing import Any, Dict, Optional, List


def synthstel_record_handler(example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """generated with cursor gpt-5.2 High

    Custom record handler for SynthSTEL.
    Converts Hugging Face fields into STEB's expected format:
      - text: [most_style, least_style]
      - label: style_feature

    SynthSTEL fields (HF): positive, negative, feature, feature_clean
      - positive: text exhibiting the style feature (=> most_style)
      - negative: text lacking the style feature (=> least_style)
      - feature_clean (fallback: feature): style feature label
    """
    texts: List[str] = []

    most_style = example.get("positive")
    least_style = example.get("negative")
    style_feature = example.get("feature_clean") or example.get("feature")

    if isinstance(most_style, str):
        most_style = most_style.strip()
        if most_style:
            texts.append(most_style)

    if isinstance(least_style, str):
        least_style = least_style.strip()
        if least_style:
            texts.append(least_style)

    if len(texts) != 2:
        return None

    if isinstance(style_feature, str):
        style_feature = style_feature.strip().lower()
    else:
        style_feature = None

    if not style_feature:
        return None

    return {
        "text": texts,
        "label": style_feature,
    }

