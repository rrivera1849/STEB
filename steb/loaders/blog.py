import os
from typing import Any, Dict, List

from datasets import concatenate_datasets, load_dataset


def _age_bucket(
    age: int,
) -> str | None:
    """
    Maps a blogger's integer age to one of the three Schler et al. (2006)
    age groups, or returns ``None`` if the age falls outside them.

    Args:
        age: The blogger's self-reported age.

    Returns:
        ``"10s"`` for ages 13-17, ``"20s"`` for 23-27, ``"30s"`` for 33-47,
        or ``None`` if the age falls in one of the gap ranges (18-22, 28-32)
        or outside 13-47.
    """
    if 13 <= age <= 17:
        return "10s"
    if 23 <= age <= 27:
        return "20s"
    if 33 <= age <= 47:
        return "30s"
    return None


def load_blog(
    data_dir: str,
) -> List[Dict[str, Any]]:
    """
    Loads the Blog Authorship Corpus (all splits combined) from Hugging
    Face and labels each post by a single demographic attribute.

    The *data_dir* argument encodes the target attribute as its last path
    component: ``"age"`` produces three classes (``"10s"``, ``"20s"``,
    ``"30s"``) following the Schler et al. (2006) age groupings, while
    ``"gender"`` produces two classes (``"male"``, ``"female"``).
    The corpus has no held-out test split, so we concatenate the
    available train and validation splits to expose the full ~727k posts.

    Args:
        data_dir: Path whose last component is ``"age"`` or ``"gender"``.

    Returns:
        A list of records with ``text`` and ``label`` keys.
    """
    attribute = os.path.basename(data_dir)
    if attribute not in {"age", "gender"}:
        raise ValueError(
            f"Unknown blog attribute '{attribute}'. Expected 'age' or 'gender'."
        )

    splits = load_dataset(
        "barilan/blog_authorship_corpus",
        trust_remote_code=True,
    )
    ds = concatenate_datasets([splits[name] for name in splits])

    records: List[Dict[str, Any]] = []
    for row in ds:
        text = row.get("text")
        if not text or not isinstance(text, str):
            continue

        if attribute == "age":
            label = _age_bucket(row.get("age"))
        else:
            gender = row.get("gender")
            label = gender if gender in {"male", "female"} else None

        if label is None:
            continue

        records.append({"text": text, "label": label})

    unique_labels = set(r["label"] for r in records)
    print(f"Unique labels: {unique_labels}")
    return records
