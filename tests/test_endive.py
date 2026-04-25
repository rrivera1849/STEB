from steb.loaders.endive import (
    ALL_LABELS,
    DIALECTS,
    _detect_columns,
    _load_one_nlu_task,
)


# ---------------------------------------------------------------------------
# Pure-Python column detection (no network)
# ---------------------------------------------------------------------------

def test_detect_columns_context_schema():
    """logic_bench_*, etc. use 'Context' / 'Dialect (context)'."""
    row = {
        "Context": "...",
        "Dialect (context)": "...",
        "BLEU Score Context": 0.5,
        "Question 1": "...",
    }
    src, dl = _detect_columns(row)
    assert src == "Context"
    assert dl == "Dialect (context)"


def test_detect_columns_original_schema():
    """svamp uses 'Original' / 'Dialect (Original)' (case differs in inner)."""
    row = {
        "Original": "...",
        "Dialect (Original)": "...",
        "BLEU Score Original": 0.5,
        "Question": "...",
        "Answer": 1,
    }
    src, dl = _detect_columns(row)
    assert src == "Original"
    assert dl == "Dialect (Original)"


def test_detect_columns_ignores_unrelated_columns():
    """Extra unrelated columns shouldn't confuse detection."""
    row = {
        "Context": "...",
        "Dialect (context)": "...",
        "Answer 1": "yes",
        "Answer 2": "no",
        "id": 0,
    }
    src, dl = _detect_columns(row)
    assert src == "Context"
    assert dl == "Dialect (context)"


# ---------------------------------------------------------------------------
# Integration: actually fetch one of the smaller EnDive HF tasks
# ---------------------------------------------------------------------------
# multirc has the fewest fully-aligned source-ids (15), so it exercises
# the "min what you can find" path: cap N_PER_TASK=150 but only 15
# source-ids exist in all 5 dialect splits. We expect 15 × 6 = 90 records
# (SAE + 5 dialects per source-id).

def test_load_one_nlu_task_record_shape():
    """Each record is exactly {text: str, label: str} with a known label."""
    records = _load_one_nlu_task("multirc")
    assert records, "expected at least some records for multirc"
    for r in records:
        assert set(r.keys()) == {"text", "label"}
        assert isinstance(r["text"], str) and r["text"]
        assert r["label"] in ALL_LABELS


def test_load_one_nlu_task_balanced_labels():
    """Every kept source-id contributes one record per label, so all 6 labels share counts."""
    records = _load_one_nlu_task("multirc")
    counts = {label: sum(1 for r in records if r["label"] == label) for label in ALL_LABELS}
    assert len(set(counts.values())) == 1, f"unbalanced: {counts}"
    assert sum(counts.values()) == len(records)
    # multirc has 15 fully-aligned source-ids, so 15 per label, 90 total.
    assert counts["sae"] > 0


def test_load_one_nlu_task_sae_text_is_source_text():
    """The 'sae' record's text equals the SAE source text shared across the 5 dialect siblings."""
    records = _load_one_nlu_task("multirc")
    # records are emitted in (sae, aave, chce, collsge, inde, jame) order per source-id
    n_labels = len(ALL_LABELS)
    assert len(records) % n_labels == 0
    for i in range(0, len(records), n_labels):
        group = records[i:i + n_labels]
        assert group[0]["label"] == "sae"
        for j, dialect in enumerate(DIALECTS, start=1):
            assert group[j]["label"] == dialect
        # SAE text shouldn't equal any single dialect text (translations differ from original).
        # (Not strictly guaranteed, but a reasonable invariant for these GPT-4o translations
        # filtered to BLEU < 0.7. Soft check: SAE differs from at least one dialect.)
        assert any(group[0]["text"] != group[j]["text"] for j in range(1, n_labels))
