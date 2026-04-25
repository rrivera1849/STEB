from steb.loaders.endive import (
    DIALECTS,
    N_PER_DATASET,
    _detect_columns,
    load_endive,
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
# Integration: actually fetch a small EnDive HF dataset
# ---------------------------------------------------------------------------

# multirc has the fewest aligned source-ids (15), so it exercises the
# "min what you can find" path: cap N_PER_DATASET=100, but only 15
# source-ids exist in all 5 dialect splits, so the loader returns 75.

def test_load_endive_returns_one_record_per_source_per_dialect():
    """Each kept source-id should produce exactly one record per dialect."""
    records = load_endive("endive/multirc")

    # All 5 dialects should be represented equally.
    per_dialect = {d: 0 for d in DIALECTS}
    for r in records:
        per_dialect[r["label"]] += 1

    counts = set(per_dialect.values())
    assert len(counts) == 1, f"Per-dialect counts should match, got {per_dialect}"

    n_per_dialect = next(iter(counts))
    assert 0 < n_per_dialect <= N_PER_DATASET
    assert len(records) == n_per_dialect * len(DIALECTS)


def test_load_endive_record_shape():
    """Each record is exactly {text: str, label: str}."""
    records = load_endive("endive/multirc")
    assert records, "loader should return at least one record"

    for r in records:
        assert set(r.keys()) == {"text", "label"}
        assert isinstance(r["text"], str) and r["text"]
        assert r["label"] in DIALECTS


def test_load_endive_caps_at_n_per_dataset():
    """For a larger HF task, loader returns N_PER_DATASET source-ids per dialect (not more)."""
    records = load_endive("endive/sst-2")  # 638 fully-aligned source-ids

    per_dialect = {d: 0 for d in DIALECTS}
    for r in records:
        per_dialect[r["label"]] += 1

    for d, count in per_dialect.items():
        assert count == N_PER_DATASET, f"{d}: expected {N_PER_DATASET}, got {count}"
