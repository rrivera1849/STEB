import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from prepare_authbench_en import (
    length_bucket,
    primary_genre,
    build_query_author_map,
    build_retrieval_records,
    build_submetrics_config,
    build_config,
)


def test_length_bucket_thresholds():
    assert length_bucket(1) == "short"
    assert length_bucket(10) == "short"
    assert length_bucket(11) == "medium"
    assert length_bucket(100) == "medium"
    assert length_bucket(101) == "long"
    assert length_bucket(500) == "long"
    assert length_bucket(501) == "extra_long"
    assert length_bucket(5000) == "extra_long"


def test_primary_genre_splits_on_slash():
    assert primary_genre("blog/student") == "blog"
    assert primary_genre("news") == "news"
    assert primary_genre("social_media/youtube_comment") == "social_media"


def test_build_query_author_map():
    ground_truth_rows = [
        {"query_id": "q1", "author_id": "a1", "positive_ids": ["c1"]},
        {"query_id": "q2", "author_id": "a2", "positive_ids": ["c2"]},
    ]
    result = build_query_author_map(ground_truth_rows)
    assert result == {"q1": "a1", "q2": "a2"}


def _sample_query_rows():
    return [
        {"query_id": "q1", "content": "short text", "genre": "news", "token_length": 5},
        {"query_id": "q2", "content": "medium text", "genre": "blog/student", "token_length": 50},
        {"query_id": "q_orphan", "content": "no author", "genre": "news", "token_length": 5},
    ]


def _sample_candidate_rows():
    return [
        {"candidate_id": "c1", "content": "cand 1", "author_id": "a1", "genre": "news", "token_length": 200},
        {"candidate_id": "c2", "content": "cand 2", "author_id": "a2", "genre": "blog/student", "token_length": 600},
    ]


def test_build_retrieval_records_drops_queries_without_author():
    query_rows = _sample_query_rows()
    candidate_rows = _sample_candidate_rows()
    query_author = {"q1": "a1", "q2": "a2"}  # q_orphan intentionally missing

    records = build_retrieval_records(query_rows, candidate_rows, query_author)

    assert len(records) == 4  # 2 queries kept + 2 candidates, orphan dropped
    labels = {(r["label"], r["is_query"]) for r in records}
    assert ("a1", True) in labels
    assert ("a2", True) in labels
    assert ("a1", False) in labels
    assert ("a2", False) in labels


def test_build_retrieval_records_fields():
    query_rows = _sample_query_rows()[:1]
    candidate_rows = _sample_candidate_rows()[:1]
    query_author = {"q1": "a1"}

    records = build_retrieval_records(query_rows, candidate_rows, query_author)

    query_record = next(r for r in records if r["is_query"])
    assert query_record["text"] == "short text"
    assert query_record["label"] == "a1"
    assert query_record["genre"] == "news"
    assert query_record["primary_genre"] == "news"
    assert query_record["length_bucket"] == "short"

    candidate_record = next(r for r in records if not r["is_query"])
    assert candidate_record["length_bucket"] == "long"


def test_build_submetrics_config_length_bucket_restricts_query_side_only():
    query_rows = _sample_query_rows()[:2]
    candidate_rows = _sample_candidate_rows()
    query_author = {"q1": "a1", "q2": "a2"}
    records = build_retrieval_records(query_rows, candidate_rows, query_author)

    submetrics = build_submetrics_config(records)

    assert submetrics["length_short"] == {
        "field": "length_bucket", "value": "short", "sides": "query",
    }


def test_build_submetrics_config_topic_pool_restricts_both_sides():
    query_rows = _sample_query_rows()[:2]
    candidate_rows = _sample_candidate_rows()
    query_author = {"q1": "a1", "q2": "a2"}
    records = build_retrieval_records(query_rows, candidate_rows, query_author)

    submetrics = build_submetrics_config(records)

    assert submetrics["topic_pool_news"] == {
        "field": "primary_genre", "value": "news", "sides": "both",
    }
    assert submetrics["topic_pool_blog"] == {
        "field": "primary_genre", "value": "blog", "sides": "both",
    }


def test_build_submetrics_config_reads_genres_off_records():
    """Genre submetrics come from the actual records, not a hardcoded list."""
    query_rows = _sample_query_rows()[:1]  # only "news" genre present
    candidate_rows = []
    query_author = {"q1": "a1"}
    records = build_retrieval_records(query_rows, candidate_rows, query_author)

    submetrics = build_submetrics_config(records)

    topic_pool_keys = [k for k in submetrics if k.startswith("topic_pool_")]
    assert topic_pool_keys == ["topic_pool_news"]


def test_build_config_inlines_submetrics_as_predicates():
    """
    Predicate specs are small enough to inline directly in config.json --
    no generated sidecar file needed (unlike the earlier label-list
    implementation).
    """
    records = build_retrieval_records(
        _sample_query_rows()[:1], _sample_candidate_rows()[:1], {"q1": "a1"},
    )
    config = build_config(records)
    submetrics = config["tasks"]["retrieval"]["submetrics"]
    assert isinstance(submetrics, dict)
    assert all(isinstance(spec, dict) for spec in submetrics.values())
