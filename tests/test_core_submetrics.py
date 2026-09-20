import json
import os
import tempfile

from steb import core


def test_resolve_submetrics_config_passthrough_for_dict():
    inline = {"tweets_vs_switchboard": ["tweets", "switchboard"]}
    resolved = core._resolve_submetrics_config(inline, data_dir="some_dataset")
    assert resolved is inline


def test_resolve_submetrics_config_loads_sibling_file(monkeypatch):
    with tempfile.TemporaryDirectory() as raw_datasets_dir:
        dataset_dir = os.path.join(raw_datasets_dir, "my_dataset")
        os.makedirs(dataset_dir)
        submetrics_on_disk = {"short": ["a_query", "a_target", "b_target"]}
        with open(os.path.join(dataset_dir, "submetrics.json"), "w") as f:
            json.dump(submetrics_on_disk, f)

        monkeypatch.setattr(core, "RAW_DATASETS_DIR", raw_datasets_dir)

        resolved = core._resolve_submetrics_config("submetrics.json", data_dir="my_dataset")

        assert resolved == submetrics_on_disk


def test_submetrics_need_metadata_false_for_literal_lists():
    literal = {"tweets_vs_switchboard": ["tweets", "switchboard"]}
    assert core._submetrics_need_metadata(literal, data_dir="irrelevant") is False


def test_submetrics_need_metadata_true_for_predicate_spec():
    predicate = {"length_short": {"field": "length_bucket", "value": "short"}}
    assert core._submetrics_need_metadata(predicate, data_dir="irrelevant") is True


def test_submetrics_need_metadata_empty_config():
    assert core._submetrics_need_metadata({}, data_dir="irrelevant") is False


def test_matches_submetric_predicate_both_sides():
    spec = {"field": "genre", "value": "news", "sides": "both"}
    assert core._matches_submetric_predicate({"genre": "news", "is_query": True}, spec) is True
    assert core._matches_submetric_predicate({"genre": "blog", "is_query": True}, spec) is False
    assert core._matches_submetric_predicate({"genre": "blog", "is_query": False}, spec) is False


def test_matches_submetric_predicate_query_side_only_keeps_all_targets():
    spec = {"field": "length_bucket", "value": "short", "sides": "query"}
    # Query must match the predicate...
    assert core._matches_submetric_predicate({"length_bucket": "short", "is_query": True}, spec) is True
    assert core._matches_submetric_predicate({"length_bucket": "long", "is_query": True}, spec) is False
    # ...but every target is kept regardless, so the candidate pool stays full.
    assert core._matches_submetric_predicate({"length_bucket": "long", "is_query": False}, spec) is True


def test_matches_submetric_predicate_none_metadata_never_matches():
    spec = {"field": "genre", "value": "news"}
    assert core._matches_submetric_predicate(None, spec) is False


class _FakeTask:
    """Records what it was called with; returns a canned metric dict."""

    def __init__(self):
        self.calls = []

    def evaluate(self, X, y):
        self.calls.append((list(X), list(y)))
        return {"n": len(X)}


def test_evaluate_submetrics_predicate_filters_using_metadata():
    processed_data = (["e1", "e2", "e3"], ["a_query", "b_target", "c_target"])
    metadata = [
        {"length_bucket": "short", "is_query": True},
        {"length_bucket": "long", "is_query": False},
        {"length_bucket": "long", "is_query": False},
    ]
    submetrics_config = {
        "length_short": {"field": "length_bucket", "value": "short", "sides": "query"},
    }
    task = _FakeTask()

    result = core._evaluate_submetrics(submetrics_config, processed_data, task, metadata=metadata)

    # a_query (short) plus both targets (kept regardless of bucket) = 3 kept.
    assert result["length_short"] == {"n": 3}
    assert task.calls == [(["e1", "e2", "e3"], ["a_query", "b_target", "c_target"])]


def test_evaluate_submetrics_predicate_without_metadata_reports_error():
    processed_data = (["e1"], ["a_query"])
    submetrics_config = {"length_short": {"field": "length_bucket", "value": "short"}}
    task = _FakeTask()

    result = core._evaluate_submetrics(submetrics_config, processed_data, task, metadata=None)

    assert "error" in result["length_short"]
    assert task.calls == []


def test_evaluate_submetrics_literal_list_still_works_without_metadata():
    """Existing inline-dict (label-list) submetrics are unaffected."""
    processed_data = (["e1", "e2"], ["tweets", "switchboard"])
    submetrics_config = {"tweets_only": ["tweets"]}
    task = _FakeTask()

    result = core._evaluate_submetrics(submetrics_config, processed_data, task)

    assert result["tweets_only"] == {"n": 1}
    assert task.calls == [(["e1"], ["tweets"])]
