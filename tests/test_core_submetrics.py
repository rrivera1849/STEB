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
