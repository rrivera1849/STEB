import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import run_new_datasets


def test_parse_models_file_skips_blank_and_comment_lines():
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("# Style-Specific Models\n")
        f.write("rrivera1849/LUAR-MUD\n")
        f.write("\n")
        f.write("lisa_checkpoint\n")
        path = f.name

    try:
        models = run_new_datasets.parse_models_file(path)
    finally:
        os.remove(path)

    assert models == [
        ("LUAR-MUD", "rrivera1849/LUAR-MUD"),
        ("lisa_checkpoint", "lisa_checkpoint"),
    ]


def test_discover_new_datasets_finds_missing_directories():
    with tempfile.TemporaryDirectory() as results_dir:
        os.makedirs(os.path.join(results_dir, "existing_dataset"))

        new_datasets = run_new_datasets.discover_new_datasets(
            ["existing_dataset", "brand_new_dataset"],
            results_dir,
        )

    assert new_datasets == ["brand_new_dataset"]


def test_discover_new_datasets_empty_when_all_present():
    with tempfile.TemporaryDirectory() as results_dir:
        os.makedirs(os.path.join(results_dir, "a"))
        os.makedirs(os.path.join(results_dir, "b"))

        new_datasets = run_new_datasets.discover_new_datasets(["a", "b"], results_dir)

    assert new_datasets == []


def test_partition_models_splits_evenly_by_count():
    models = [(str(i), str(i)) for i in range(5)]

    chunks = run_new_datasets.partition_models(models, num_workers=2)

    assert [len(chunk) for chunk in chunks] == [3, 2]
    assert sum(chunks, []) == models


def test_partition_models_handles_more_workers_than_models():
    models = [("a", "a")]

    chunks = run_new_datasets.partition_models(models, num_workers=3)

    assert chunks == [[("a", "a")], [], []]


def test_build_command_uses_all_subcommand_and_output_folder():
    command = run_new_datasets.build_command(
        "rrivera1849/LUAR-MUD",
        "authbench_attribution_en",
        "/tmp/results",
    )

    assert command == [
        "steb", "all", "rrivera1849/LUAR-MUD",
        "--dataset", "authbench_attribution_en",
        "--output-folder", "/tmp/results",
        "--progress-bar",
    ]


def test_discover_gpus_returns_empty_when_nvidia_smi_missing(monkeypatch):
    def fake_run(*args, **kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert run_new_datasets.discover_gpus() == []


def test_discover_gpus_parses_indices(monkeypatch):
    class FakeResult:
        stdout = "0\n1\n2\n"

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: FakeResult())

    assert run_new_datasets.discover_gpus() == [0, 1, 2]
