import os
import sys
import tempfile

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import export_leaderboard_json


def test_build_model_url_map_links_only_hub_repo_ids():
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("rrivera1849/LUAR-MUD\n")
        f.write("lisa_checkpoint\n")
        f.write("surface_pos.yaml\n")
        path = f.name

    try:
        url_map = export_leaderboard_json.build_model_url_map(path)
    finally:
        os.remove(path)

    assert url_map == {"LUAR-MUD": "https://huggingface.co/rrivera1849/LUAR-MUD"}


def test_sheet_to_table_includes_model_url_and_handles_nan():
    df = pd.DataFrame(
        {"score_a": [10.0, float("nan")], "score_b": [20.0, 30.0]},
        index=["LUAR-MUD", "lisa_checkpoint"],
    )

    table = export_leaderboard_json._sheet_to_table(
        df,
        sort_column="score_b",
        model_url_by_name={"LUAR-MUD": "https://huggingface.co/rrivera1849/LUAR-MUD"},
    )

    assert table["sort_column"] == "score_b"
    assert table["columns"] == ["score_a", "score_b"]

    rows_by_model = {row["model"]: row for row in table["rows"]}
    assert rows_by_model["LUAR-MUD"]["model_url"] == "https://huggingface.co/rrivera1849/LUAR-MUD"
    assert rows_by_model["LUAR-MUD"]["score_a"] == 10.0
    assert rows_by_model["lisa_checkpoint"]["model_url"] is None
    assert rows_by_model["lisa_checkpoint"]["score_a"] is None

    # sorted by score_b descending
    assert [row["model"] for row in table["rows"]] == ["lisa_checkpoint", "LUAR-MUD"]
