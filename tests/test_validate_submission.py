"""Tests for scripts/validate_submission.py."""
from pathlib import Path

from scripts.validate_submission import (
    collect_models_in_leaderboard,
    collect_models_in_models_file,
)


def test_collect_models_in_leaderboard_parses_table_rows(tmp_path: Path) -> None:
    md = tmp_path / "leaderboard.md"
    md.write_text(
        "# Leaderboard\n\n"
        "| Model | a | b |\n"
        "|---|---|---|\n"
        "| LUSR | **1.0** | 2.0 |\n"
        "| star | 3.0 | 4.0 |\n"
    )
    assert collect_models_in_leaderboard(md) == {"LUSR", "star"}


def test_collect_models_in_leaderboard_missing_file(tmp_path: Path) -> None:
    assert collect_models_in_leaderboard(tmp_path / "nope.md") == set()


def test_published_models_are_in_real_leaderboard() -> None:
    names = collect_models_in_leaderboard()
    assert "LUSR" in names
    assert "Model" not in names


def test_models_file_contains_short_and_full_ids() -> None:
    ids = collect_models_in_models_file()
    assert "rrivera1849/LUSR" in ids
    assert "LUSR" in ids
