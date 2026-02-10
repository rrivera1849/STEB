import os
from itertools import zip_longest
from typing import Dict, List, Any, Iterator, Optional
from pathlib import Path

def load_onestop_english_corpus(data_dir: str) -> List[Dict[str, Any]]:
    """
    Load OneStopEnglishCorpus dataset from text files. Similar to https://github.com/nishkalavallabhi/OneStopEnglishCorpus/issues/4
    
    Reads text files from Texts-SeparatedByReadingLevel and pairs Advanced /
    Intermediate / Elementary files by base filename. Each input .txt contains
    multiple lines; we align by line index and create **one record per aligned
    line-triplet** (advanced, intermediate, elementary).
    We use nested folder for Intermediate

    This (hopefully) avoids encoding issues in the pre-generated CSV files.
    
    Args:
        data_dir: Path to the dataset directory
        
    Returns:
        List of records with 'text' (ordered list [advanced, intermediate, elementary]) 
        and 'label' ("complexity") fields.
    """
    root = Path(data_dir) / "Texts-SeparatedByReadingLevel"
    adv_dir = root / "Adv-Txt"
    int_dir = root / "Int-Txt" / "Int-Txt"
    ele_dir = root / "Ele-Txt"

    for d in (root, adv_dir, int_dir, ele_dir):
        if not d.exists():
            raise ValueError(f"Directory not found: {d}")

    def base_name(filename: str) -> str:
        # Remove -adv.txt, -int.txt, -ele.txt (case-insensitive) if present.
        lower = filename.lower()
        for suffix in ("-adv.txt", "-int.txt", "-ele.txt"):
            if lower.endswith(suffix):
                return filename[: -len(suffix)]
        return os.path.splitext(filename)[0]

    def index_txt(dir_path: Path) -> Dict[str, Path]:
        return {base_name(p.name): p for p in dir_path.glob("*.txt")}

    def read_lines(path: Path) -> List[str]:
        """
        Read a .txt file as lines.
        Strips leading and trailing whitespace.
        If the file is not valid UTF-8, the entire file is skipped
        and an error message is printed.
        """
        try:
            with path.open("r", encoding="utf-8-sig") as f:
                return [line.strip() for line in f]
        except UnicodeDecodeError as e:
            print(f"Error reading {path}: not valid UTF-8 ({e}). File skipped.")
            return []

    adv_paths = index_txt(adv_dir)
    int_paths = index_txt(int_dir)
    ele_paths = index_txt(ele_dir)

    common_paths = sorted(adv_paths.keys() & int_paths.keys() & ele_paths.keys())

    records: List[Dict[str, Any]] = []
    for base in common_paths:
        adv_lines = read_lines(adv_paths[base])
        int_lines = read_lines(int_paths[base])
        ele_lines = read_lines(ele_paths[base])

        # Align by line index; only keep triples where all 3 lines exist and are non-empty.
        for a, i, e in zip_longest(adv_lines, int_lines, ele_lines, fillvalue=None):
            if a is None or i is None or e is None:
                continue
            if a.strip() == "" or i.strip() == "" or e.strip() == "":
                continue
            records.append({"text": [a, i, e], "label": "complexity"})

    return records

