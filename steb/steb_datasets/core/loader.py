import os
from typing import Any, Dict, List

# Top-level register categories — excluded from sub-label assignment
_MAIN_LABELS = {"IN", "NA", "HI", "LY", "SP", "IP", "ID", "OP"}

# Mapping from short code (uppercase) to human-readable snake_case label.
# Full register taxonomy: https://link.springer.com/article/10.1007/s10579-022-09624-1/tables/1
_LABEL_MAP: Dict[str, str] = {
    # IN — Informational Description/Explanation
    "CM": "course_materials",
    "DP": "description_of_a_person",
    "DT": "description_of_a_thing",
    "EN": "encyclopedia_article",
    "FI": "faq_about_information",
    "IB": "information_blog",
    "LT": "legal_terms_and_conditions",
    "OI": "other_information",
    "RA": "research_article",
    "TR": "technical_report",
    # NA — Narrative
    "HA": "historical_article",
    "MA": "magazine_article",
    "NE": "news_report_blog",
    "ON": "other_narrative",
    "PB": "personal_blog",
    "SR": "sports_report",
    "SS": "short_story",
    "TB": "travel_blog",
    # HI — How-To/Instructional
    "FH": "faq_about_how_to",
    "HT": "how_to",
    "OH": "other_how_to",
    "RE": "recipe",
    "TS": "technical_support",
    # LY — Lyrical
    "OL": "other_lyrical",
    "PO": "poem",
    "PR": "prayer",
    "SL": "song_lyrics",
    # IP — Informational Persuasion
    "DS": "description_with_intent_to_sell",
    "ED": "editorial",
    "OE": "other_informational_persuasion",
    "PA": "persuasive_article_or_essay",
    # OP — Opinion
    "AD": "advertisement",
    "AV": "advice",
    "LE": "letter_to_editor",
    "OB": "opinion_blog",
    "OO": "other_opinion",
    "RS": "religious_blogs_sermons",
    "RV": "reviews",
    "RR": "reader_viewer_responses",
    # ID — Interactive Discussion
    "DF": "discussion_forum",
    "OF": "other_forum",
    "QA": "question_answer_forum",
    # SP — Spoken
    "FS": "formal_speech",
    "IT": "interview",
    "OS": "other_spoken",
    "TA": "transcript_of_video_audio",
    "TV": "tv_movie_script",
}


def load_core_dataset(data_dir: str) -> List[Dict[str, Any]]:
    """
    Load the CORE corpus from TSV files (train, dev, test).

    TSV format (no header):
        col 0: space-separated register label(s) (uppercase codes)
        col 1: CORE document id
        col 2: text content

    For each document, one record is emitted per sub-label assigned to it
    (e.g. a document labelled "IN CM NA HA" yields two records: one for
    "course_materials" and one for "historical_article"). Top-level category
    codes (IN, NA, HI, LY, SP, IP, ID, OP) and the special value "OTHER"
    are ignored.

    All files (train, dev, test) are merged into a single pool so that the
    task draws samples from the complete corpus.

    Dataset: https://github.com/TurkuNLP/CORE-corpus
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"CORE-corpus directory not found: {data_dir}")

    records: List[Dict[str, Any]] = []
    found_any = False

    for filename in ["train.tsv", "dev.tsv", "test.tsv"]:
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            continue
        found_any = True

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                parts = line.split("\t")
                if len(parts) < 3:
                    continue

                labels_str = parts[0]
                text = parts[2]

                if not text.strip():
                    continue

                for token in labels_str.strip().split():
                    readable = _LABEL_MAP.get(token.upper())
                    if readable is not None:
                        records.append({"text": text, "label": readable})

    if not found_any:
        raise FileNotFoundError(f"No TSV files (train.tsv, dev.tsv, test.tsv) found in: {data_dir}")

    return records
