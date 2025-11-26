import os
import csv
import re
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


def parse_id_component(component: str, style_type: str) -> Optional[Tuple[str, int]]:
    """
    Parse an ID component and return (base_id, position).
    Position 0 = most style, Position 1 = least style.
    Returns None if the component doesn't match expected patterns.
    """
    component = component.lower().strip()

    if style_type == "formality":
        # f-XXX (more formal) = position 0, i-XXX (less formal) = position 1
        if component.startswith("f-"):
            return (component[2:], 0)
        elif component.startswith("i-"):
            return (component[2:], 1)

    elif style_type == "simplicity":
        # s-... (more simple) = position 0, x-... or c-... (more complex) = position 1
        if component.startswith("s-"):
            return (component[2:], 0)
        elif component.startswith("x-") or component.startswith("c-"):
            return (component[2:], 1)

    elif style_type == "contraction":
        # ction-X (more contraction) = position 0, wiki-X (less contraction) = position 1
        if "ction-" in component:
            match = re.search(r'ction-(.+)', component)
            if match:
                return (match.group(1), 0)
        elif component.startswith("wiki-"):
            return (component[5:], 1)

    elif style_type == "emotives":
        # emoteXX (more emotive) = position 0, emojiXX (less emotive) = position 1
        if component.startswith("emote"):
            return (component[5:], 0)
        elif component.startswith("emoji"):
            return (component[5:], 1)

    elif "substitution" in style_type:  # nbr_substitution or similar
        # leet-XX (more substitution) = position 0, norm-XX (normal) = position 1
        if component.startswith("leet-"):
            return (component[5:], 0)
        elif component.startswith("norm-"):
            return (component[5:], 1)

    return None


def split_id_components(id_part: str, style_type: str) -> List[str]:
    """
    Split an ID part into its two components based on known prefixes.
    Uses '-' or '_' as delimiters to find component boundaries.
    """
    id_part = id_part.lower().strip()

    # Define prefixes for each style type
    if style_type == "formality":
        prefixes = ['i', 'f']
        delimiter = '-'
    elif style_type == "simplicity":
        prefixes = ['s', 'c', 'x']
        delimiter = '-'
    elif style_type == "contraction":
        # Special case: ction can have a prefix
        if 'ction-' in id_part and 'wiki-' in id_part:
            # Find the two components
            ction_idx = id_part.find('ction-')
            wiki_idx = id_part.find('wiki-')
            if ction_idx < wiki_idx:
                return [id_part[ction_idx:wiki_idx].rstrip('-'), id_part[wiki_idx:]]
            else:
                return [id_part[wiki_idx:ction_idx].rstrip('-'), id_part[ction_idx:]]
        return []
    elif style_type == "emotives":
        prefixes = ['emote', 'emoji']
        delimiter = None  # No delimiter for emotives
    elif "substitution" in style_type:
        prefixes = ['leet', 'norm']
        delimiter = '-'
    else:
        return []

    # For emotives, split differently
    if style_type == "emotives":
        # Find positions of each prefix
        positions = []
        for prefix in prefixes:
            idx = id_part.find(prefix)
            if idx >= 0:
                positions.append((idx, prefix))

        # Sort by position to preserve order in ID
        positions.sort()

        # Extract components in order
        components = []
        for i, (pos, prefix) in enumerate(positions):
            if i < len(positions) - 1:
                next_pos = positions[i + 1][0]
                components.append(id_part[pos:next_pos].strip('-').strip('_'))
            else:
                components.append(id_part[pos:].strip('-').strip('_'))

        return components

    # For other types, split by finding prefix positions
    # Split the id_part by delimiter and reconstruct components
    parts = id_part.split(delimiter)

    components = []
    current_component = []

    for part in parts:
        part = part.strip()
        if not part:
            continue

        # Check if this part starts a new component (matches a prefix)
        starts_new = False
        for prefix in prefixes:
            if part.startswith(prefix):
                starts_new = True
                break

        if starts_new and current_component:
            # Save the previous component
            components.append(delimiter.join(current_component))
            current_component = [part]
        elif starts_new:
            current_component = [part]
        else:
            current_component.append(part)

    # Don't forget the last component
    if current_component:
        components.append(delimiter.join(current_component))

    return components


def extract_pairs_from_row(row: Dict[str, str], style_type: str) -> List[Tuple[str, str]]:
    """
    Extract ordered text pairs from a row in the TSV file.
    Returns list of (most_style_text, least_style_text) tuples.
    """
    pairs = []

    anchor1 = row.get('Anchor 1', '').strip()
    anchor2 = row.get('Anchor 2', '').strip()
    alt11 = row.get('Alternative 1.1', '').strip()
    alt12 = row.get('Alternative 1.2', '').strip()
    id_str = row.get('ID', '').strip()

    if not id_str or not all([anchor1, anchor2, alt11, alt12]):
        return pairs

    # Parse ID: format is typically "QQ_<anchor_ids>_<alt_ids>--<number>"
    # e.g., "QQ_i-454-f-454_f-2482-i-2482--1"
    match = re.match(r'QQ_([^_]+)_([^_]+)', id_str)
    if not match:
        return pairs

    anchor_ids = match.group(1)
    alt_ids = match.group(2)

    # Split anchor IDs (e.g., "i-454-f-454" or "s-t1-101-c-101")
    anchor_components = split_id_components(anchor_ids, style_type)
    if len(anchor_components) == 2:
        comp1_result = parse_id_component(anchor_components[0], style_type)
        comp2_result = parse_id_component(anchor_components[1], style_type)

        if comp1_result and comp2_result:
            _, order1 = comp1_result
            _, order2 = comp2_result

            # Create ordered pair [most_style, least_style]
            if order1 == 0:  # comp1 is most style (anchor1)
                pairs.append((anchor1, anchor2))
            else:  # comp2 is most style (anchor2)
                pairs.append((anchor2, anchor1))

    # Split alternative IDs similarly
    alt_components = split_id_components(alt_ids, style_type)
    if len(alt_components) == 2:
        comp1_result = parse_id_component(alt_components[0], style_type)
        comp2_result = parse_id_component(alt_components[1], style_type)

        if comp1_result and comp2_result:
            _, order1 = comp1_result
            _, order2 = comp2_result

            # Create ordered pair [most_style, least_style]
            if order1 == 0:  # comp1 is most style (alt11)
                pairs.append((alt11, alt12))
            else:  # comp2 is most style (alt12)
                pairs.append((alt12, alt11))

    return pairs


def load_stel(data_dir: str) -> List[Dict[str, any]]:
    """
    Load STEL dataset from TSV files in characteristics and dimensions directories.

    Returns:
        List of records with 'text' (ordered pair) and 'label' (style type) fields.
    """
    # Handle nested STEL directory structure
    stel_path = os.path.join(data_dir, "Data", "STEL")
    if not os.path.exists(stel_path):
        stel_path = data_dir

    # Collect all TSV files from characteristics and dimensions
    tsv_files = []
    for subdir in ["characteristics", "dimensions"]:
        subdir_path = os.path.join(stel_path, subdir)
        if os.path.exists(subdir_path):
            for filename in os.listdir(subdir_path):
                if filename.endswith(".tsv"):
                    tsv_files.append(os.path.join(subdir_path, filename))

    # Collect records with deduplication
    records = []
    seen_pairs = defaultdict(set)  # Track seen pairs per style_type to ensure uniqueness

    for tsv_file in tsv_files:
        with open(tsv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')

            for row in reader:
                style_type = row.get('style type', '').strip().lower()

                if not style_type:
                    continue

                # Skip em-subj-pronoun
                if 'em-subj-pronoun' in style_type or 'em_subj_pronoun' in style_type:
                    continue

                # Extract pairs from this row
                pairs = extract_pairs_from_row(row, style_type)

                for most_style, least_style in pairs:
                    if most_style and least_style:
                        # Create a hashable key for deduplication
                        pair_key = (most_style.strip(), least_style.strip())

                        # Only add if we haven't seen this pair before
                        if pair_key not in seen_pairs[style_type]:
                            seen_pairs[style_type].add(pair_key)
                            # Create a record with text (ordered pair) and label (style type)
                            records.append({
                                'text': [most_style, least_style],
                                'label': style_type
                            })

    return records


def stel_record_handler(record):
    """
    Custom record handler for STEL dataset.
    Extracts 'text' and 'label' from each record.
    """
    return {
        'text': record.get('text'),
        'label': record.get('label')
    }