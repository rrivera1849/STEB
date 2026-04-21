"""
Config validation for STEB dataset config.json files.

Validates required fields, type-specific requirements, task/processor
mappings, and record handler configuration.
"""
import importlib
import json
import os
from typing import Dict, List, Tuple

from termcolor import colored

from .core import SUPPORTED_TASKS

VALID_DATASET_TYPES = ("huggingface", "custom")

VALID_PROCESSORS = (
    "probing",
)


def validate_config(
    config: Dict,
    config_path: str = "<unknown>",
) -> List[str]:
    """
    Validates a single dataset config.json dictionary.

    Args:
        config: The parsed JSON config dictionary.
        config_path: Path to the config file (for error messages).

    Returns:
        A list of error strings. Empty list means the config is valid.
    """
    errors = []

    # Required top-level fields
    if "type" not in config:
        errors.append("Missing required field: 'type'")
    elif config["type"] not in VALID_DATASET_TYPES:
        errors.append(f"Invalid type '{config['type']}'. Must be one of: {VALID_DATASET_TYPES}")

    if "record_handler" not in config:
        errors.append("Missing required field: 'record_handler'")
    else:
        rh = config["record_handler"]
        has_custom = "custom_record_handler_function" in rh
        has_text_getter = "text_getter" in rh
        has_label_getter = "label_getter" in rh

        if not has_text_getter:
            errors.append("record_handler missing 'text_getter'")
        if not has_label_getter and not has_custom:
            errors.append("record_handler must have 'label_getter' or 'custom_record_handler_function'")

    if "tasks" not in config:
        errors.append("Missing required field: 'tasks'")
    elif not config["tasks"]:
        errors.append("'tasks' must contain at least one task")
    else:
        for task_name, task_config in config["tasks"].items():
            if task_name not in SUPPORTED_TASKS:
                errors.append(f"Unknown task: '{task_name}'. Supported: {SUPPORTED_TASKS}")
            if "processor" in task_config and task_config["processor"] not in VALID_PROCESSORS:
                errors.append(f"Unknown processor '{task_config['processor']}' in task '{task_name}'")
            if "record_handler" in task_config:
                task_rh = task_config["record_handler"]
                for key in task_rh:
                    if key not in ("text_getter", "label_getter", "label_getter_function", "custom_record_handler_function"):
                        errors.append(f"Task '{task_name}' record_handler has unknown key: '{key}'")

    # Type-specific validation
    if config.get("type") == "huggingface":
        if "loader_kwargs" not in config:
            errors.append("HuggingFace datasets require 'loader_kwargs'")
        else:
            lk = config["loader_kwargs"]
            if "path" not in lk:
                errors.append("loader_kwargs missing 'path'")
            if "split" not in lk:
                errors.append("loader_kwargs missing 'split'")

    elif config.get("type") == "custom":
        if "loader_function" not in config:
            errors.append("Custom datasets require 'loader_function'")
        if "data_dir" not in config:
            errors.append("Custom datasets require 'data_dir'")

    return errors


def validate_all_configs() -> Tuple[int, int]:
    """
    Validates all dataset config.json files in the steb_datasets directory.

    Returns:
        A tuple of (num_valid, num_invalid) counts.
    """
    package_dir = os.path.dirname(os.path.abspath(__file__))
    datasets_dir = os.path.join(package_dir, "steb_datasets")

    num_valid = 0
    num_invalid = 0

    for entry in sorted(os.scandir(datasets_dir), key=lambda e: e.name):
        if not entry.is_dir():
            continue
        config_path = os.path.join(entry.path, "config.json")
        if not os.path.exists(config_path):
            continue

        with open(config_path, "r") as f:
            try:
                config = json.load(f)
            except json.JSONDecodeError as e:
                print(colored(f"  INVALID {entry.name}: malformed JSON - {e}", "red"))
                num_invalid += 1
                continue

        errors = validate_config(config, config_path)
        if errors:
            print(colored(f"  INVALID {entry.name}:", "red"))
            for err in errors:
                print(colored(f"    - {err}", "red"))
            num_invalid += 1
        else:
            print(colored(f"  OK {entry.name}", "green"))
            num_valid += 1

    return num_valid, num_invalid
