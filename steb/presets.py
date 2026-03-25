from typing import Any, Dict

from .core import get_supported_datasets, get_supported_tasks


def get_exhaustive_config() -> Dict[str, Any]:
    """
    Returns a configuration that runs all tasks on all datasets.
    """
    tasks = get_supported_tasks()

    args_dict = {
        "order_alignment": {
            "episode_sizes": [1],
            "n_episodes_per_class": 2
        },
        # "retrieval": {
        #     "episode_sizes": [1, 2, 5],
        #     "n_episodes_per_class": 1
        # },
        "clustering": {
            "episode_sizes": [1, 2, 5],
            "n_episodes_per_class": 100
        },
        "all_to_all_pair_classification": {
            "episode_sizes": [1, 2, 5],
            "n_episodes_per_class": 100
        },
        "pre_defined_pair_classification": {
            "episode_sizes": [1],
            "n_episodes_per_class": 2
        }
    }

    to_return = []
    for task in tasks:
        if task == "retrieval":
            continue

        supported_datasets = get_supported_datasets(task)

        for dataset in supported_datasets:
            if "fisher" in dataset:
                continue

            to_return.append({
                "task": task,
                "datasets": [dataset],
                **args_dict[task]
            })
    return {"config": {"tasks": to_return}}


def get_moderate_config() -> Dict[str, Any]:
    """Returns a configuration that runs each dataset once with a set of pre-defined arguments."""
    tasks = get_supported_tasks()

    args_dict = {
        "order_alignment": {
            "episode_sizes": [1],
            "n_episodes_per_class": 100,
        },
        # "retrieval": {
        #     "episode_sizes": [50],
        #     "n_episodes_per_class": 1
        # },
        "clustering": {
            "episode_sizes": [1],
            "n_episodes_per_class": 100
        },
        "all_to_all_pair_classification": {
            "episode_sizes": [1],
            "n_episodes_per_class": 100
        },
        "pre_defined_pair_classification": {
            "episode_sizes": [1],
            "n_episodes_per_class": 2
        }
    }

    to_return = []
    for task in tasks:
        if task == "retrieval":
            continue

        supported_datasets = get_supported_datasets(task)

        for dataset in supported_datasets:
            if "dummy" in dataset:
                continue
            if "fisher" in dataset:
                continue

            to_return.append({
                "task": task,
                "datasets": [dataset],
                **args_dict[task]
            })
    return {"config": {"tasks": to_return}}

PRESETS = {
    "sanity": {
        "description": "Quick sanity check using dummy datasets.",
        "config": {
            "tasks": [
                {
                    "task": "order_alignment", 
                    "datasets": ["dummy_order_alignment"],
                    "episode_sizes": [1],
                    "n_episodes_per_class": 2
                },
                {
                    "task": "retrieval", 
                    "datasets": ["dummy_retrieval"],
                    "episode_sizes": [50],
                    "n_episodes_per_class": 1
                }
            ]
        }
    },
    "fast": {
        "description": "Evaluates on a single dataset per task.",
        "config": {
            "tasks": [
                {
                    "task": "order_alignment", 
                    "datasets": ["corpus-of-diverse-styles"],
                    "episode_sizes": [2],
                    "n_episodes_per_class": 50,
                },
                {
                    "task": "retrieval", 
                    "datasets": ["dummy_retrieval"], # TODO: Replace with a real dataset once we've fixed the speed issue.
                    "episode_sizes": [50],
                    "n_episodes_per_class": 1
                },
                {
                    "task": "clustering", 
                    "datasets": ["sms_spam"],
                    "episode_sizes": [1],
                    "n_episodes_per_class": 100,
                },
                {
                    "task": "all_to_all_pair_classification", 
                    "datasets": ["corpus-of-diverse-styles"],
                    "episode_sizes": [1, 5],
                    "n_episodes_per_class": 100,
                },
                {
                    "task": "pre_defined_pair_classification", 
                    "datasets": ["pan15_authorship_verification_english_test"],
                    "episode_sizes": [1],
                    "n_episodes_per_class": 2,
                }
            ]
        }
    },
    "moderate": {
        "description": "Run each dataset once with a set of pre-defined arguments.",
        "func": get_moderate_config
    },
    "exhaustive": {
        "description": "Run all available tasks on all available datasets with a set of pre-defined arguments.",
        "func": get_exhaustive_config
    }
}



def resolve_preset(preset_name: str) -> Dict[str, Any]:
    """
    Resolves a preset name to a configuration dictionary.

    Args:
        preset_name: The name of the preset to resolve.

    Returns:
        A dict containing a "config" key with "tasks" list.
    """
    if preset_name not in PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}. Available presets: {list(PRESETS.keys())}")

    preset = PRESETS[preset_name]

    if "func" in preset:
        return preset["func"]()

    return preset
