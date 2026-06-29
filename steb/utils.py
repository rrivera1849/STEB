"""
This module reads the application's configuration from environment variables,
config files, or defaults, and sets up global variables for the processed data,
results, and raw datasets directories. The HuggingFace ``load_dataset`` cache is
left to HuggingFace itself (configurable via the standard ``HF_DATASETS_CACHE``
and ``HF_HOME`` environment variables).
"""
import os
import configparser
from pathlib import Path
from termcolor import cprint

# Defaults
DEFAULT_PROCESSED_DATA_DIR = os.path.join(Path.home(), ".local", "share", "steb", "processed_datasets")
DEFAULT_RESULTS_DIR = "results" # Relative to CWD
# DEFAULT_RAW_DATASETS_DIR is relative to CWD (repository root), matching the README expectation of ./raw_datasets
DEFAULT_RAW_DATASETS_DIR = "raw_datasets"

def get_config_value(
    env_var: str,
    config_section: str,
    config_key: str,
    default: str,
) -> str:
    """
    Retrieves a configuration value from (priority):
    1. Environment Variable
    2. config.ini (in CWD)
    3. ~/.steb/config.ini
    4. Default value
    """
    if os.environ.get(env_var):
        return os.environ[env_var]

    # CWD config overrides user home config (later reads take priority)
    config = configparser.ConfigParser()
    config_files = [
        os.path.join(Path.home(), ".steb", "config.ini"),
        "config.ini"
    ]
    config.read(config_files)

    if config.has_option(config_section, config_key):
        return config[config_section][config_key]

    return default

PROCESSED_DATA_DIR = get_config_value("STEB_PROCESSED_DATA_DIR", "Application_Paths", "processed_dataset_dir", DEFAULT_PROCESSED_DATA_DIR)
RESULTS_DIR = get_config_value("STEB_RESULTS_DIR", "Application_Paths", "results_dir", DEFAULT_RESULTS_DIR)
RAW_DATASETS_DIR = get_config_value("STEB_RAW_DATASETS_DIR", "Application_Paths", "raw_datasets_dir", DEFAULT_RAW_DATASETS_DIR)

cprint("Processed data directory: ", "blue", end="")
cprint(PROCESSED_DATA_DIR, "white", "on_blue")
cprint("Results directory: ", "blue", end="")
cprint(RESULTS_DIR, "white", "on_blue")
cprint("Raw datasets directory: ", "blue", end="")
cprint(RAW_DATASETS_DIR, "white", "on_blue")

# Ensure directories exist (except valid raw_datasets which must be provided by user or exist)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
# We don't verify RAW_DATASETS_DIR here as it might not be needed for all commands (e.g. if dataset is already processed)
