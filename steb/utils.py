"""
This module reads the application's configuration from environment variables,
config files, or defaults, and sets up global variables for the cache and processed data directories.
"""
import os
import configparser
from pathlib import Path
from termcolor import cprint

# Defaults
DEFAULT_CACHE_DIR = os.path.join(Path.home(), ".cache", "steb")
DEFAULT_PROCESSED_DATA_DIR = os.path.join(Path.home(), ".local", "share", "steb", "processed_datasets")
DEFAULT_RESULTS_DIR = "results" # Relative to CWD
DEFAULT_RAW_DATASETS_DIR = "raw_datasets" # Relative to CWD, though usually users should set this if not in repo root

def get_config_value(env_var: str, config_section: str, config_key: str, default: str) -> str:
    """
    Retrieves a configuration value from (priority):
    1. Environment Variable
    2. config.ini (in CWD)
    3. ~/.steb/config.ini
    4. Default value
    """
    # 1. Environment Variable
    if os.environ.get(env_var):
        return os.environ[env_var]
    
    # Check config files
    config = configparser.ConfigParser()
    # Read in reverse order of priority so later reads overwrite earlier ones?
    # Actually configparser.read can take a list. 
    # But we want CWD to override User Home.
    # So we read User Home first, then CWD.
    config_files = [
        os.path.join(Path.home(), ".steb", "config.ini"),
        "config.ini"
    ]
    config.read(config_files)

    if config.has_option(config_section, config_key):
        return config[config_section][config_key]
    
    return default

CACHE_DIR = get_config_value("STEB_CACHE_DIR", "Application_Paths", "cache_dir", DEFAULT_CACHE_DIR)
PROCESSED_DATA_DIR = get_config_value("STEB_PROCESSED_DATA_DIR", "Application_Paths", "processed_dataset_dir", DEFAULT_PROCESSED_DATA_DIR)
RESULTS_DIR = get_config_value("STEB_RESULTS_DIR", "Application_Paths", "results_dir", DEFAULT_RESULTS_DIR)
RAW_DATASETS_DIR = get_config_value("STEB_RAW_DATASETS_DIR", "Application_Paths", "raw_datasets_dir", DEFAULT_RAW_DATASETS_DIR)

cprint(f"Cache directory: ", "blue", end="")
cprint(f"{CACHE_DIR}", "white", "on_blue")
cprint(f"Processed data directory: ", "blue", end="")
cprint(f"{PROCESSED_DATA_DIR}", "white", "on_blue")
cprint(f"Results directory: ", "blue", end="")
cprint(f"{RESULTS_DIR}", "white", "on_blue")
cprint(f"Raw datasets directory: ", "blue", end="")
cprint(f"{RAW_DATASETS_DIR}", "white", "on_blue")

# Ensure directories exist (except valid raw_datasets which must be provided by user or exist)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
# We don't verify RAW_DATASETS_DIR here as it might not be needed for all commands (e.g. if dataset is already processed)
