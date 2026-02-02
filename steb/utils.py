
"""
This module reads the application's configuration from the config.ini file
and sets up global variables for the cache and processed data directories.
"""
import configparser
import os

# Get the directory where this module is located (steb/)
module_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to get the STEB root directory
steb_root = os.path.dirname(module_dir)
# Construct path to config.ini relative to the module location
config_file_path = os.path.join(steb_root, "config.ini")

config = configparser.ConfigParser()
config.read(config_file_path)
CACHE_DIR = config["Application_Paths"]["cache_dir"]
PROCESSED_DATA_DIR = config["Application_Paths"]["processed_dataset_dir"]
