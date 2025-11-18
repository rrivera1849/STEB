
"""
This module reads the application's configuration from the config.ini file
and sets up global variables for the cache and processed data directories.
"""
import configparser
config = configparser.ConfigParser()
config.read("config.ini")
CACHE_DIR = config["Application_Paths"]["cache_dir"]
PROCESSED_DATA_DIR = config["Application_Paths"]["processed_dataset_dir"]
