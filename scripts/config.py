"""
Central configuration helper for VayuDrishti data pipeline.
"""
import os
import json
from pathlib import Path

# Base directory is the workspace root
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "configs" / "pipeline_config.json"

def load_config(config_path=None):
    if config_path is None:
        config_path = CONFIG_PATH
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config

def get_path(key, config=None):
    if config is None:
        config = load_config()
    rel_path = config["paths"][key]
    return str(BASE_DIR / rel_path)

if __name__ == "__main__":
    cfg = load_config()
    print("Configuration loaded successfully:")
    print("Project:", cfg["project_name"])
    print("IBTrACS Raw:", get_path("ibtracs_raw", cfg))
