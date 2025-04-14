import os
import yaml

def load_config(config_path="config.yml"):
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            return config