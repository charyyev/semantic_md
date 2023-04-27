import datetime
import os
from copy import deepcopy

import yaml


class Config:
    """
    Config class that merges base and personal config file.
    Has some useful functions for ease of use.
    """

    def __init__(self, file=None):
        if file is None:
            file = "user.yaml"
        else:
            file = f"{file}.yaml"

        def load_recursive(config, stack):
            if config in stack:
                raise AssertionError("Attempting to build recursive configuration.")

            config_path = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            config_path = os.path.join(config_path, "configs", config)
            with open(config_path, "r", encoding="UTF-8") as file_handle:
                cfg = yaml.safe_load(file_handle)

            base = (
                {}
                if "extends" not in cfg
                else load_recursive(cfg["extends"], stack + [config])
            )
            base = _recursive_update(base, cfg)
            return base

        self._config = load_recursive(file, [])

    def get_subpath(self, subpath):
        subpath_dict = self._config["subpaths"]
        if subpath not in list(subpath_dict.keys()):
            raise ValueError(f"Subpath {subpath} not known.")
        base_path = os.path.normpath(self._config["project_root_dir"])
        path_ending = os.path.normpath(subpath_dict[subpath])
        return os.path.join(base_path, path_ending)

    def build_subpath(self, subpath):
        base_path = os.path.normpath(self._config["project_root_dir"])
        path_ending = os.path.normpath(subpath)
        return os.path.join(base_path, path_ending)

    def get_name_stem(self):
        name = self._config["model_type"]
        note = self._config["note"]
        now = datetime.datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
        name_stem = f"model_{name}_{note}___{now}"
        return name_stem

    def __getitem__(self, item):
        return self._config.__getitem__(item)

    def __setitem__(self, key, value):
        return self._config.__setitem__(key, value)

    def get_config(self):
        return self._config


def _recursive_update(base: dict, cfg: dict) -> dict:
    for k, v in cfg.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            base[k] = _recursive_update(base[k], v)
        else:
            base[k] = v
    return base
