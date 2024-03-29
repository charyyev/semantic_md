import datetime
import os
import re

import yaml


class Config:
    """
    Config class that merges base and personal config file.
    Can read in config files recursively (as specified by the extends keyword)
    Has some useful functions for ease of use.
    """

    def __init__(self, file=None):
        self.timestamp = datetime.datetime.now().strftime("%Y-%m-%d__%H-%M-%S")

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
        self._add_additional_info()

    def _add_additional_info(self):
        additions = {"timestamp": self.timestamp}
        self._add_metadata(additions)

    def _add_metadata(self, additions: dict):
        """
        adds certain metadata to the config for later reading (typically timestamp,
        git hash, etc.)
        """
        if "metadata" not in self._config:
            self._config["metadata"] = {}

        for k, v in additions.items():
            if k in self._config["metadata"]:
                raise ValueError(
                    f"Please do not specify a {k} field in the metadata field of the "
                    f"config, as it is later added in post-processing"
                )
            else:
                self._config["metadata"][k] = v

    def get_subpath(self, subpath):
        """
        Computes path relative to source directory path.
        """
        subpath_dict = self._config["subpaths"]
        if subpath not in list(subpath_dict.keys()):
            raise ValueError(f"Subpath {subpath} not known.")
        base_path = os.path.normpath(self._config["project_root_dir"])
        path_ending = os.path.normpath(subpath_dict[subpath])
        # return absolute path is that is specified, else concat with root dir
        if os.path.isabs(path_ending):
            return path_ending
        return os.path.join(base_path, path_ending)

    def build_subpath(self, subpath):
        """
        Computes path relative to source directory path.
        """
        base_path = os.path.normpath(self._config["project_root_dir"])
        path_ending = os.path.normpath(subpath)
        return os.path.join(base_path, path_ending)

    def _replace_configs_in_note(self, match: re.Match):
        """
        Does the dynamic replacement
        """
        text = match.group(1)
        pattern = r"\[(.+?)\]"
        matches = re.findall(pattern, text)

        current_config = self._config
        for key in list(matches):
            if key in current_config:
                current_config = current_config[key]
            else:
                raise ValueError(f"{text} does not exists in config file")

        return f"{list(matches)[-1]}~~{current_config}"

    def _build_note(self):
        """
        Adaptive naming of runs if specified in config.
        Example config[hyperparameters][train][batch_size] is replaced with
        batch_size~~<batch_size> (e.g. batch_size~~32)
        """
        text = self._config["note"]
        pattern = r"(config(?:\[[^\]]+\])+)"
        return re.sub(pattern, self._replace_configs_in_note, text)

    def get_name_stem(self):
        """
        Builds name stem for saving models with unified naming scheme.
        """
        name = self._config["model_type"]
        note = self._build_note()
        name_stem = f"model_{name}_{note}___{self.timestamp}"
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
