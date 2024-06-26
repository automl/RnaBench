import os
import yaml
import pathlib

from RnaBench.lib.rna_folding_algorithms.DL.ProbTransformer.prob_transformer.utils.handler.base_handler import Handler

"""
reads a yml config or a dict and safes it into experiment folder
"""


class AttributeDict(Handler):
    def __init__(self, dictionary, name):
        super().__init__()

        for key in dictionary:
            if isinstance(dictionary[key], dict):
                if not hasattr(self, "sub_config"):
                    self.sub_config = []
                self.sub_config.append(key)
                setattr(self, key, AttributeDict(dictionary[key], key))
            else:
                setattr(self, key, dictionary[key])

    def __repr__(self):
        return str(self.__dict__)

    def __str__(self):
        return str(self.__dict__)

    @property
    def get_dict(self):
        return self.__dict__

    @property
    def dict(self):
        return self.__dict__

    def set_attr(self, name, value):
        if isinstance(value, pathlib.Path):
            value = value.as_posix()
        self.__setattr__(name, value)


class ConfigHandler(AttributeDict):

    def __init__(self, config_file=None, config_dict=None):

        if config_file is None and config_dict is None:
            raise UserWarning("ConfigHandler: config_file and config_dict is None")

        elif config_file is not None and config_dict is None:
            with open(config_file, 'r') as f:
                config_dict = yaml.load(f, Loader=yaml.Loader)

        super().__init__(config_dict, "main")

        self.check_experiment_config()

    def check_experiment_config(self):
        if hasattr(self, "expt"):
            for attr_name in ['experiment_name']:
                if not hasattr(self.expt, attr_name):
                    raise UserWarning(f"ConfigHandler: {attr_name} is missing")
                elif isinstance(self.expt.__getattribute__(attr_name), str):
                    self.expt.__setattr__(attr_name, str(self.expt.__getattribute__(attr_name)))

    def save_config(self, dir, file_name="config.yml"):
        dir = pathlib.Path(dir)
        self.save_mkdir(dir)
        if os.path.isfile(dir / file_name):
            file_name = self.counting_name(dir, file_name, suffix=True)
        with open(dir / file_name, 'w+') as f:
            config_dict = self.get_dict
            yaml.dump(config_dict, f, default_flow_style=False, encoding='utf-8')
        return dir / file_name
