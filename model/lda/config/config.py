import logging

from singleton_decorator import singleton
import yaml


class InvalidConfigException(Exception):
    pass


@singleton
class LdaConfig(object):
    def __init__(self, profile):
        self.profile = profile
        self.log = logging.getLogger('lda_model')
        self.current_config = self.get_current_config()

    def get_current_config(self):
        cfg = None
        with open("model/lda/config/config.yml", 'r') as ymlfile:
            cfg = yaml.load(ymlfile)

        if cfg is not None:
            self.log.info("Succesfully loaded configuration")
            return cfg[self.profile]

        if self.current_config is not None:
            self.log.warning("Failed to load new configuration. returning last valid config")
            return self.current_config
        else:
            raise InvalidConfigException("Could not load new config. No existing config available either")
