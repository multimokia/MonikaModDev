from contracts.renpy import Config

CONFIG: Config = None


def init(config: Config):
    global CONFIG
    CONFIG = config