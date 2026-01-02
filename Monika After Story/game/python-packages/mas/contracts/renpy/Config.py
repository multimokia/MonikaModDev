from typing import Protocol


class Config(Protocol):
    gamedir = ""
    basedir = ""
    renpy_base = ""
    version = ""