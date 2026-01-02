import re

import logging
from logs.constants import DEF_FMT, DEF_DATEFMT, LT_MAP


class MASLogFormatter(logging.Formatter):
    """
    log formatter all other mas logs should extend if they want
    custom functionality.

    Features:
        - uses our own log tags
        - defaults the format with time and level name
    """
    NEWLINE_MATCHER = re.compile(r"(?<!\r)\n")
    LINE_TERMINATOR = "\r\n"

    def __init__(self, fmt=None, datefmt=None):
        if fmt is None:
            fmt = DEF_FMT
        if datefmt is None:
            datefmt = DEF_DATEFMT

        super().__init__(fmt=fmt, datefmt=datefmt)

    def format(self, record):
        """
        Override of format - mainly replaces the levelname prop
        """
        self.update_levelname(record)
        # return self.replace_lf(
        #     super().format(record)
        # )
        return super().format(record)

    def update_levelname(self, record):
        """
        Updates the levelname of the record. Use in custom formatter
        functions.
        """
        record.levelname = LT_MAP.get(record.levelno, record.levelname)

    # @classmethod
    # def replace_lf(cls, msg):
    #     """
    #     Replaces all line feeds with carriage returns and a line feed
    #     """
    #     return re.sub(cls.NEWLINE_MATCHER, cls.LINE_TERMINATOR, msg)