import datetime
import logging
import os
import platform

from .constants import LOG_HEADER, LOG_MAP
import logging.handlers as loghandlers

from .formatters.MASLogFormatter import MASLogFormatter


#Full logging info
def init_log(name, append=True, formatter=None, adapter_ctor=None, header=None, rotations=5, LOG_MAXSIZE_B=None):
    """
    Initializes a logger with a handler with the name and files given.

    IN:
        name - name of the logger, this will be the same as the file, with the file appending '.txt'
        append - Whether or not we're appending this log or clearing it on load
            (Default: True)
        formatter - custom logging.Formatter to be used.
            If None is provided, the default MASLogFormatter is used.
            (Default: None)
        adapter_ctor - Constructor reference to the adapter we want to use. If None, no adapter is used
            (Default: None)
        header - Header block for logs to use. If None, the default header printing version info is used. If False, no header is used.
            (Default: None)
        rotations - Integer representing the amount of log rotations we should have. If 0, no rotations are used.
            (Default: 5)

    NOTE: ALL LOGS ARE IN renpy.config.basedir/log/
    All logs flush and rotate once they're 5 mb in size.
    """
    _kwargs = {
        "filename": os.path.join(LOG_PATH, name + '.log'),
        "mode": ("a" if append else "w"),
        "encoding": "utf-8",
        "delay": header is False  #We auto delay here if no header to only gen the file once we need to
    }

    #Setup header
    if header is None:
        header = LOG_HEADER

    if append:
        handler = loghandlers.RotatingFileHandler(
            maxBytes=LOG_MAXSIZE_B,
            backupCount=rotations,
            **_kwargs
        )
    else:
        handler = logging.FileHandler(**_kwargs)

    log = logging.getLogger(name)

    #Allow all severities to be logged
    log.setLevel(logging.DEBUG)
    handler.setLevel(logging.DEBUG)

    #Add the handler so we can print log header info
    log.addHandler(handler)

    #Write as this has no formatting yet
    if header is not False:
        log.info(
            header.format(
                _date=datetime.datetime.now().strftime("%a %b %d %H:%M:%S %Y"),
                system_info="{0} {1} - build: {2}".format(platform.system(), platform.release(), platform.version()),
                renpy_ver=renpy.version(),
                game_ver=renpy.config.version,
                separator="=" * 50
            )
        )

    if formatter is None:
        handler.setFormatter(MASLogFormatter())

    else:
        #Now apply formatting to all further uses
        handler.setFormatter(formatter)

    if adapter_ctor is not None:
        log = adapter_ctor(log)

    #Add it to the map
    LOG_MAP[name] = log

    return log


def get_log(name):
    """
    Gets a log from the log map

    IN:
        name - log name

    RETURNS: log, or None if no log
    """
    return LOG_MAP.get(name)


def is_inited(name):
    """
    Checks if a log has been inited

    IN:
        name - log name
    """
    return name in LOG_MAP
