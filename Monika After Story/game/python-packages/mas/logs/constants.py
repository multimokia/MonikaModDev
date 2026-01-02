import logging

# Map for filename : Logger
LOG_MAP = dict()

# log tags
LT_INFO = "info"
LT_WARN = "Warning! ;_;"
LT_ERROR = "!ERROR! T_T"

LT_MAP = {
    logging.INFO: LT_INFO,
    logging.WARN: LT_WARN,
    logging.ERROR: LT_ERROR,
}

# Consts
DEF_FMT = "[%(asctime)s] [%(levelname)s]: %(message)s"
DEF_DATEFMT = "%Y-%m-%d %H:%M:%S"

# Add the header to each log, including OS info + MAS version number
# NOTE: python logging does not auto handle CRLF, so we need to explicitly manage that for the header
LOG_HEADER = "\n\n{_date}\n{system_info}\n{renpy_ver}\n\nVERSION: {game_ver}\n{separator}"

# Unformatted logs use these consts (spj/pnm)
MSG_INFO = "[" + LT_INFO + "]: {0}"
MSG_WARN = "[" + LT_WARN + "]: {0}"
MSG_ERR = "[" + LT_ERROR + "]: {0}"

MSG_INFO_ID = "    " + MSG_INFO
MSG_WARN_ID = "    " + MSG_WARN
MSG_ERR_ID = "    " + MSG_ERR

# Load strs for files
LOAD_TRY = "Attempting to load '{0}'..."
LOAD_SUCC = "'{0}' loaded successfully."
LOAD_FAILED = "Load failed."

JSON_LOAD_FAILED = "Failed to load json at '{0}'."
FILE_LOAD_FAILED = "Failed to load file at '{0}'. | {1}"
NAME_BAD = "name must be unique."