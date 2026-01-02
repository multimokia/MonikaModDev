from .MASLogFormatter import MASLogFormatter

class MASNewlineLogFormatter(MASLogFormatter):
    """
    log formatter with newline prefix support.
    """

    def apply_newline_prefix(self, record, msg):
        """
        Applies a newline to a msg if the record supports it.
        The newline is prefixed to the start of the message.

        IN:
            record - LogRecord to generate the format for
            msg - the currently formatted message

        RETURNS: msg with newline if appropriate
        """
        try:
            if record.pfx_newline:
                return "\n" + msg
        except:
            pass
        return msg

    def format(self, record):
        """
        Applies a prefix newline if appropriate.
        """
        return self.apply_newline_prefix(
            record,
            super().format(record)
        )
