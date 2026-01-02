from logs.adapters.MASExtraPropLogAdapter import MASExtraPropLogAdapter

class MASNewlineLogAdapter(MASExtraPropLogAdapter):
    """
    Log adapter_ctor with an option for newline kwargs.
    The newline kwarg is pfx_newline.
    """

    def __init__(self, logger, extra_props=None, newline_def=False):
        """
        IN:
            logger - the logger to adapt
            extra_props - additional props, other than the newline one.
                Optional.
                (Default: None)
            newline_def - the default value for the newline prop
                (Default: False)
        """
        if extra_props is None:
            extra_props = {}

        extra_props["pfx_newline"] = newline_def
        super(MASNewlineLogAdapter, self).__init__(logger, extra_props)