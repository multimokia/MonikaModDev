import logging

class MASExtraPropLogAdapter(logging.LoggerAdapter):
    """
    Log adapter_ctor that enables defaulting of props on LogRecord objects.
    Use this if you need extra props.

    PROPERTES:
        extra_props - dictionary of extra props and their default values
    """

    def __init__(self, logger, extra_props):
        """
        IN:
            logger - the logger to adapt
            extra_props - dict of props to default on LogRecord objects.
                key: name of prop
                value: default value
        """
        super(MASExtraPropLogAdapter, self).__init__(logger, extra_props)

    def _add_extra_prop(self, prop_name, kwargs):
        """
        Adds a prop from the kwargs to the extra kwargs.
        Assumes extra is set.

        IN:
            prop_name - name of the prop to get from kwargs
            kwargs - should contain the prop data (if exists)

        OUT:
            kwargs - prop data moved to extra if found.
        """
        if prop_name not in kwargs:
            return

        kwargs["extra"][prop_name] = kwargs.pop(prop_name)

    def process(self, msg, kwargs):
        """
        Override of process.

        The main difference is to update the existing extra dict if it exists
        """
        self.set_extra(kwargs)
        return msg, kwargs

    def set_extra(self, kwargs):
        """
        Sets the extra kwarg with our extra data, updating existing if
        found. Also pulls extra props directly from the kwargs if
        those props are found.

        OUT:
            kwargs - the kwargs to set extra in
        """
        if "extra" in kwargs:
            new_extra = dict(self.extra)
            new_extra.update(kwargs["extra"])
            kwargs["extra"] = new_extra
        else:
            kwargs["extra"] = dict(self.extra)

        for prop_name in self.extra:
            self._add_extra_prop(prop_name, kwargs)