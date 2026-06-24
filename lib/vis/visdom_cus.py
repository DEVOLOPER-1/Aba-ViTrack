# import visdom  # Commented out to avoid dependency issues
import numpy as np


class Visdom(object):
    """Dummy class to replace the missing visdom library."""

    def __init__(self, *args, **kwargs):
        pass

    def register(self, *args, **kwargs):
        # Silently ignore all visualization requests
        pass

    def __getattr__(self, name):
        # Silently ignore any other method calls
        return lambda *args, **kwargs: None