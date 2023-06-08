"""Utility functions for use in the library."""
from functools import wraps
from typing import Callable


def spread(fn: Callable):
    """Creates a new function from the given function so that it takes one
    dict (PyTree) and spreads the arguments."""

    @wraps(fn)
    def inner(kwargs):
        return fn(**kwargs)

    return inner
