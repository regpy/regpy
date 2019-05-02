"""Misc helper functions."""

from .tests import test_adjoint
from .cg_methods import CGNE_reg
from .cg_methods import CG

from functools import wraps
from logging import getLogger


@property
def classlogger(self):
    return getLogger(type(self).__qualname__)


def memoized_property(prop):
    attr = '__memoized_' + prop.__qualname__

    @property
    @wraps(prop)
    def mprop(self):
        try:
            return getattr(self, attr)
        except AttributeError:
            setattr(self, attr, prop(self))
            return getattr(self, attr)

    return mprop
