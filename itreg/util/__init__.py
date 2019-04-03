"""Misc helper functions."""

from .tests import test_adjoint
from .cg_methods import CGNE_reg
from .cg_methods import CG

from logging import getLogger


@property
def classlogger(self):
    cls = type(self)
    try:
        return cls.__logger
    except AttributeError:
        cls.__logger = getLogger(cls.__qualname__)
        return cls.__logger


def instantiate(cls):
    return cls()


@instantiate
class emptycontext:
    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass
