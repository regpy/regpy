"""Misc helper functions."""

from .tests import test_adjoint
from .cg_methods import CGNE_reg
from .cg_methods import CG

from logging import getLogger


@property
def classlogger(self):
    try:
        return self.__logger
    except AttributeError:
        self.__logger = getLogger(type(self).__qualname__)
        return self.__logger


def instantiate(cls):
    return cls()


@instantiate
class emptycontext:
    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass
