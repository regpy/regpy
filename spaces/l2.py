import logging

from .space import Space

__all__ = ['L2']

log = logging.getLogger(__name__)


class L2(Space):
    def __init__(self, *shape):
        super().__init__(shape, log)

    def gram(self, x):
        return x

    def gram_inv(self, x):
        return x
