from functools import wraps
from logging import getLogger
import numpy as np


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


def complex2real(z, axis=-1):
    assert z.dtype.kind == 'c'
    x = np.lib.stride_tricks.as_strided(
        z.real, shape=z.shape + (2,),
        strides=z.strides + (z.real.dtype.itemsize,))
    return np.moveaxis(x, -1, axis)


def real2complex(x, axis=-1):
    assert x.dtype.kind != 'c'
    assert x.shape[axis] == 2
    x = np.moveaxis(x, axis, -1)
    if x.flags.c_contiguous:
        return x.view(dtype=np.result_type(1j, x))[..., 0]
    else:
        z = np.array(x[..., 0], dtype=np.result_type(1j, x))
        z.imag = x[..., 1]
        return z
