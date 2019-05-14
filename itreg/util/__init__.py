from collections import namedtuple
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


def named(names, *values):
    if names is None:
        return tuple(values)
    elif isinstance(names, type):
        return names(*values)
    else:
        return getnamedtuple(names)(*values)


def getnamedtuple(*names):
    names = tuple(names)
    try:
        return getnamedtuple._cache[names]
    except KeyError:
        cls = namedtuple('NamedTuple', *names)
        getnamedtuple._cache[names] = cls
        return cls
getnamedtuple._cache = {}


def set_defaults(params, **defaults):
    defaults.update(params)
    return defaults


def complex2real(z, axis=-1):
    assert is_complex_dtype(z.dtype)
    if z.flags.c_contiguous:
        x = z.view(dtype=z.real.dtype).reshape(z.shape + (2,))
    else:
        # TODO Does this actually work in all cases, or do we have to perform a
        # copy here?
        x = np.lib.stride_tricks.as_strided(
            z.real, shape=z.shape + (2,),
            strides=z.strides + (z.real.dtype.itemsize,))
    return np.moveaxis(x, -1, axis)


def real2complex(x, axis=-1):
    assert is_real_dtype(x.dtype)
    assert x.shape[axis] == 2
    x = np.moveaxis(x, axis, -1)
    if x.dtype.kind == 'f' and x.flags.c_contiguous:
        return x.view(dtype=np.result_type(1j, x))[..., 0]
    else:
        z = np.array(x[..., 0], dtype=np.result_type(1j, x))
        z.imag = x[..., 1]
        return z


def is_real_dtype(dtype):
    return np.dtype(dtype).kind in 'biuf'


def is_complex_dtype(dtype):
    return np.dtype(dtype).kind == 'c'


def is_uniform(x):
    x = np.asarray(x)
    assert x.ndim == 1
    diffs = x[1:] - x[:-1]
    return np.allclose(diffs, diffs[0])


def linspace_circle(num, *, start=0, stop=None, endpoint=False):
    if not stop:
        stop = start + 2*np.pi
    angles = np.linspace(start, stop, num, endpoint)
    return np.stack((np.cos(angles), np.sin(angles)), axis=1)
