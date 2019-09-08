from functools import wraps
from logging import getLogger
import numpy as np

from .get_directions import get_directions

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
            pass
        setattr(self, attr, prop(self))
        return getattr(self, attr)

    return mprop


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
    if np.issubdtype(x.dtype, np.floating) and x.flags.c_contiguous:
        return x.view(dtype=np.result_type(1j, x))[..., 0]
    else:
        z = np.array(x[..., 0], dtype=np.result_type(1j, x))
        z.imag = x[..., 1]
        return z


def is_real_dtype(obj):
    try:
        dtype = obj.dtype
    except AttributeError:
        dtype = np.dtype(obj)
    return (np.issubdtype(dtype, np.number) and not
            np.issubdtype(dtype, np.complexfloating))


def is_complex_dtype(obj):
    try:
        dtype = obj.dtype
    except AttributeError:
        dtype = np.dtype(obj)
    return np.issubdtype(dtype, np.complexfloating)


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


def make_repr(self, *args, **kwargs):
    arglist = []
    for arg in args:
        arglist.append(repr(arg))
    for k, v in sorted(kwargs.items()):
        arglist.append("{}={}".format(repr(k), repr(v)))
    return '{}({})'.format(type(self).__qualname__, ', '.join(arglist))


eps = np.finfo(float).eps
