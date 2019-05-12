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


def set_defaults(params, **defaults):
    defaults.update(params)
    return defaults


def complex2real(z, axis=-1):
    assert z.dtype.kind == 'c'
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
    assert x.dtype.kind != 'c'
    assert x.shape[axis] == 2
    x = np.moveaxis(x, axis, -1)
    if x.flags.c_contiguous:
        return x.view(dtype=np.result_type(1j, x))[..., 0]
    else:
        z = np.array(x[..., 0], dtype=np.result_type(1j, x))
        z.imag = x[..., 1]
        return z


def realdot(a, b):
    if a.dtype.kind == b.dtype.kind == 'c':
        if a.flags.c_contiguous and b.flags.c_contiguous:
            # This is an optimization: iterating through contiguous memory once
            # may be faster than twice, as done in the `else` branch.
            return np.vdot(a.view(dtype=a.real.dtype), b.view(dtype=b.real.dtype))
        else:
            return np.vdot(a.real, b.real) + np.vdot(a.imag, b.imag)
    else:
        return np.vdot(a.real, b.real)


def realmul(a, b):
    """Elementwise product of complex arrays implicitly considered as real
    arrays of double dimension:

        realmut(complex(x1, y1), complex(x2, y2)) == complex(x1*x2, y1*y1)
    """
    a = np.asarray(a)
    b = np.asarray(b)
    if is_complex_dtype(a.dtype) and is_complex_dtype(b.dtype):
        return a.real * b.real + 1j * a.imag * b.imag
    else:
        return a.real * b.real


def getnamedtuple(*names):
    names = tuple(names)
    try:
        return getnamedtuple._cache[names]
    except KeyError:
        cls = namedtuple('NamedTuple', *names)
        getnamedtuple._cache[names] = cls
        return cls
getnamedtuple._cache = {}


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
