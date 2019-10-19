from functools import wraps
from logging import getLogger

import numpy as np
from scipy.spatial.qhull import Voronoi


@property
def classlogger(self):
    """The [`logging.Logger`][1] instance. Every subclass has a separate instance, named by its
    fully qualified name. Subclasses should use it instead of `print` for any kind of status
    information to allow users to control output formatting, verbosity and persistence.

    [1]: https://docs.python.org/3/library/logging.html#logging.Logger
    """
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
    if params is not None:
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
    if np.isscalar(obj):
        obj = np.asarray(obj)
    try:
        dtype = obj.dtype
    except AttributeError:
        dtype = np.dtype(obj)
    return (
        np.issubdtype(dtype, np.number) and not np.issubdtype(dtype, np.complexfloating)
    )


def is_complex_dtype(obj):
    if np.isscalar(obj):
        obj = np.asarray(obj)
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
        stop = start + 2 * np.pi
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


def bounded_voronoi(nodes, left, down, up, right):
    """Computes the Voronoi diagram with a bounding box
    """

    # Extend the set of nodes by reflecting along boundaries
    nodes_left = 2 * np.array([left - 1e-6, 0]) - nodes
    nodes_down = 2 * np.array([0, down - 1e-6]) - nodes
    nodes_right = 2 * np.array([right + 1e-6, 0]) - nodes
    nodes_up = 2 * np.array([0, up + 1e-6]) - nodes

    # Compute the extended Voronoi diagram
    evor = Voronoi(np.concatenate([nodes, nodes_up, nodes_down, nodes_left, nodes_right]))

    # Shrink the Voronoi diagram
    regions = [evor.regions[reg] for reg in evor.point_region[:nodes.shape[0]]]
    used_vertices = np.unique([i for reg in regions for i in reg])
    regions = [[np.where(used_vertices == i)[0][0] for i in reg] for reg in regions]
    vertices = [evor.vertices[i] for i in used_vertices]

    return regions, vertices


def broadcast_shapes(*shapes):
    a = np.ones((max(len(s) for s in shapes), len(shapes)), dtype=int)
    for i, s in enumerate(shapes):
        a[-len(s):, i] = s
    result = np.max(a, axis=1)
    for r, x in zip(result, a):
        if np.any((x != 1) & (x != r)):
            raise ValueError('Shapes can not be broadcast')
    return result


def trig_interpolate(val, n):
    # TODO get rid of fftshift
    """Computes `n` Fourier coeffients to the point values given by by `val`
    such that `ifft(fftshift(coeffs))` is an interpolation of `val`.
    """
    if n % 2 != 0:
        ValueError('n should be even')
    N = len(val)
    coeffhat = np.fft.fft(val)
    coeffs = np.zeros(n, dtype=complex)
    if n >= N:
        coeffs[:N // 2] = coeffhat[:N // 2]
        coeffs[-(N // 2) + 1:] = coeffhat[N // 2 + 1:]
        if n > N:
            coeffs[N // 2] = 0.5 * coeffhat[N // 2]
            coeffs[-(N // 2)] = 0.5 * coeffhat[N // 2]
        else:
            coeffs[N // 2] = coeffhat[N // 2]
    else:
        coeffs[:n // 2] = coeffhat[:n // 2]
        coeffs[n // 2 + 1:] = coeffhat[-(n // 2) + 1:]
        coeffs[n // 2] = 0.5 * (coeffhat[n // 2] + coeffhat[-(n // 2)])
    coeffs = n / N * np.fft.ifftshift(coeffs)
    return coeffs
