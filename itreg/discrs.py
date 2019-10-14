from copy import copy
import numpy as np
from itertools import accumulate

from itreg import util, operators


class Discretization:
    """Discrete space R^shape or C^shape (viewed as a real space) without any
    additional structure.
    """

    log = util.classlogger

    def __init__(self, shape, dtype=float):
        # Upcast dtype to represent at least (single-precision) floats, no
        # bools or ints
        dtype = np.result_type(np.float32, dtype)
        # Allow only float and complexfloat, disallow objects, strings, times
        # or other fancy dtypes
        assert np.issubdtype(dtype, np.inexact)
        self.dtype = dtype
        try:
            self.shape = tuple(shape)
        except TypeError:
            self.shape = (shape,)

    def zeros(self, dtype=None):
        """Return the zero element of the space.
        """
        return np.zeros(self.shape, dtype=dtype or self.dtype)

    def ones(self, dtype=None):
        """Return an element of the space initalized to 1.
        """
        return np.ones(self.shape, dtype=dtype or self.dtype)

    def empty(self, dtype=None):
        """Return an uninitalized element of the space.
        """
        return np.empty(self.shape, dtype=dtype or self.dtype)

    def iter_basis(self):
        elm = self.zeros()
        for idx in np.ndindex(self.shape):
            elm[idx] = 1
            yield elm
            if self.is_complex:
                elm[idx] = 1j
                yield elm
            elm[idx] = 0

    def rand(self, rand=np.random.rand, dtype=None):
        """Return a random element of the space.

        The random generator can be passed as argument. For complex dtypes,
        real and imaginary parts are generated independently.

        Parameters
        ----------
        rand : callable
            The random function to use. Should accept the shape as integer
            parameters and return a real array of that shape. The functions in
            :mod:`numpy.random` conform to this.
        """
        dtype = dtype or self.dtype
        r = rand(*self.shape)
        if not np.can_cast(r.dtype, dtype):
            raise ValueError(
                'random generator {} can not produce values of dtype {}'.format(rand, dtype))
        if util.is_complex_dtype(dtype) and not util.is_complex_dtype(r.dtype):
            c = np.empty(self.shape, dtype=dtype)
            c.real = r
            c.imag = rand(*self.shape)
            return c
        else:
            return np.asarray(r, dtype=dtype)

    def randn(self, dtype=None):
        return self.rand(np.random.randn, dtype)

    @property
    def is_complex(self):
        return util.is_complex_dtype(self.dtype)

    @property
    def size(self):
        if self.is_complex:
            return 2 * np.prod(self.shape)
        else:
            return np.prod(self.shape)

    @property
    def ndim(self):
        return len(self.shape)

    @util.memoized_property
    def identity(self):
        return operators.Identity(self)

    def __contains__(self, x):
        if x.shape != self.shape:
            return False
        elif util.is_complex_dtype(x.dtype):
            return self.is_complex
        elif util.is_real_dtype(x.dtype):
            return True
        else:
            return False

    def flatten(self, x):
        x = np.asarray(x)
        assert self.shape == x.shape
        if self.is_complex:
            if util.is_complex_dtype(x.dtype):
                return util.complex2real(x).ravel()
            else:
                aux = self.empty()
                aux.real = x
                return util.complex2real(aux).ravel()
        elif util.is_complex_dtype(x.dtype):
            raise TypeError('Real discretization can not handle complex vectors')
        return x.ravel()

    def fromflat(self, x):
        x = np.asarray(x)
        assert util.is_real_dtype(x.dtype)
        if self.is_complex:
            return util.real2complex(x.reshape(self.shape + (2,)))
        else:
            return x.reshape(self.shape)

    def complex_space(self):
        other = copy(self)
        other.dtype = np.result_type(1j, self.dtype)
        return other

    def real_space(self):
        other = copy(self)
        other.dtype = np.empty(0, dtype=self.dtype).real.dtype
        return other

    def __eq__(self, other):
        # Only handle the base class to avoid accidental equality of subclass
        # instances.
        if type(self) == type(other) == Discretization:
            return (
                self.shape == other.shape and
                self.dtype == other.dtype
            )
        else:
            return NotImplemented

    def __add__(self, other):
        if isinstance(other, Discretization):
            return DirectSum(self, other, flatten=True)
        else:
            return NotImplemented

    def __radd__(self, other):
        if isinstance(other, Discretization):
            return DirectSum(other, self, flatten=True)
        else:
            return NotImplemented


class Grid(Discretization):
    def __init__(self, *coords, axisdata=None, dtype=float):
        views = []
        if axisdata and not coords:
            coords = [d.shape[0] for d in axisdata]
        for n, c in enumerate(coords):
            if isinstance(c, int):
                v = np.arange(c)
            elif isinstance(c, tuple):
                v = np.linspace(*c)
            else:
                v = np.asarray(c).view()
            if 1 == v.ndim < len(coords):
                s = [1] * len(coords)
                s[n] = -1
                v = v.reshape(s)
            # TODO is this really necessary given that we probably perform a
            # copy using asarray anyway?
            v.flags.writeable = False
            views.append(v)
        self.coords = np.asarray(np.broadcast_arrays(*views))
        assert self.coords[0].ndim == len(self.coords)
        # TODO ensure coords are ascending?

        super().__init__(self.coords[0].shape, dtype)

        axes = []
        extents = []
        for i in range(self.ndim):
            slc = [0] * self.ndim
            slc[i] = slice(None)
            axis = self.coords[i][tuple(slc)]
            axes.append(axis)
            extents.append(abs(axis[-1] - axis[0]))
        self.axes = np.asarray(axes)
        self.extents = np.asarray(extents)

        if axisdata is not None:
            self.axisdata = tuple(axisdata)
            assert len(self.axisdata) == len(self.coords)
            for i in range(len(self.axisdata)):
                assert self.shape[i] == self.axisdata[i].shape[0]


class UniformGrid(Grid):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        spacing = []
        for axis in self.axes:
            assert util.is_uniform(axis)
            spacing.append(axis[1] - axis[0])
        self.spacing = np.asarray(spacing)
        self.volume_elem = np.prod(self.spacing)

    def frequencies(self, centered=True, axes=None):
        if axes is None:
            axes = range(self.ndim)
        axes = set(axes)
        frqs = []
        for i, (s, l) in enumerate(zip(self.shape, self.spacing)):
            if i in axes:
                # Use (spacing * shape) in denominator instead of extents, since the grid is assumed to be periodic.
                if centered:
                    frqs.append(np.arange(-(s//2), (s+1)//2) / (s*l))
                else:
                    frqs.append(np.concatenate((np.arange(0, (s+1)//2), np.arange(-(s//2), 0))) / (s*l))
            else:
                frqs.append(self.axes[i])
        return np.asarray(np.broadcast_arrays(*np.ix_(*frqs)))


class DirectSum(Discretization):
    """The direct sum of an arbirtary number of discretizations.

    Elements of the direct sum will always be 1d real arrays.

    Note that constructing DirectSum instances can be done more comfortably
    simply by adding :class:`~itreg.spaces.discrs.Discretization` instances.
    However, for generic code, when it's not known whether the summands are
    themselves direct sums, it's better to avoid the `+` overload due the
    `flatten` parameter (see below), since otherwise the number of summands is
    not fixed.

    Parameters
    ----------
    *summands : variable number of :class:`~itreg.spaces.discrs.Discretization`
        The discretizations to be summed.
    flatten : bool, optional
        Whether summands that are themselves DirectSums should be merged into
        this instance. If False, DirectSum is not associative, but the join and
        split methods behave more predictably. Default: False, but will be set
        to True when constructing the DirectSum via Discretization.__add__,
        i.e. when using the `+` operator, in order to make repeated sums like
        `A + B + C` unambiguous.
    """

    def __init__(self, *summands, flatten=False):
        assert all(isinstance(s, Discretization) for s in summands)
        self.summands = []
        for s in summands:
            if flatten and isinstance(s, type(self)):
                self.summands.extend(s.summands)
            else:
                self.summands.append(s)
        self.idxs = [0] + list(accumulate(s.size for s in self.summands))
        super().__init__(self.idxs[-1])

    def __eq__(self, other):
        return (
            isinstance(other, type(self)) and
            len(self.summands) == len(other.summands) and
            all(s == t for s, t in zip(self.summands, other.summands))
        )

    def join(self, *xs):
        """Transform a collection of elements of the summands to an element of
        the direct sum.

        Parameters
        ----------
        *xs : variable number of arrays
            The elements of the summands. The number should match the number of
            summands, and for all i, xs[i] should be an element of
            self.summands[i].

        Returns
        -------
        1d real array representing and element of the direct sum.
        """
        assert all(x in s for s, x in zip(self.summands, xs))
        elm = self.empty()
        for s, x, start, end in zip(self.summands, xs, self.idxs, self.idxs[1:]):
            elm[start:end] = s.flatten(x)
        return elm

    def split(self, x):
        """Split an element of the direct sum into a tuple of elements of the
        summands.

        Parameters
        ----------
        x : array
            An array representing an element of the direct sum.

        Returns
        -------
        tuple of discretizations
            The components of x for the summands.
        """
        assert x in self
        return tuple(
            s.fromflat(x[start:end])
            for s, start, end in zip(self.summands, self.idxs, self.idxs[1:])
        )

    def __getitem__(self, item):
        return self.summands[item]

    def __iter__(self):
        return iter(self.summands)

    def __len__(self):
        return len(self.summands)
