from ..util import (classlogger, memoized_property, named, is_real_dtype,
                    is_complex_dtype, realdot, is_uniform)
from ..operators import Identity, LinearOperator

import numpy as np
from itertools import chain


class GenericDiscretization:
    """Abstract base class for discretizations.
    """

    log = classlogger

    def __init__(self, shape, names=None, dtype=float):
        # Upcast dtype to represent at least (single-precision) floats, no
        # bools or ints
        dtype = np.result_type(np.float32, dtype)
        # Disallow objects, strings, times or other fancy dtypes
        assert dtype.kind in 'fc'
        self.dtype = dtype
        self.shape = named(names, *shape)
        self.names = names

    def zero(self, dtype=None):
        """Return the zero element of the space.
        """
        return np.zeros(self.shape, dtype=dtype or self.dtype)

    def one(self, dtype=None):
        """Return an element of the space initalized to 1.
        """
        return np.ones(self.shape, dtype=dtype or self.dtype)

    def empty(self, dtype=None):
        """Return an uninitalized element of the space.
        """
        return np.empty(self.shape, dtype=dtype or self.dtype)

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
        r = rand(self.shape)
        if not np.can_cast(r.dtype, dtype):
            raise ValueError(
                'random generator {} can not produce values of dtype {}'.format(rand, dtype))
        if is_complex_dtype(dtype) and not is_complex_dtype(r.dtype):
            c = np.empty(self.shape, dtype=dtype)
            c.real = r
            c.imag = rand(self.shape)
            return c
        else:
            return np.asarray(r, dtype=dtype)

    @property
    def is_complex(self):
        return is_complex_dtype(self.dtype)

    @property
    def dim(self):
        if self.is_complex:
            return 2 * np.prod(self.shape)
        else:
            return np.prod(self.shape)

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def size(self):
        return np.prod(self.shape)

    @memoized_property
    def identity(self):
        return Identity(self)

    def __contains__(self, x):
        if x.shape != self.shape:
            return False
        elif is_complex_dtype(x.dtype):
            return self.is_complex()
        elif is_real_dtype(x.dtype):
            return True
        else:
            return False

    # TODO
    # def from1d(self, x):
    #     pass
    # def to1d(self, x):
    #     pass


class Grid(GenericDiscretization):
    def __init__(self, *coords, names=None, axisdata=None):
        views = []
        if axisdata and not coords:
            coords = [d.shape[0] for d in axisdata]
        for n, c in enumerate(coords):
            if isinstance(c, int):
                s = [1] * len(coords)
                s[n] = -1
                v = np.arange(c).reshape(s)
            else:
                v = np.asarray(c).view()
            v.flags.writeable = False
            views.append(v)
        # TODO Names would be nice, but providing an ndarray is more important.
        # Maybe add a "namedarray" class?
        # self.coords = named(names, *np.broadcast_arrays(*views))
        self.coords = np.broadcast_arrays(*views)
        assert self.coords[0].ndim == len(self.coords)
        # TODO ensure coords are ascending?

        super().__init__(self.coords[0].shape, names)

        axes = []
        extents = []
        for i in range(self.ndim):
            slc = [0] * self.ndim
            slc[i] = slice(None)
            axis = self.coords[i][tuple(slc)]
            axes.append(axis)
            extents.append(abs(axis[1] - axis[0]))
        # self.axes = named(names, *axes)
        self.axes = np.asarray(axes)
        # self.extents = named(names, *extents)
        self.extents = np.asarray(extents)
        self.volume_elem = np.prod(self.extents)

        if axisdata is not None:
            self.axisdata = named(names, *axisdata)
            assert len(self.axisdata) == len(self.coords)
            for i in range(len(self.axisdata)):
                assert self.shape[i] == self.axisdata[i].shape[0]


class UniformGrid(Grid):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        spacing = []
        for axis in self.axes:
            assert is_uniform(axis)
            spacing.append(axis[1] - axis[0])
        self.spacing = named(self.names, *spacing)

    # TODO
    # @memoized_property
    # def fft_dual_grid(self):

    # maybe also add fft methods directly


class Space:
    __registry = []

    def __init_subclass__(cls):
        try:
            Discr = cls.Discretization
        except AttributeError:
            return
        for base in cls.__bases__:
            try:
                assert issubclass(Discr, base.Discretization)
            except AttributeError:
                pass
        Space.__registry.append(cls)

    @classmethod
    def class_on(cls, discr):
        candidates = [space for space in Space.__registry
                      if issubclass(space, cls)
                      and isinstance(discr, space.Discretization)]
        if not candidates:
            raise RuntimeError('No space of type {} found on discretization {}'
                               .format(cls, discr))
        if len(candidates) > 1:
            # Omit all spaces that are (direct or indirect) base classes of another
            # space, i.e. retain only most specialized spaces
            bases = set(chain.from_iterable(space.mro()[1:] for space in candidates))
            candidates = [space for space in candidates if space not in bases]
        if len(candidates) > 1:
            # Retain only most specialized discretizations
            bases = set(chain.from_iterable(space.Discretization.mro()[1:] for space in candidates))
            candidates = [space for space in candidates if space.Discretization not in bases]
        if len(candidates) > 1:
            raise RuntimeError('Multiple possible spaces of type {} found on discretization {}'
                               .format(cls, discr))
        return candidates[0]

    @classmethod
    def on(cls, discr, *args, **kwargs):
        return cls.class_on(discr)(discr, *args, **kwargs)

    def __new__(cls, discr, *args, **kwargs):
        if hasattr(cls, 'Discretization'):
            return super().__new__(cls)
        else:
            c = cls.class_on(discr)
            return c.__new__(c, discr, *args, **kwargs)

    def __init__(self, discr):
        assert hasattr(self, 'Discretization')
        assert isinstance(discr, self.Discretization)
        self.discr = discr


class HilbertSpace(Space):
    @property
    def gram(self):
        """The gram matrix as a LinearOperator
        """
        raise NotImplementedError

    @property
    def gram_inv(self):
        """The inverse of the gram matrix as a LinearOperator
        """
        raise NotImplementedError

    def inner(self, x, y):
        """Compute the inner product between to elements.

        This is a convenience wrapper around :meth:`gram`.

        Parameters
        ----------
        x, y : arrays
            The elements for which the inner product should be computed.

        Returns
        -------
        float
            The inner product.
        """
        return realdot(x, self.gram(y))

    def norm(self, x):
        """Compute the norm of an element.

        This is a convenience wrapper around :meth:`norm`.

        Parameters
        ----------
        x : array
            The elements for which the norm should be computed.

        Returns
        -------
        float
            The norm.
        """
        return np.sqrt(self.inner(x, x))


class L2(HilbertSpace):
    Discretization = GenericDiscretization

    @property
    def gram(self):
        return self.discr.identity

    @property
    def gram_inv(self):
        return self.discr.identity


class HilbertPullBack(HilbertSpace):
    Discretization = GenericDiscretization

    def __init__(self, op, space):
        assert isinstance(space, HilbertSpace)
        assert isinstance(op, LinearOperator)
        assert op.range == space.discr
        super().__init__(op.domain)
        self.op = op
        self.space = space

    @memoized_property
    def gram(self):
        return self.op.adjoint * self.space.gram * self.op

    @memoized_property
    def gram_inv(self):
        # TODO This is not the inverse of gram!
        return self.op.adjoint * self.space.gram_inv * self.op
