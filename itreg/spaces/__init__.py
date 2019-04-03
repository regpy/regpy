"""Various vector space classes.

The main purpose of a space is to define an array shape (and thereby a
dimension) and a Gram matrix (and thereby an inner product).

Banach spaces are not covered yet.
"""

from itreg.util import classlogger

import numpy as np


class Space:
    """Abstract base class for spaces.

    Parameters
    ----------
    shape : tuple of int
        Shape of array elements of the space.
    dtype : type
        The dtype of the elements of the space. Usually `complex` or `float`.

    Attributes
    ----------
    shape : tuple of int
        The array shape.
    """

    log = classlogger

    def __init__(self, shape, coords, dtype=float):
        self.shape = shape
        self.coords=coords
        self.dtype = np.dtype(dtype)

    def gram(self, x):
        """Evaluate the Gram matrix.

        This is an abstract method. Child classes should override it.

        Parameters
        ----------
        x : array
            The point at which to evaluate.

        Returns
        -------
        array
            The value.
        """
        raise NotImplementedError

    def gram_inv(self, x):
        """Evaluate the inverse of the Gram matrix.

        This is an abstract method. Child classes should override it.

        The inverse is usually only needed in the domain of an operator, so not
        all child classes need to implement this.

        Parameters
        ----------
        x : array
            The point at which to evaluate.

        Returns
        -------
        array
            The value.
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
        return np.vdot(x, self.gram(y))

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
        return np.sqrt(np.real(self.inner(x, x)))

    def zero(self):
        """Return the zero element of the space.
        """
        return np.zeros(self.shape, self.dtype)

    def one(self):
        """Return an element of the space initalized to 1.
        """
        return np.ones(self.shape, self.dtype)

    def empty(self):
        """Return an uninitalized element of the space.
        """
        return np.empty(self.shape, self.dtype)

    def rand(self, rand=np.random.rand):
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
        r = rand(self.shape)
        if self.dtype == r.dtype:
            return r
        # Copy if dtypes don't match
        x = self.empty()
        if self.dtype.kind == 'c':
            x.real[:] = r
            x.imag[:] = rand(self.shape)
            x /= np.sqrt(2)
        else:
            x[:] = r
        


from .l2 import L2, parameters_domain_l2
from .sobolev import Sobolev, parameters_domain_sobolev
