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

    Attributes
    ----------
    shape : tuple of int
        The array shape.
    """

    log = classlogger

    def __init__(self, shape):
        self.shape = shape

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


from .l2 import L2
