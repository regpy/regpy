"""Forward operators."""

import logging
import numpy as np


class Operator(object):
    """Abstract base class for operators.

    Parameters
    ----------
    domx : :class:`Space <itreg.spaces.Space>`
        The domain on which the operator is defined.
    domy : :class:`Space <itreg.spaces.Space>`, optional
        The operator's codomain. Defaults to `domx`.
    log : :class:`logging.Logger`, optional
        The logger to be used. Defaults to the root logger.


    Attributes
    ----------
    domx : :class:`Space <itreg.spaces.Space>`
        The domain.
    domy : :class:`Space <itreg.spaces.Space>`
        The codomain.
    log : :class:`logging.Logger`
        The logger in use.

    """

    def __init__(self, domx, domy=None, log=logging.getLogger()):
        self.domx = domx
        self.domy = domy or domx
        self.log = log

    def __call__(self, x):
        """Evaluate the operator.

        This is an abstract method. Child classes should override it.

        The class should memorize the point as its "current point". Subsequent
        calls to :meth:`derivative` should be evaluated at this point.

        Parameters
        ----------
        x : array
            The point at which to evaluate.

        Returns
        -------
        array
            The value.

        """
        raise NotImplementedError()

    def derivative(self):
        """Compute the derivative as a :class:`LinearOperator`.

        This is an abstract method. Child classes should override it.

        The returned operator should evaluate the derivative at the "current
        point".

        For efficiency, the derivative may share data with the operator. Callers
        should *not* assume the derivative to be valid after the operator has
        been evaluated at a different point, or modified its state in any other
        way.

        Returns
        -------
        :class:`LinearOperator`
            The derivative.

        """
        raise NotImplementedError()


class LinearOperator(Operator):
    """Abstract base class for linear operators.

    This class contains some methods that are only applicable to linear
    operators.

    """

    def adjoint(self, x):
        """Evaluate the adjoint of the operator.

        This is an abstract method. Child classes should override it.

        The adjoint should be with respect to the standard :math:`L^2` inner
        product, not the inner products in :attr:`domx` and :attr:`domy`.

        Parameters
        ----------
        x : array
            The point at which to evaluate.

        Returns
        -------
        array
            The value.

        """
        raise NotImplementedError()

    def derivative(self):
        """Default derivative implementation for linear operators.

        Linear operators are their own derivative.

        Returns
        -------
        :class:`LinearOperator`
            `self`
        """
        return self

    def abs_squared(self, x):
        """Evaluate the absolute-squared of the operator.

        Here, the adjoint is taken with respect to the inner products in
        :attr:`domx` and :attr:`domy`, i.e. this method evaluates

        .. math:: G_X^{-1} T^* G_Y T.

        The method can be overridden by child classes if more efficient
        implementations are available.

        Parameters
        ----------
        x : array
            The point at which to evaluate.

        Returns
        -------
        array
            The value.

        """
        z = self(x)
        z = self.domy.gram(z)
        z = self.adjoint(z)
        z = self.domx.gram_inv(z)
        return z

    def norm(self, iterations=10):
        """Estimate the operator norm using the power method.

        The norm is taken between the spaces :attr:`domx` and :attr:`domy`.

        Parameters
        ----------
        iterations : int
            The number of power iterations. Defaults to 10.

        Returns
        -------
        float
            An estimate for the operator norm.

        """
        h = np.random.rand(*self.domx.shape)
        nrm = np.sqrt(np.sum(h**2))
        for i in range(iterations):
            h = h / nrm
            h = self.abs_squared(h)
            nrm = np.sqrt(np.sum(h**2))
        return np.sqrt(nrm)


from .volterra import Volterra  # NOQA

__all__ = [
    'LinearOperator',
    'Operator',
    'Volterra'
]
