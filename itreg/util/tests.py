import logging
import numpy as np

__all__ = ['test_adjoint']

log = logging.getLogger(__name__)


def test_adjoint(op, tolerance=1e-10, iterations=10, log=log):
    """Numerically test validity of :meth:`adjoint` method.

    Checks if ::

        inner(y, op(x)) == inner(op.adjoint(x), y)

    in :math:`L^2` up to some tolerance for random choices of `x` and `y`.

    Parameters
    ----------
    op : :class:`LinearOperator <itreg.operators.LinearOperator>`
        The operator.
    tolerance : float, optional
        The maximum allowed difference between the inner products. Defaults to
        1e-10.
    iterations : int, optional
        How often to repeat the test. Defaults to 10.
    log : :class:`logging.Logger`, optional
        The logger to which status info should be written.

    Raises
    ------
    AssertionError
        If any test fails.

    """
    for i in range(iterations):
        x = np.random.rand(*op.domx.shape)
        fx = op(x)
        y = np.random.rand(*op.domy.shape)
        fty = op.adjoint(y)
        err = np.abs(np.vdot(y[:], fx[:]) - np.vdot(fty[:], x[:]))
        log.info('err = {}'.format(err))
        assert(err < tolerance)
