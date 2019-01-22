import logging
import numpy as np


def test_adjoint(op, tolerance=1e-10, iterations=10):
    """Numerically test validity of :meth:`adjoint` method.

    Checks if ::

        inner(y, op(x)) == inner(op.adjoint(x), y)

    in :math:`L^2` up to some tolerance for random choices of `x` and `y`.

    Parameters
    ----------
    op : :class:`~itreg.operators.LinearOperator`
        The operator.
    tolerance : float, optional
        The maximum allowed difference between the inner products. Defaults to
        1e-10.
    iterations : int, optional
        How often to repeat the test. Defaults to 10.

    Raises
    ------
    AssertionError
        If any test fails.
    """
    log = logging.getLogger(__name__)

    for i in range(iterations):
        x = op.domain.rand()
        fx = op(x)
        y = op.range.rand()
        fty = op.adjoint(y)
        err = np.abs(np.vdot(y, fx) - np.vdot(fty, x))
        log.info('err = {}'.format(err))
        assert err < tolerance
