import logging
import numpy as np

__all__ = ['test_adjoint']

log = logging.getLogger(__name__)


def test_adjoint(op, tolerance=1e-10, iterations=10):
    for i in range(iterations):
        x = np.random.rand(*op.domx.shape)
        fx = op(x)
        y = np.random.rand(*op.domy.shape)
        fty = op.adjoint(y)
        err = np.abs(np.vdot(y[:], fx[:]) - np.vdot(fty[:], x[:]))
        log.info('err = {}'.format(err))
        assert(err < tolerance)
    return True
