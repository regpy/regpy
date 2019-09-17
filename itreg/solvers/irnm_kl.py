"""IRNM_KL Solver """

from . import Solver
from itreg.solvers.sqp import SQP


class IRNM_KL(Solver):

    """The iteratively regularized Newton method with shifted Kullback-Leibler
    divergence

    Solves the potentially non-linear, ill-posed equation:

       .. math:: T(x) = y,

    where :math:`T` is a Frechet-differentiable operator.

    The penalty is determined by self.op.domain.gram(X).

    Parameters
    ----------
    op : :class:`Operator <itreg.operators.Operator>`
        The forward operator.
    data : array
        The right hand side.
    init : array
        The initial guess.
    alpha_step : float, optional
        Decreasing step for reg. parameter. Standard value: 2/3
    alpha0 : float, optional
        Starting reg. parameter for IRNM. Standard value: 5e-6
    intensity : float, optional
        Intensity of the operator. Standard value: 1

    Attributes
    ----------
    op : :class:`Operator <itreg.operators.Operator>`
        The forward operator.
    data : array
        The right hand side.
    x : array
        The current point.
    y : array
        The value at the current point.
    alpha_step : float
        Decreasing step for reg. parameter. Standard value: 2/3
    alpha0 : float, optional
        Starting reg. parameter for IRNM. Standard value: 5e-6
    intensity : float
        Intensity of the operator. Standard value: 1
    k : int
        Number of iterations.
    """

    def __init__(self, op, data, init, continuum, alpha0=5e-6, alpha_step=2/3.,
                 intensity=1):
        """Initialize parameters """

        super().__init__()
        self.op = op
        self.data = data
        self.init = init
        self.x = self.init
        self.y = self.op(self.x)

        # Parameters for the outer iteration (Newton method)
        self.k = 0
        self.alpha_step = alpha_step
        self.intensity = intensity
        self.alpha = alpha0 * self.intensity

    def next(self):
        """Run a single IRNM_KL iteration.

        Returns
        -------
        bool
            Always True, as the IRNM_KL method never stops on its own.

        """
        self._sqp = SQP(self.op, self.data, self.init, self.x, self.y,
                        self.alpha, self.k, self.x, self.intensity)
        self.x = self._sqp.run()
        self.k += 1
        self.y = self.op(self.x)
        # prepare next step
        self.alpha *= self.alpha_step

        return True
