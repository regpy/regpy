import numpy as np

from itreg.solvers import Solver, TikhonovCG, HilbertSpaceSetting


class IrgnmCG(Solver):
    """The IRGNM_CG method.

    Solves the potentially non-linear, ill-posed equation:

       .. math:: T(x) = y,

    where   :math:`T` is a Frechet-differentiable operator. The number of
    iterations is effectively the regularization parameter and needs to be
    picked carefully.

    IRGNM stands for Iteratively Regularized Gauss Newton Method. CG stands for
    the Conjugate Gradient method. The regularized Newton equations are solved
    by the conjugate gradient method applied to the normal equation. The "outer
    iteration" and the "inner iteration" are referred to as the Newton
    iteration and the CG iteration, respectively. The CG method with all its
    iterations is run in each Newton iteration.

    Parameters
    ----------
    op : :class:`Operator <itreg.operators.Operator>`
        The forward operator.
    data : array
        The right hand side.
    init : array
        The initial guess.
    cgmaxit : int, optional
        Maximum number of CG iterations.
    cgtol : list of float, optional
        Contains three tolerances:
        The first entry controls the relative accuracy of the Newton update in
        preimage (space of "x").
        The second entry controls the relative accuracy of the Newton update in
        data space.
        The third entry controls the reduction of the residual.
    regpar0 : float, optional
    regpar_step : float, optional
        With these (regpar0, regpar_step) we compute the regulization parameter
        for the k-th Newton step by regpar0*regpar_step^k.

    Attributes
    ----------
    op : :class:`Operator <itreg.operators.Operator>`
        The forward operator.
    data : array
        The right hand side.
    init : array
        The initial guess.
    cgmaxit : int, optional
        Maximum number of CG iterations.
    regpar0 : float
    regpar_step : float
        Needed for the computation of the regulization parameter for the k-th
        Newton step.
    k : int
        Is the k-th Newton step.
    cgtol : list of float
        Contains three tolerances.
    x : array
        The current point.
    y : array
        The value at the current point.
    """

    def __init__(self, setting, data, regpar, regpar_step=2/3, init=None, cgpars={}):
        super().__init__()
        self.setting = setting
        self.data = data
        if init is None:
            init = self.setting.op.domain.zeros()
        self.init = np.asarray(init)
        self.x = np.copy(self.init)
        self.y, self.deriv = self.setting.op.linearize(self.x)
        self.regpar = regpar
        self.regpar_step = regpar_step
        self.cgpars = cgpars

    def _next(self):
        self.log.info('Running Tikhonov solver.')
        step, _ = TikhonovCG(
            setting=HilbertSpaceSetting(self.deriv, self.setting.Hdomain, self.setting.Hcodomain),
            data=self.data - self.y,
            regpar=self.regpar,
            xref=self.init - self.x,
            **self.cgpars
        ).run()
        self.x += step
        self.y, self.deriv = self.setting.op.linearize(self.x)
        self.regpar *= self.regpar_step
