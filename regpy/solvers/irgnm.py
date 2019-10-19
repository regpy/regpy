import numpy as np

from regpy.solvers import Solver, HilbertSpaceSetting
from regpy.solvers.tikhonov import TikhonovCG


class IrgnmCG(Solver):
    """The Iteratively Regularized Gauss-Newton Method method. In each iteration, minimizes

        ||T(x_n) + T'[x_n] h - data||**2 + regpar_n * ||x_n + h - init||**2

    where `T` is a Frechet-differentiable operator, using `regpy.solvers.tikhonov.TikhonovCG`.
    `regpar_n` is a decreasing geometric sequence of regularization parameters.

    Parameters
    ----------
    setting : regpy.solvers.HilbertSpaceSetting
        The setting of the forward problem.
    data : array-like
        The measured data.
    regpar : float
        The initial regularization parameter. Must be positive.
    regpar_step : float, optional
        The factor by which to reduce the `regpar` in each iteration. Default: `2/3`.
    init : array-like, optional
        The initial guess. Default: the zero array.
    cgpars : dict
        Parameter dictionary passed to the inner `regpy.solvers.tikhonov.TikhonovCG` solver.
    """

    def __init__(self, setting, data, regpar, regpar_step=2/3, init=None, cgpars=None):
        super().__init__()
        self.setting = setting
        """The problem setting."""
        self.data = data
        """The measured data."""
        if init is None:
            init = self.setting.op.domain.zeros()
        self.init = np.asarray(init)
        """The initial guess."""
        self.x = np.copy(self.init)
        self.y, self.deriv = self.setting.op.linearize(self.x)
        self.regpar = regpar
        """The regularizaton parameter."""
        self.regpar_step = regpar_step
        """The `regpar` factor."""
        if cgpars is None:
            cgpars = {}
        self.cgpars = cgpars
        """The additional `regpy.solvers.tikhonov.TikhonovCG` parameters."""

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
