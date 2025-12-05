from copy import copy
from regpy.util import Errors

from ..general import Setting, RegSolver

__all__ = ["LinearizedSolve"]

class LinearizedSolve(RegSolver):
    r"""General method for iterated linearization of operator and subsequent solution of linearized problem. In each iteration, minimizes

    .. math::
        S_{data}(T'[x_n]h+T(x_n)) + regpar_{n} \cdot R(h+x_n)

    where :math:`T` is a Frechet-differentiable operator, using the given `inner_solver`.
    :math:`regpar_n` is a decreasing geometric sequence of regularization parameters.

    Parameters
    ----------
    setting : regpy.solvers.RegularizationSetting
        The setting of the forward problem.
    inner_solver : Solver class
        Solver class used for the solution of the linearized problem.
    inner_solver_stoprule: stoprule or callable
        Either a callable that can be applied to a setting to yield a stoprule or a Stoprule which is used for the inner solver.
    inner_solver_pars : dict, optional default: dict()
        Parameter dictionary for the inner solver. Should contain all necessary parameters which are not included in the linearized setting.
    data : array-like, optional default: None
        The measured data. If None it is taken from setting.
    regpar : float, optional default: None
        The initial regularization parameter. Must be positive. If None it is taken from setting.
    regpar_step : float, optional
        The factor by which to reduce the `regpar` in each iteration. Default: :math:`2/3`.
    init : array-like, optional default: None
        The initial guess. If None it is set to the zero array.
    """
    def __init__(
               self, setting, 
               inner_solver,
               inner_solver_stoprule,
               inner_solver_pars=dict(),
               data=None, regpar=None,
                regpar_step=2 / 3, 
                 init=None, 
         ):
        super().__init__(setting)
        if data is None:
            if(setting.data is not None):
                data=setting.data
            else:
                raise ValueError(Errors.value_error("Data has to be included in setting or given directly."))
        else:
            setting.data=data#sets data in setting if there is no data
        if(regpar is None):
            if(not setting.is_tikhonov):
                raise ValueError(Errors.value_error("Regularization parameter has to be included in setting or given directly."))
            regpar=setting.regpar
        self.data=data
        """The measured data."""
        if init is None:
            init = self.op.domain.zeros()
        self.init = init
        """The initial guess."""
        self.x = copy(self.init)
        self.y, self.deriv = self.op.linearize(self.x)
        self.regpar = regpar
        """The regularization parameter."""
        self.regpar_step = regpar_step
        """The `regpar` factor."""
        self.inner_solver=inner_solver
        self.inner_solver_pars=inner_solver_pars
        self.inner_solver_stoprule=inner_solver_stoprule

    def _next(self):
        # Linearized setting
        inner_setting=Setting(self.deriv,self.setting.penalty,self.data_fid.shift(data_shift=-self.y),regpar=self.regpar,penalty_shift=-self.x)        
        if(callable(self.inner_solver_stoprule)):
            inner_stoprule=self.inner_solver_stoprule(inner_setting)
        else:
            inner_stoprule=copy(self.inner_solver_stoprule)
        # Running inner solver
        step, _ = self.inner_solver(inner_setting,**self.inner_solver_pars).run(stoprule=inner_stoprule)
        self.x += step
        self.y , self.deriv = self.op.linearize(self.x)
        self.regpar *= self.regpar_step
        self.log.info(f"its.{self.iteration_step_nr}: alpha={self.regpar}")
    
