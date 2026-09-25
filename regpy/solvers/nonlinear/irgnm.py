from __future__ import annotations

from typing import TYPE_CHECKING
from math import sqrt,isqrt
from numbers import Real
from typing import Iterable, Iterator

import numpy as np

from regpy.stoprules import CountIterations
from regpy.util import Errors
from regpy.operators import ObsDataT, Operator
from regpy.vecsps import ReadOnlyVector, Vector
from .. import RegSolver, Setting, Naturals, check_Hilbert_applicability
from ..linear import LavrentievCG, TikhonovCG

__all__ = ["IrgnmCG","LevenbergMarquardt","IrgnmCGPrec"]


def _regularization_parameter_iterator(
    regpars: Iterable[float] | None,
    regpar: float,
    regpar_step: float,
) -> Iterator[float]:
    r"""Create the regularization-parameter iterator.
    
    ``regpar`` and ``regpar_step`` retain their historical meaning when no
    explicit iterator is supplied.
    
    Parameters
    ----------
    regpars: Iterable[float] | None
        Regpars.
    regpar: float
        Regularization parameter.
    regpar_step: float
        Regpar step.
    
    
    Returns
    -------
    Iterator[float]
        Result described above.
    """
    if regpars is None:
        return iter((regpar * (regpar_step) ** Naturals()))
    try:
        return iter(regpars)
    except TypeError as error:
        raise TypeError(Errors.value_error(
            "regpars must be an iterable of positive real numbers.",
            regpars,
        )) from error


def _next_regularization_parameter(
    regpar_iterator: Iterator[float],
) -> float:
    r"""Read and validate one regularization parameter.
    
    Parameters
    ----------
    regpar_iterator: Iterator[float]
        Regpar iterator.
    
    
    Returns
    -------
    float
        Computed scalar value.
    """
    value = next(regpar_iterator)
    if not isinstance(value, Real) or value <= 0:
        raise ValueError(Errors.value_error(
            "All regularization parameters must be positive real numbers.",
            value,
        ))
    return float(value)


def _inner_cg_parameters(
    cg_pars: dict | None,
    codomain_free: bool,
) -> dict:
    r"""Return independent, compatible parameters for an inner CG solve.
    
    Parameters
    ----------
    cg_pars: dict | None
        Options for cg pars.
    codomain_free: bool
        Codomain free.
    
    
    Returns
    -------
    dict
        Result described above.
    """
    if cg_pars is None:
        pars = {'reltolx': 1 / 3, 'all_tol_criteria': False}
        if not codomain_free:
            pars['reltoly'] = 1 / 3
        return pars
    pars = dict(cg_pars)
    obsolete = {'regpar', 'update_setting'} & pars.keys()
    if obsolete:
        names = ', '.join(sorted(obsolete))
        raise ValueError(Errors.value_error(
            f"cg_pars contains obsolete TikhonovCG parameters: {names}."
        ))
    if codomain_free:
        if pars.get('reltoly') is not None:
            raise ValueError(Errors.value_error(
                "reltoly is unavailable for codomain-free inner CG solves."
            ))
        pars.pop('reltoly', None)
    return pars


def _run_inner_cg(
    solver: RegSolver,
    reference: ReadOnlyVector,
    stoprule: CountIterations,
) -> tuple[Vector, Vector | None]:
    r"""Solve one linearized Tikhonov problem.
    
    Parameters
    ----------
    solver: RegSolver
        Solver.
    reference: ReadOnlyVector
        Input vector; its coefficients are only read.
    stoprule: CountIterations
        Stoprule.
    
    
    Returns
    -------
    tuple[Vector, Vector | None]
        The first vector is the inimizers of the Tikhonov functional. In the codomain_free case
        the second vector is None, otherwise it is the operator image of the first. 
    """
    cg_pars = dict(solver.cg_pars)
    cg_pars.setdefault(
        'logging_level', solver.inner_it_logging_level
    )
    if solver.setting.codomain_free:
        solver.deriv.adjoint_data(
            solver.data, out=solver._adjoint_data
        )
        solver.op.domain.copyto(
            solver._normal_rhs, solver._adjoint_data
        )
        solver._normal_rhs -= solver._adjoint_eval
        inner_setting = Setting(
            solver.deriv.adjoint_eval,
            solver.h_domain,
            solver.h_domain,
            regpar=solver.regpar,
            init=reference,
            codomain_free=True,
            explicitDataWarning=False,
            logging_level=solver.inner_it_logging_level,
        )
        return LavrentievCG(
            setting=inner_setting,
            data=solver._normal_rhs,
            x0=solver.deriv.domain.zeros(),
            **cg_pars,
        ).run(stoprule=stoprule)

    inner_setting = Setting(
        solver.deriv,
        solver.h_domain,
        solver.h_codomain,
        regpar=solver.regpar,
        init=reference,        
        explicitDataWarning=False,
        logging_level=solver.inner_it_logging_level,
    )
    return TikhonovCG(
        setting=inner_setting,
        data=solver.data - solver.y,
        x0=solver.deriv.domain.zeros(),
        **cg_pars,
    ).run(stoprule=stoprule)

class IrgnmCG(RegSolver):
    r"""The Iteratively Regularized Gauss-Newton Method method. In each iteration, minimizes
    
    .. math::
        \Vert T(x_{n}) + T'[x_n] h - data\Vert^{2}
        + regpar_{n} \cdot \Vert x_{n} + h - init\Vert^{2}
    
    where :math:`T` is a Fréchet-differentiable operator, using `regpy.solvers.linear.tikhonov.TikhonovCG`.
    By default, :math:`regpar_n` is a decreasing geometric sequence of
    regularization parameters. An arbitrary iterable can be supplied instead.
    
    Parameters
    ----------
    setting: Setting
        The setting of the forward problem.
    data: Vector | ObsDataT | None [default: None]
        The measured data. If it is None it is taken from the setting. For a
        codomain-free setting it may be any object accepted by
        :meth:`Operator.adjoint_data`.
    regpar: float [default: 1.0]
        The initial regularization parameter. Must be positive.
    regpar_step: float [default: 2 / 3]
        The factor by which to reduce ``regpar`` in each iteration when
        ``regpars`` is not supplied.
    init: ReadOnlyVector | None [default: None]
        Initial guess. In the default case, it is taken from the setting.
    cg_pars: dict | None [default: None]
        Parameters for the inner
        :class:`~regpy.solvers.linear.tikhonov.TikhonovCG` or, in the
        codomain-free case,
        :class:`~regpy.solvers.linear.tikhonov.Lavrentiev` solver. ``regpar``
        and ``update_setting`` are not accepted.
    cgstop: int | None [default: 1000]
        Maximum number of inner CG iterations.
    inner_it_logging_level: int | str [default: 'WARNING']
        Logging level for inner iterations.
    simplified_op: Operator | None [default: None]
        An operator the with the same mapping properties as setting.op, which is cheaper to evaluate. 
        It is used for the derivative in the Newton equation. This option is
        unavailable for codomain-free settings.
    logging_level: int | str [default: 'INFO']
        Logging level.
    regpars: Iterable[float] | None [default: None]
        Regularization parameters for the outer iterations. If omitted, a
        sequence ``regpar * regpar_step**Naturals()`` is used. If supplied, ``regpar`` and ``regpar_step``
        are ignored. Iteration ends after the last supplied parameter.
    
    Notes
    -----
    In the codomain-free case the normal-equation right-hand side is formed
    as ``deriv.adjoint_data(data) - adjoint_eval`` and the inner problem is
    delegated to :class:`~regpy.solvers.linear.tikhonov.Lavrentiev`.
    """

    def __init__(
                self, setting:Setting,
                data:Vector | ObsDataT | None=None,
                regpar:float=1., regpar_step:float=2 / 3,
                init:ReadOnlyVector | None=None,
                cg_pars:dict | None=None,
                cgstop:int | None=1000,
                inner_it_logging_level:int | str = "WARNING",
                simplified_op:Operator | None = None,
                logging_level:int | str = "INFO",
                regpars:Iterable[float] | None = None,
         ) -> None:
        out, par = self.check_applicability(
            setting, user_data=data, user_init=init
        )
        if not out['applicable']:
            raise ValueError('IrgnmCG not applicable to this setting. ' + out['info'])
        super().__init__(setting)
        self.log.setLevel(logging_level)
        self.init = par['init']
        """The initial guess."""
        self.data = par['data']
        """The measured data."""                
        if self.data is None:
            raise ValueError(Errors.value_error(
                "Data have to be included in the setting or given directly."
            ))
        if setting.codomain_free and simplified_op is not None:
            raise ValueError(Errors.value_error(
                "simplified_op is unavailable for codomain-free IrgnmCG."
            ))
        self.x = self.op.domain.empty()
        self.op.domain.copyto(self.x, self.init)
        if setting.codomain_free:
            self.y = None
            self._adjoint_eval, self.deriv = self.op.linearize(
                self.x, return_adjoint_eval=True
            )
            self._adjoint_data = self.op.domain.empty()
            self._normal_rhs = self.op.domain.empty()
        elif simplified_op:
            self.simplified_op = simplified_op
            _, self.deriv = self.simplified_op.linearize(self.x)
            self.y = self.op(self.x)
        else:
            self.y, self.deriv = self.op.linearize(self.x)
        self._regpar_iterator = _regularization_parameter_iterator(
            regpars, regpar, regpar_step
        )
        try:
            self.regpar = _next_regularization_parameter(
                self._regpar_iterator
            )
        except StopIteration as error:
            raise ValueError(Errors.value_error(
                "regpars must provide at least one regularization parameter.",
                regpars,
            )) from error
        """The regularization parameter."""
        self.regpar_step = regpar_step
        """The geometric factor used when ``regpars`` is not supplied."""
        self.cg_pars = _inner_cg_parameters(
            cg_pars, setting.codomain_free
        )
        """The additional `regpy.solvers.linear.tikhonov.TikhonovCG` parameters."""
        self.cgstop = cgstop
        """Maximum number of iterations for inner CG solver, or None"""
        self.inner_it_logging_level = inner_it_logging_level
        self._nr_inner_steps = 0

    def _next(self) -> None:
        if self.cgstop is not None:
            stoprule = CountIterations(self.cgstop)
        else:
            stoprule = CountIterations(2**15)
        # Disable info logging, but don't override log level for all CountIterations instances.
        #stoprule.log = self.log.getChild('CountIterations')
        stoprule.log.setLevel(self.inner_it_logging_level)
        inner_init = self.op.domain.empty()
        self.op.domain.copyto(inner_init, self.init)
        inner_init -= self.x
        step, _ = _run_inner_cg(self, inner_init, stoprule)
        self.x += step
        if self.setting.codomain_free:
            self._adjoint_eval, self.deriv = self.op.linearize(
                self.x,
                out=self._adjoint_eval,
                return_adjoint_eval=True,
            )
            self.y = None
        elif hasattr(self,'simplified_op'):
            _, self.deriv = self.simplified_op.linearize(self.x)
            self.y = self.op(self.x)
        else:
            self.y , self.deriv = self.op.linearize(self.x)
        self._nr_inner_steps = stoprule.iteration
        self.log.info('its.{}: alpha={}, CG its:{}'.format(self.iteration_step_nr,self.regpar,self._nr_inner_steps))
        try:
            self.regpar = _next_regularization_parameter(
                self._regpar_iterator
            )
        except StopIteration:
            self.converge()
    
    def nr_inner_its(self) -> int:
        r"""Nr inner its.
        
        
        Returns
        -------
        int
            Computed scalar value.
        """
        return self._nr_inner_steps

    @staticmethod
    def check_applicability(
        setting: Setting,
        op_norm: float | None = None,
        user_data: Vector | ObsDataT | None = None,
        user_init: ReadOnlyVector | None = None,
    ) -> tuple[dict, dict]:
        r"""Check whether IRGNM-CG can be used for ``setting``.
        
        Parameters
        ----------
        setting: Setting
            Problem setting containing the operator, penalty, and data fidelity.
        op_norm: float | None [default: None]
            Operator-norm estimate; None requests numerical estimation.
        user_data: Vector | ObsDataT | None [default: None]
            Writable vector used by this operation.
        user_init: ReadOnlyVector | None [default: None]
            Input vector; its coefficients are only read.
        
        
        Returns
        -------
        tuple[dict, dict]
             The first dictionary contains applicability information, the second is empty here. 
        """
        out, par = check_Hilbert_applicability(
            setting,
            op_norm=op_norm,
            user_data=user_data,
            user_init=user_init,
        )
        out['applicable'] = out['info'] == ''
        return out, par
        

class LevenbergMarquardt(RegSolver):
    r"""The Levenberg-Marquardt method. In each iteration, minimizes
    
    .. math::
        \Vert T(x_{n}) + T'[x_n] h - data\Vert^{2}
        + regpar_{n} \cdot \Vert h\Vert^{2}
    
    where :math:`T` is a Fréchet-differentiable operator, using `regpy.solvers.linear.tikhonov.TikhonovCG`.
    By default, :math:`regpar_n` is a decreasing geometric sequence of
    regularization parameters. An arbitrary iterable can be supplied instead.
    
    Parameters
    ----------
    setting: Setting
        The setting of the forward problem.
    data: Vector | ObsDataT | None [default: None]
        The measured data. If it is None it is taken from the setting. For a
        codomain-free setting it may be any object accepted by
        :meth:`Operator.adjoint_data`.
    regpar: float [default: 1.0]
        The initial regularization parameter. Must be positive.
    regpar_step: float [default: 2 / 3]
        The factor by which to reduce ``regpar`` in each iteration when
        ``regpars`` is not supplied.
    init: ReadOnlyVector | None [default: None]
        Initial guess. In the default case, it is taken from the setting.
    cg_pars: dict | None [default: None]
        Parameters for the inner
        :class:`~regpy.solvers.linear.tikhonov.TikhonovCG` or, in the
        codomain-free case,
        :class:`~regpy.solvers.linear.tikhonov.Lavrentiev` solver. ``regpar``
        and ``update_setting`` are not accepted.
    cgstop: int | None [default: 1000]
        Maximum number of inner CG iterations.
    inner_it_logging_level: int | str [default: 'WARNING']
        Logging level for inner iterations.
    simplified_op: Operator | None [default: None]
        An operator the with the same mapping properties as setting.op, which is cheaper to evaluate. 
        It is used for the derivative in the Newton equation. This option is
        unavailable for codomain-free settings.
    logging_level: int | str [default: 'INFO']
        Logging level.
    regpars: Iterable[float] | None [default: None]
        Regularization parameters for the outer iterations. If omitted, a
        sequence ``regpar * regpar_step**Naturals()`` is used. If supplied, ``regpar`` and ``regpar_step``
        are ignored. Iteration ends after the last supplied parameter.
    
    Notes
    -----
    In the codomain-free case the normal-equation right-hand side is formed
    as ``deriv.adjoint_data(data) - adjoint_eval`` and the inner problem is
    delegated to :class:`~regpy.solvers.linear.tikhonov.Lavrentiev`.
    """

    def __init__(
               self, setting:Setting,
                data:Vector | ObsDataT | None=None,
                regpar:float=1., regpar_step:float=2 / 3,
                init:ReadOnlyVector | None=None,
                cg_pars:dict | None=None,
                cgstop:int | None=1000,
                inner_it_logging_level:int | str = "WARNING",
                simplified_op:Operator|None = None,
                logging_level:int | str = "INFO",
                regpars:Iterable[float] | None = None,
         ) -> None:
        out, par = self.check_applicability(
            setting, user_data=data, user_init=init
        )
        if not out['applicable']:
            raise ValueError(
                'LevenbergMarquardt not applicable to this setting. '
                + out['info']
            )
        super().__init__(setting)
        self.log.setLevel(logging_level)
        self.init = par['init']
        """The initial guess."""
        self.data = par['data']
        """The measured data."""  
        if self.data is None:
            raise ValueError(Errors.value_error(
                "Data have to be included in the setting or given directly."
            ))
        if setting.codomain_free and simplified_op is not None:
            raise ValueError(Errors.value_error(
                "simplified_op is unavailable for codomain-free "
                "LevenbergMarquardt."
            ))
        self.x = self.op.domain.empty()
        self.op.domain.copyto(self.x, self.init)
        if setting.codomain_free:
            self.y = None
            self._adjoint_eval, self.deriv = self.op.linearize(
                self.x, return_adjoint_eval=True
            )
            self._adjoint_data = self.op.domain.empty()
            self._normal_rhs = self.op.domain.empty()
        elif simplified_op:
            self.simplified_op = simplified_op
            _, self.deriv = self.simplified_op.linearize(self.x)
            self.y = self.op(self.x)
        else:
            self.y, self.deriv = self.op.linearize(self.x)
        self._regpar_iterator = _regularization_parameter_iterator(
            regpars, regpar, regpar_step
        )
        try:
            self.regpar = _next_regularization_parameter(
                self._regpar_iterator
            )
        except StopIteration as error:
            raise ValueError(Errors.value_error(
                "regpars must provide at least one regularization parameter.",
                regpars,
            )) from error
        """The regularization parameter."""
        self.regpar_step = regpar_step
        """The geometric factor used when ``regpars`` is not supplied."""
        self.cg_pars = _inner_cg_parameters(
            cg_pars, setting.codomain_free
        )
        """The additional `regpy.solvers.linear.tikhonov.TikhonovCG` parameters."""
        self.cgstop = cgstop
        """Maximum number of iterations for inner CG solver, or None"""
        self.inner_it_logging_level = inner_it_logging_level
        self._nr_inner_steps = 0

    def _next(self) -> None:
        if self.cgstop is not None:
            stoprule = CountIterations(self.cgstop)
        else:
            stoprule = CountIterations(2**15)
        # Disable info logging, but don't override log level for all CountIterations instances.
        stoprule.log = self.log.getChild('CountIterations')
        stoprule.log.setLevel("WARNING")
        inner_init = self.op.domain.zeros(read_only=True)
        step, _ = _run_inner_cg(self, inner_init, stoprule)
        self.x += step
        if self.setting.codomain_free:
            self._adjoint_eval, self.deriv = self.op.linearize(
                self.x,
                out=self._adjoint_eval,
                return_adjoint_eval=True,
            )
            self.y = None
        elif hasattr(self,'simplified_op'):
            _, self.deriv = self.simplified_op.linearize(self.x)
            self.y = self.op(self.x)
        else:
            self.y , self.deriv = self.op.linearize(self.x)
        self._nr_inner_steps = stoprule.iteration
        self.log.info('its.{}: alpha={}, CG its:{}'.format(self.iteration_step_nr,self.regpar,self._nr_inner_steps))
        try:
            self.regpar = _next_regularization_parameter(
                self._regpar_iterator
            )
        except StopIteration:
            self.converge()
    
    def nr_inner_its(self) -> int:
        r"""Nr inner its.
        
        
        Returns
        -------
        int
            Computed scalar value.
        """
        return self._nr_inner_steps

    check_applicability = staticmethod(IrgnmCG.check_applicability)


from regpy.operators import EinSum,PtwMultiplication
from regpy import util
from scipy.sparse.linalg import eigsh
from regpy.vecsps import UniformGridFcts
        
class IrgnmCGPrec(RegSolver):
    r"""The Iteratively Regularized Gauss-Newton Method method. In each iteration, minimizes
    
    .. math::
        \Vert F(x_n) + F'[x_n] h - data\Vert^2 + \text{regpar}_n  \Vert x_n + h - init\Vert^2
    
    where :math:`F` is a Fréchet-differentiable operator, by solving in every iteration step the problem
    
    .. math::
        \underset{Mh = g}{\mathrm{minimize}}    \Vert T (M  g) - rhs\Vert^2 + \text{regpar} \Vert M  (g - x_{ref})\Vert^2
    
    with `regpy.solvers.linear.tikhonov.TikhonovCG' and spectral preconditioner :math:`M`.
    By default, :math:`regpar_n` is a decreasing geometric sequence of
    regularization parameters. An arbitrary iterable can be supplied instead.
    The spectral preconditioner :math:`M` is chosen, such that:
    
    .. math::
        M  A  M \approx Id
    
    where :math:`A = (T^t Gram_{codomain} T + \text{regpar} Id) = T^* T + \text{regpar} Id` 
    
    Note that the Tikhonov CG solver computes an orthonormal basis of vectors spanning the Krylov subspace of 
    the order of the number of iterations: :math:`\{v_j\}`
    We approximate A by the operator:
    
    .. math::
        C_k: v \mapsto \text{regpar} v +\sum_{j=1}^k \langle v, v_j\rangle \lambda_j v_j
    
    where lambda are the biggest eigenvalues of :math:`T*T`.
    
    We choose: :math:`M = C_k^{-1/2} and M^{-1} = C_k^{1/2}`
    
    It is:
    
    .. math::
        M     &: v \mapsto \frac{1}{\sqrt{\text{regpar}}} v + \sum_{j=1}^{k} \left[\frac{1}{\sqrt{\lambda_j+\text{regpar}}}-\frac{1}{\sqrt{\text{regpar}}}\right] \langle v_j, v\rangle v_j \\
        M^{-1}&: v \mapsto \sqrt{\text{regpar}} v + \sum_{j=1}^{k} \left[\sqrt{\lambda_j+\text{regpar}} -\sqrt{\text{regpar}}\right] \langle v_j, v\rangle v_j.
    
    This only works for UniformGridFcts domains.
    
    Parameters
    ----------
    setting: Setting
        The setting of the forward problem. The domain of the operator has to be of type UniformGridFcts.
    data: ReadOnlyVector | None [default: None]
        The measured data. If it is None it is taken from the setting.
    regpar: float [default: 1.0]
        The initial regularization parameter. Must be positive.
    regpar_step: float [default: 2 / 3]
        The factor by which to reduce ``regpar`` in each iteration when
        ``regpars`` is not supplied.
    init: ReadOnlyVector | None [default: None]
        The initial guess.
    cg_pars: dict | None [default: None]
        Parameter dictionary passed to the inner
        :class:`regpy.solvers.linear.tikhonov.TikhonovCG` solver.
    cgstop: int | None [default: None]
        Maximum number of inner CG iterations.
    precpars: dict | None [default: None]
        Parameters for the spectral preconditioner. The defaults are Krylov
        order 6 and four eigenvalues.
    logging_level: int | str [default: 'INFO']
        Logging level.
    regpars: Iterable[float] | None [default: None]
        Regularization parameters for the outer iterations. If omitted, a
        sequence ``regpar * regpar_step**Naturals()`` is used. If supplied, ``regpar`` and ``regpar_step``
        are ignored. Iteration ends after the last supplied parameter.
    """

    def __init__(
        self, setting:Setting, data:ReadOnlyVector | None=None, regpar:float=1.,
        regpar_step:float=2 / 3, init:ReadOnlyVector | None=None,
        cg_pars:dict | None=None, cgstop:int | None=None,
        precpars:dict | None=None,
        logging_level:int | str="INFO",
        regpars:Iterable[float] | None = None,
        ) -> None:
        out, par = self.check_applicability(
            setting, user_data=data, user_init=init
        )
        if not out['applicable']:
            raise ValueError(
                'IrgnmCGPrec not applicable to this setting. ' + out['info']
            )
        if(not isinstance(setting.op.domain,UniformGridFcts)):
            raise ValueError(f"Computation of preconditioner requires UniformGridFcts, but got domain of type {type(setting.op.domain)}.")
        super().__init__(setting)
        self.log.setLevel(logging_level)
        self.init = par['init']
        """The initial guess."""
        self.data = par['data']
        """The measured data."""  
        if self.data is None:
            raise ValueError(Errors.value_error(
                "Data have to be included in the setting or given directly."
            ))
        self.x = self.op.domain.empty()
        self.op.domain.copyto(self.x, self.init)
        self.y, self.deriv = self.op.linearize(self.x)
        self._regpar_iterator = _regularization_parameter_iterator(
            regpars, regpar, regpar_step
        )
        try:
            self.regpar = _next_regularization_parameter(
                self._regpar_iterator
            )
        except StopIteration as error:
            raise ValueError(Errors.value_error(
                "regpars must provide at least one regularization parameter.",
                regpars,
            )) from error
        """The regularization parameter."""
        self.regpar_step = regpar_step
        """The geometric factor used when ``regpars`` is not supplied."""
        if cg_pars is None:
            self.cg_pars = {}
        else:
            self.cg_pars = _inner_cg_parameters(
                cg_pars, codomain_free=False
            )
        self.cgstop = cgstop
        """The additional `regpy.solvers.linear.tikhonov.TikhonovCG` parameters."""
        
        self.k=0
        """Counts the number of iterations"""

        if precpars is None:
            self.krylov_order = 6
            """Order of krylov space in which the spectral preconditioner is computed"""
            self.number_eigenvalues = 4
            """Spectral preconditioner computed only from the biggest eigenvalues """
        else: 
            self.krylov_order = precpars['krylov_order']
            self.number_eigenvalues = precpars['number_eigenvalues']

        self.krylov_basis = np.zeros((self.krylov_order, *self.h_domain.vecsp.shape),dtype=self.op.domain.dtype)
        """Orthonormal Basis of Krylov subspace"""
        self.krylov_basis_img = np.zeros((self.krylov_order, *self.h_codomain.vecsp.shape),dtype=self.op.codomain.dtype)
        """Image of the Krylov Basis under the derivative of the operator"""
        self.krylov_basis_img_2 = np.zeros((self.krylov_order, *self.h_codomain.vecsp.shape),dtype=self.op.codomain.dtype)
        """Gram matrix applied to image of the Krylov Basis"""
        self.need_prec_update = True
        """Is an update of the preconditioner needed"""

    @staticmethod
    def check_applicability(
        setting: Setting,
        op_norm: float | None = None,
        user_data: ReadOnlyVector | None = None,
        user_init: ReadOnlyVector | None = None,
    ) -> tuple[dict, dict]:
        r"""Check whether preconditioned IRGNM-CG-Prec can be used.
        
        Parameters
        ----------
        setting: Setting
            Problem setting containing the operator, penalty, and data fidelity.
        op_norm: float | None [default: None]
            Operator-norm estimate; None requests numerical estimation.
        user_data: ReadOnlyVector | None [default: None]
            Input vector; its coefficients are only read.
        user_init: ReadOnlyVector | None [default: None]
            Input vector; its coefficients are only read.
        
        
        Returns
        -------
        tuple[dict, dict]
            The first dictionary contains applicability information, the second is empty here. 
        """
        out, par = IrgnmCG.check_applicability(
            setting,
            op_norm=op_norm,
            user_data=user_data,
            user_init=user_init,
        )
        if not isinstance(setting.op.domain, UniformGridFcts):
            out['info'] += 'Operator domain is not UniformGridFcts. '
        if setting.codomain_free:
            out['info'] += (
                'IrgnmCGPrec requires codomain vectors for its spectral '
                'preconditioner. '
            )
        out['applicable'] = out['info'] == ''
        return out, par
    
    def _next(self) -> None:
        if self.cgstop is not None:
            stoprule = CountIterations(self.cgstop)
            # Disable info logging, but don't override log level for all
            # CountIterations instances.
        else:
            stoprule = CountIterations(2**15)
        stoprule.log = self.log.getChild('CountIterations')
        stoprule.log.setLevel("WARNING")
        self.log.info('Running Tikhonov solver.')
        
        if self.need_prec_update:
            self.log.info('Spectral Preconditioner needs to be updated')
            step, _ = TikhonovCG(
                setting=Setting(
                    self.deriv,
                    self.h_domain,
                    self.h_codomain,
                    regpar=self.regpar,
                    explicitDataWarning=False,
                ),
                data=self.data - self.y,
                krylov_basis=self.krylov_basis,
                init=self.init - self.x,
                **self.cg_pars
            ).run(stoprule=stoprule)
            for i in range(0, self.krylov_order):
                self.krylov_basis_img[i, :] = self.deriv(self.krylov_basis[i, :])
                self.krylov_basis_img_2[i, :] = self.h_codomain.gram(self.krylov_basis_img[i, :])
            self.need_prec_update = False
            self._preconditioner_update()
            self.log.info('Spectral preconditioner updated')
          
        else:
            step, _ = TikhonovCG(
                setting=Setting(
                    self.deriv,
                    self.h_domain,
                    self.h_codomain,
                    regpar=self.regpar,
                    explicitDataWarning=False,
                ),
                data=self.data - self.y,
                init=self.init-self.x,
                preconditioner=self.preconditioner,
                **self.cg_pars
            ).run(stoprule=stoprule)

            
        self.x += step
        self.y, self.deriv = self.op.linearize(self.x)
        
        self.k+=1
        if (isqrt(self.k))**2 == self.k:
            self.need_prec_update = True
        try:
            self.regpar = _next_regularization_parameter(
                self._regpar_iterator
            )
        except StopIteration:
            self.converge()
                       
    def _preconditioner_update(self) -> None:
        r"""perform lanzcos method to calculate the preconditioner
        """
        L = np.zeros((self.krylov_order, self.krylov_order), dtype=self.op.domain.dtype)
        for i in range(0, self.krylov_order):
            L[i, i] = np.vdot(self.krylov_basis_img[i, :], self.krylov_basis_img_2[i, :])
            for j in range(i+1, self.krylov_order):
                L[i, j] = np.vdot(self.krylov_basis_img[i, :], self.krylov_basis_img_2[j, :])
                L[j, i] = L[i, j].conjugate()
        r"""Express `T*T` in Krylov_basis"""
        lamb, U = eigsh(L, self.number_eigenvalues, which='LM')
        """Perform the computation of eigenvalues and eigenvectors"""
        diag_lamb = np.diag( np.sqrt(1 / (lamb + self.regpar) ) - sqrt(1 / self.regpar) )
        M_krylov = self.op.domain.volume_elem*U @ diag_lamb @ U.transpose().conjugate()
        chars1 = ''.join(chr(i) for i in range(ord('a'), ord('a') + self.krylov_basis.ndim))
        chars2 = ''.join(chr(i) for i in range(ord('A'), ord('A') + self.krylov_basis.ndim))
        krylov_M_krylov=np.einsum(f"{chars1[0]+chars2[0]},{chars2}->{chars1[0]+chars2[1:]}",M_krylov,self.krylov_basis,optimize=True)
        tensors=(self.krylov_basis.conjugate(),krylov_M_krylov)
        subscript=f"{chars1[1:]},{chars1},{chars1[0]+chars2[1:]}"
        self.preconditioner=EinSum(subscript,self.op.domain,tensors=tensors,codomain=self.op.domain)+PtwMultiplication(self.op.domain,1/sqrt(self.regpar))
        """Compute preconditioner"""
