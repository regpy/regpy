from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Any
    from regpy.vecsps.base import ReadOnlyVector, Vector, VectorSpaceBase
    from regpy.operators.base import Operator
    from regpy.solvers.setting import Setting
    from regpy.stoprules.general import StopRule
    from regpy._typing import Mask
import numpy as np
from functools import partial

from regpy.vecsps.numpy import *
import regpy.operators as rop
from regpy.hilbert import L2
from regpy.util import Errors, memoized_property

from .base import Functional, AbstractFunctional, Composed
from .integral_functionals import LppPower, L1MeasureSpace, Huber

__all__ = ["VectorIntegralFunctional", "LppL2", "L1L2", "HuberL2", "TVUniformGridFcts"]

class VectorIntegralFunctional(Functional):
    r"""Implements a discrete version of vector-valued integral functionals 

    .. math::
        v\mapsto \int_D f(|v(x)|,x)d\mu(x)

    defined on vector-valued functions :math:`f:D\to \mathbb{R}^n` given by 

    .. math::
        v\mapsto \sum_i w_if_i(\|v_i\|)
    
    for v in  `regpy.vecsps.MeasureSpaceFcts`. 
    Here, :math:`f_i` are scalar functions on the real line that are assumed to be even and convex, 
    and :math:`\|\cdot\|` is the entrywise point-value norm. For a
    matrix-valued field such as a color-image gradient, p=2 yields the
    Frobenius norm over both the color and spatial-derivative axes.
    
    The norm on the vector values can be given as a method taking the vector-valued function and the vector axis 
    as tuple and returning the norm values. By default, the Euclidean norm is used.  
    
    Parameters
    ----------
    vdomain: VectorSpaceBase
        Domain of vector-valued functions on which it is defined. Needs some Measure therefore a `regpy.vecsps.MeasureSpaceFcts`
    scalar_func: Functional | None [default: None]
        Scalar separable functional to be used as (even!) functions :math:`f_i`.
        Either provides a specific separable Functional  
        or an AbstractFunctional which results in a `regpy.functionals.IntegralFunctionalBase`.
    scalar_func_args: dict[str, Any] | None [default: None]
        Additional arguments for the scalar_func if needed (e.g. 'p' for `Lpp`).
    vector_norm_p: int | float [default: 2]
        This is the exponent in the norm of the vectors given by a :math:`p` norm :math:`(|x_1|^p+\dots+|x_n|^p)^{1/p}}`. 
        By default, the Euclidean norm is used with :math:`p=2`.
    Lipschitz: float [default: np.inf]
        For the Euclidean norm as inner norm, :math:`Hess f_i(\|v\| )` has two eigenvalues: :math:`f_i''(\|v\|) `
        and :math:`f_i'(\|v\|)/\|v\|`. Using this fact, analytic expression may be computed for 
        Lipschitz = supremum of all possible EV and for convexity_param = infimum of all EV may be 
        computed for specific :math:`f_i` and passed as arguments to accelerate optimization methods.
    convexity_param: float [default: 0.0]
        For the Euclidean norm as inner norm, :math:`Hess f_i(\|v\| )` has two eigenvalues: :math:`f_i''(\|v\|) `
        and :math:`f_i'(\|v\|)/\|v\|`. Using this fact, analytic expression may be computed for 
        Lipschitz = supremum of all possible EV and for convexity_param = infimum of all EV may be 
        computed for specific :math:`f_i` and passed as arguments to accelerate optimization methods.
    methods: set[str] | None [default: None]
        Names of the implemented functional operations.
    conj_methods: set[str] | None [default: None]
        Names of the implemented functional operations.
    """    

    def __init__(self, vdomain: VectorSpaceBase, scalar_func: Functional | None = None, scalar_func_args: dict[str, Any] | None=None, 
                 vector_norm_p: int | float=2,
                 Lipschitz: float =np.inf, convexity_param: float = 0.,
                 methods: set[str] | None = None, conj_methods: set[str] | None = None
                 ) -> None:
        if not isinstance(vdomain, MeasureSpaceFcts) or vdomain.ndim_codomain < 1:
            raise ValueError(Errors.not_instance(
                vdomain, MeasureSpaceFcts,
                "The vector domain must be a measure space with at least "
                "one point-value axis.",
            ))
        self.vdomain = vdomain
        self.sdomain = vdomain.scalar_space()
        self.scalar_func_args = scalar_func_args if scalar_func_args is not None else {}

        if isinstance(scalar_func, AbstractFunctional):
            scalar_func = scalar_func(self.sdomain, **self.scalar_func_args)
        elif scalar_func is None:
            scalar_func = LppPower(self.sdomain, p=2., **self.scalar_func_args)
        if not scalar_func.is_separable or scalar_func.domain!=self.sdomain:
            raise ValueError(Errors.value_error(f'{scalar_func} need to be a callable giving a separable functional or already a separable functional with sdomain as domain.'))
        else:
            self.scalar_func = scalar_func

        if isinstance(vector_norm_p,(int,float)) and vector_norm_p >1 and vector_norm_p<np.inf:
            self.p = vector_norm_p
            self.q = vector_norm_p / (vector_norm_p - 1) 
            # np.linalg.norm(ord=2, axis=(-2, -1)) computes the *spectral*
            # norm of a matrix, not its Frobenius norm. For TV of a vector-
            # valued function, a pointwise Jacobian has one value axis for
            # channels and one for spatial derivatives. Use the entrywise
            # ell_p norm over *all* value axes (Frobenius for p=2).
            self.vector_norm = partial(self._pointwise_lp_norm, p=self.p)
            self.dual_vector_norm = partial(self._pointwise_lp_norm, p=self.q)
        else:
            raise ValueError(Errors.value_error("Not p-Norm", f"The provided p ={vector_norm_p} is not between 1< p < inf."))

        methods = methods if methods is not None else self.scalar_func.methods
        conj_methods = conj_methods if conj_methods is not None else self.scalar_func.conj.methods
        if self.p!=2:
            methods -= {'subgradient','hessian','proximal'}
            conj_methods =  {'subgradient','hessian','proximal'}

        super().__init__(vdomain, 
                         L2(vecsp=vdomain), 
                         convexity_param = convexity_param, 
                         Lipschitz = Lipschitz, 
                         is_separable = False, 
                         is_linear = self.scalar_func.is_linear,
                         is_quadratic = self.scalar_func.is_quadratic,
                         methods = methods, 
                         conj_methods = conj_methods
                         )
        self._sbuf = self.sdomain.zeros()
        self._vbuf = vdomain.zeros()
        self._vaxes = tuple(range(-self.vdomain.ndim+self.vdomain.ndim_domain,0))

    @staticmethod
    def _pointwise_lp_norm(x: ReadOnlyVector, axis: tuple[int, ...],
                           p: float) -> np.ndarray:
        """Entrywise l_p norm over all trailing value axes.

        For matrices and higher tensors this is a vector (Schatten-free)
        entrywise norm; for p=2 it is the Frobenius norm. Never pass a
        multi-axis tuple with ord=2 to numpy.linalg.norm, because that
        requests a spectral matrix norm instead.
        """
        magnitude = np.abs(x)
        if p == 2:
            return np.sqrt(np.sum(magnitude * magnitude, axis=axis))
        return np.sum(magnitude ** p, axis=axis) ** (1. / p)

    @property
    def _sbuf_ext(self) -> np.ndarray:
        return np.expand_dims(self._sbuf, axis=self._vaxes)

    def _eval(self, vec: ReadOnlyVector) -> float:
        self._sbuf = self.vector_norm(x = vec, axis=self._vaxes)
        return self.scalar_func(self._sbuf)

    def _conj(self, vec_star: ReadOnlyVector) -> float:
        self._sbuf = self.dual_vector_norm(x = vec_star, axis=self._vaxes)
        return self.scalar_func.conj(self._sbuf)

    def _subgradient(self, 
                     vec: ReadOnlyVector, 
                     out: Vector | None=None, 
                     **subgradient_par: Any
                     ) -> Vector:
        r"""returns
        
        .. math::
            [f_i'(\|v_i\|)\frac{v_i}{\|v_i\|}]_i
        
        Parameters
        ----------
        vec: ReadOnlyVector
            Input vector; its coefficients are only read.
        out: Vector | None [default: None]
            Writable output buffer. If None, allocate a new result.
        **subgradient_par: Any
            Subgradient par.
        
        
        Returns
        -------
        Vector
            Writable result vector.
        """
        if self.p != 2:
            raise NotImplementedError('subgradient only implemented for L2 norm')
        result = np.empty_like(vec) if out is None else out
        self._sbuf = self.vector_norm(vec, axis=self._vaxes)
        positive = self._sbuf_ext > 0
        np.divide(vec, self._sbuf_ext, where=positive, out=result)
        np.copyto(result, 0, where=~positive)
        self.scalar_func.subgradient(self._sbuf, out=self._sbuf)
        np.multiply(result, self._sbuf_ext, out=result)
        return result
    
    def _conj_subgradient(self, 
                          vec_star: ReadOnlyVector, 
                          out: Vector | None=None, 
                          **subgradient_par: Any
                          ) -> Vector:
        if self.p != 2:
            raise NotImplementedError('conjugate subgradient only implemented for L2 norm')
        result = np.empty_like(vec_star) if out is None else out
        self._sbuf = self.dual_vector_norm(vec_star, axis=self._vaxes)
        positive = self._sbuf_ext > 0
        np.divide(vec_star, self._sbuf_ext, where=positive, out=result)
        np.copyto(result, 0, where=~positive)
        self.scalar_func.conj_subgradient(self._sbuf, out=self._sbuf)
        np.multiply(result, self._sbuf_ext, out=result)
        return result

    def _proximal(self, 
                  vec: ReadOnlyVector, 
                  tau: float, 
                  mask: Mask | None=None, 
                  out: Vector | None=None, 
                  **proximal_par: Any
                  ) -> Vector:
        r"""returns :math:`[prox_{\tau f_i}(\|v_i\|)\frac{v_i}{\|v_i\|}]_i`
        
        Parameters
        ----------
        vec: ReadOnlyVector
            Input vector; its coefficients are only read.
        tau: float
            Proximal or iteration step size.
        mask: Mask | None [default: None]
            Coordinate-selection mask in the format used by the vector space.
        out: Vector | None [default: None]
            Writable output buffer. If None, allocate a new result.
        **proximal_par: Any
            Proximal par.
        
        
        Returns
        -------
        Vector
            Writable result vector.
        """
        if self.p != 2:
            raise NotImplementedError('proximal only implemented for L2 norm')
        result = np.empty_like(vec) if out is None else out
        self._sbuf = self.vector_norm(vec, axis=self._vaxes)
        positive = self._sbuf_ext > 0
        np.divide(vec, self._sbuf_ext, where=positive, out=result)
        np.copyto(result, 0, where=~positive)
        self.scalar_func.proximal(
            self._sbuf, tau, out=self._sbuf, mask=None, **proximal_par
        )
        np.multiply(result, self._sbuf_ext, out=result)
        return result

    def _conj_proximal(self, 
                       vec_star: ReadOnlyVector, 
                       tau: float, 
                       mask: Mask | None=None, 
                       out: Vector | None=None, 
                       **proximal_par: Any
                       ) -> Vector:
        if self.p != 2:
            raise NotImplementedError('conjugate proximal only implemented for L2 norm')
        result = np.empty_like(vec_star) if out is None else out
        self._sbuf = self.dual_vector_norm(vec_star, axis=self._vaxes)
        positive = self._sbuf_ext > 0
        np.divide(vec_star, self._sbuf_ext, where=positive, out=result)
        np.copyto(result, 0, where=~positive)
        self.scalar_func.conj_proximal(
            self._sbuf, tau, out=self._sbuf, mask=None, **proximal_par
        )
        np.multiply(result, self._sbuf_ext, out=result)
        return result
    
    def _hessian(self, 
                 vec: ReadOnlyVector
                 ) -> Operator:
        if self.p != 2:
            raise NotImplementedError('proximal only implemented for L2 norm')
        self._sbuf = self.vector_norm(vec, axis=self._vaxes)
        self._vbuf.fill(0)
        np.divide(vec,self._sbuf_ext,where=self._sbuf_ext>0,out=self._vbuf)
        fp = self.scalar_func.subgradient(self._sbuf)
        fp = np.expand_dims(fp,self._vaxes)
        np.divide(fp,self._sbuf_ext, where=self._sbuf_ext>0, out= fp)
        fpp = self.scalar_func.hessian(self._sbuf)(self.sdomain.ones())
        # As f is even, f' is odd, so f'(0)=0 if f'' exists. Therefore, f'(v)/v tends to f''(v) as 
        # v tends to 0, so we can use this a places where ||v|| vanishes.
        fpp = np.expand_dims(fpp,self._vaxes)
        fp[~(self._sbuf_ext>0)] = fpp[~(self._sbuf_ext>0)]
        scal_mult =rop.PtwScalarMultiplication(self.domain,fp)
        proj = rop.PtwMatrixVectorMultiplication(self.vdomain,
                                             np.expand_dims(self._vbuf, axis=self.vdomain.ndim_domain).copy()
                                             )
        # actually just a pointwise multiplication, but implemented multiplication with a 1x1 matrix 
        # to avoid conversions to scalar functions
        ptw_mult = rop.PtwMatrixVectorMultiplication(self.sdomain.vector_valued_space(1),
                                                 np.reshape(fpp - fp, self.sdomain.shape + (1, 1))
                                                )

        return scal_mult + proj.adjoint * ptw_mult * proj                                             
        
    def _conj_hessian(self, 
                      vec: ReadOnlyVector
                      ) -> Operator:
        if self.p != 2:
            raise NotImplementedError('proximal only implemented for L2 norm')
        self._sbuf = self.dual_vector_norm(vec, axis=self._vaxes)
        self._vbuf.fill(0)
        np.divide(vec,self._sbuf_ext,where=self._sbuf_ext>0,out=self._vbuf)
        fp = self.scalar_func.conj.subgradient(self._sbuf)
        fp = np.expand_dims(fp,self._vaxes)
        np.divide(fp,self._sbuf_ext, where=self._sbuf_ext>0, out= fp)
        fpp = self.scalar_func.conj.hessian(self._sbuf)(self.sdomain.ones())
        # see comments for hessian!
        fpp = np.expand_dims(fpp,self._vaxes)
        fp[~(self._sbuf_ext>0)] = fpp[~(self._sbuf_ext>0)]
        scal_mult = rop.PtwScalarMultiplication(self.domain,fp)
        proj = rop.PtwMatrixVectorMultiplication(self.vdomain,
                                             np.expand_dims(self._vbuf, axis=self.vdomain.ndim_domain).copy()
                                             )
        ptw_mult = rop.PtwMatrixVectorMultiplication(self.sdomain.vector_valued_space(1),
                                                 np.reshape(fpp - fp, self.sdomain.shape + (1, 1))
                                                )

        return scal_mult + proj.adjoint * ptw_mult * proj   

    @property
    def is_norm_squared_type(self) -> bool:
        r"""Whether the functional is norm squared type.
        
        
        Returns
        -------
        bool
            Whether the stated condition is satisfied.
        """
        return self.scalar_func.is_norm_squared_type

class LppL2(VectorIntegralFunctional):
    r"""VectorIntegralFunctional with :math:`f_i(x):=(1/p)|x|^p`.
    
    Parameters
    ----------
    vdomain: VectorSpaceBase
        A vector valued measure space.
    p: float [default: 2.0]
        exponent of the :math:`L^p` norm.
    """
    def __init__(self, vdomain: VectorSpaceBase,p: float=2.) -> None:
        sfunc = LppPower(vdomain.scalar_space(),p=p)
        super().__init__(vdomain, scalar_func=sfunc, 
                         Lipschitz=1. if p== 2 else np.inf,
                         convexity_param=1. if p==2 else 0.
                         )

class L1L2(VectorIntegralFunctional):
    r"""VectorIntegralFunctional with absolute value function as :math:`f_i`.
    
    Parameters
    ----------
    vdomain: VectorSpaceBase
        A vector valued measure space.
    conj_tol: float [default: 1e-16]
        A tolarance introduced into `L1` functional that allows also numbers slighly bigger then 1.0 in the conjugate.
    """
    def __init__(self, vdomain: VectorSpaceBase, conj_tol: float = 1e-16) -> None:
        sfunc = L1MeasureSpace(vdomain.scalar_space(), conj_tol= conj_tol)
        super().__init__(vdomain, scalar_func= sfunc, Lipschitz=np.inf)

class HuberL2(VectorIntegralFunctional):
    r"""Isotropic Huber functional of the pointwise Euclidean/Frobenius norm.

    Vector and matrix point values share the same radial Huber envelope.
    For a color-image Jacobian, each pixel uses the Frobenius norm across
    all colors and all spatial derivatives (joint isotropic TV).
    
    Parameters
    ----------
    vdomain: VectorSpaceBase
        A vector valued measure space.
    sigma: float [default: 1.0]
        Parameter in Huber functional
    """
    def __init__(self, vdomain: VectorSpaceBase,sigma: float=1.) -> None:
        sfunc = (1./sigma)*Huber(vdomain.scalar_space(),sigma=sigma)
        super().__init__(vdomain, scalar_func=sfunc, Lipschitz=sfunc.Lipschitz)

class TVUniformGridFcts(Composed):
    r"""TVUniformGridFcts interface.
    
    Parameters
    ----------
    domain: VectorSpaceBase
        Domain vector space.
    beta: float [default: 0.0]
        Beta.
    boundary_condition: str [default: 'Neum']
        Boundary condition.
    prox_method: str [default: 'dFISTA']
        Prox method.
    prox_method_param: dict [default: {}]
        Options for prox method param.
    prox_stop_rule: StopRule | None [default: None]
        Prox stop rule.
    logging_level: int | str [default: 'INFO']
        Logging level for this object.
    prox_it_logging_level: str [default: 'INFO']
        Prox it logging level.
    """
    import logging
    r"""Total Variation Norm: For :math:`C^1` functions the :math:`l^1`-norm of the gradient on a `UniformGrid`

    Parameters
    ----------
    domain : UniformGridFcts | MaskedVectorSpace
        Uniform grid or a masked space obtained from a UniformGridFcts.
        Masked vectors are extended by zero before finite differences; TV
        therefore includes jumps across the mask boundary.  The gradient
        codomain is the full-grid vector-valued measure space. 
    h_domain : regpy.hilbert.HilbertSapce (default: L2)
        Underlying Hilbert space for proximal. 
    beta: float (>=0) [optional, default:0.]
        If beta>0, the L^1-norm is approximated by a Huber functional with this parameter. 
    boundary_condition: 'Neum', 'Diri' or 'per' (default: 'Neum')
        Boundary condition in the discretization of gradient, see ForwardFDGradient
    prox_method: str, optional
        A string specifying the method for computing the prox operator. This must be a solver for the class-internal
        TV_prox_setting derived from Setting. The default is "dFISTA". An alternative -- besides most other standard solvers 
        for convex Tikhonov settings -- is "Chambolle" implementing Chambolle's dual fixed point iteration.  
    prox_method_param: dict, optional
        parameters of the prox_method. Defaults to {}
    prox_stop_rule: StoprRule or None, optional
        StopRule of the prox_methods. Defaults to DualityGapStopping(tol=1e-4,stop_at_infty = False) + CountIterations(500)
    """

    ###############################################################################
    from regpy.solvers.setting import Setting
    class TV_prox_setting(Setting):
        r"""Rockafellar-Fenchel type setting for computing the proximal mapping of the TV functional: 
        
        ::math::
            prox_{tau TV} = argmin_u  S(grad u) +  \frac{1}{2*tau} \|u-x\|_{L^2}^2
        
        where S is either an L1L2 functional (for beta==0) or a HuberL2 functional
            
        Parameters
            tv: TVUnivormGridFct 
               the TV functional
            penalty_shift: np.ndarray | None (optional)
                The argument x. Defaults to None correpsonding to x=0
            tau: float (optional)
                 Parameter of the prox. Defaults to 1. 
            
        Parameters
        ----------
        tv: TVUniformGridFcts
            Tv.
        penalty_shift: ReadOnlyVector | None [default: None]
            Input vector; its coefficients are only read.
        tau: float [default: 1.0]
            Proximal or iteration step size.
        """  
        def __init__(self, tv: TVUniformGridFcts,
                     penalty_shift: ReadOnlyVector | None = None,
                     tau:float = 1.
                     ) -> None:
            super().__init__(op=tv.grad,
                             data_fid = tv.func, 
                             penalty = L2, 
                             init = penalty_shift,
                             incorporate_init_in_penalty=True,
                             regpar = 1./tau,
                             explicitDataWarning=False)

        def _generate_full_solver_dictionary(self)->dict:
            _methods = super()._generate_full_solver_dictionary()
            _methods.update({'Chambolle':{'class':TVUniformGridFcts.Chambolle,
                                          'primal':False,
                                          'full':"Chambolle's dual fixed point iteration"}})
            return _methods

    #############################################################################
    from regpy.solvers.solver import RegSolver
    class Chambolle(RegSolver):
        r"""Fixed point iteration for the dual problem suggested by A. Chambolle (J. Math. Imaging and Vision 20: 89-97, 2004). 
        Compared to this original reference, the free variable is rescaled by the factor (tau*grid.volume_elem)
        to match the free variable of the dual of the TV_prox_setting, and it is adapted also to the case of Huber-TV, i.e., beta>0. 
        
        Parameters
        ----------
        setting: Setting
            Problem setting containing the operator, penalty, and data fidelity.
        x: ReadOnlyVector | None [default: None]
            First argument of prox
        y: ReadOnlyVector | None [default: None]
            Input vector; its coefficients are only read.
        stepsize_safety: float [default: 1.0]
            Safety parameter for the stepsize. Convergence is guaranteed for values <=1, but empirically, 
            best results are optained for stepsize_safety =2.
        **kwargs: Any
            Additional keyword arguments forwarded to the underlying implementation.
        
        Other Parameters
        ----------------
        tau: float >=0
            Second (scaling) argument of prox
        """           
        def __init__(self, setting: Setting, x: ReadOnlyVector | None=None, y: ReadOnlyVector | None=None, stepsize_safety: float = 1., **kwargs: Any) -> None:
            out,_ = TVUniformGridFcts.Chambolle.check_applicability(setting)
            if not out['applicable']:
                raise ValueError(out['info'])
            super().__init__(setting, x=x,y=y)
            self.lastx = self.x
            primal = setting._primal_setting
            # Masked spaces store flattened values, but their L2 geometry
            # retains the same cell volume as the generating uniform grid.
            self.vol_el = TVUniformGridFcts._grid_and_extension(
                primal.op.domain
            )[0].volume_elem            
            self.grad = primal.op
            self.x = x if x is not None else  self.grad.codomain.zeros()
            self.S = primal.data_fid
            self.R = primal.penalty
            self.ub = self.vol_el * setting.regpar 
            self.stepsize = stepsize_safety/self.grad.norm()**2 
            self.ref = -primal.penalty.subgradient(self.grad.domain.zeros())
            """the argument of the prox operator, transformed to the dual space"""
            if isinstance(self.S,HuberL2):
                self.hessian = setting.penalty.hessian(setting.penalty.domain.zeros())
                self.beta = setting.penalty.convexity_param # the parameter in the Huber functional
            self.y = self.grad.adjoint(self.x)

        def _next(self) -> None:
            self.lastx = self.x; 
            self.u = self.y+self.ref
            self.update = self.grad(-self.u)
            if isinstance(self.S,HuberL2):
                self.update -= (self.regpar*self.beta)*self.x
            self.update *= self.stepsize
            if isinstance(self.S,HuberL2):
                #compute projection of x + update onto feasible set
                ptw_norms = self.S.vector_norm(x = self.x + self.update, axis=self.S._vaxes)
                ptw_norms[ptw_norms<=self.ub] = self.ub
                self.x = (
                    self.ub / np.expand_dims(ptw_norms, axis=self.S._vaxes)
                ) * (self.x + self.update)
            else:         
                denom = 1. + (1. / self.ub) * self.S.vector_norm(
                    x=self.update, axis=self.S._vaxes,
                )
                self.x = (self.x + self.update) / np.expand_dims(
                    denom, axis=self.S._vaxes,
                )
            self.y = self.grad.adjoint(self.x)

        def primal(self) -> tuple[Vector | None, Vector | None]:
            r"""primal variables (of the dual problem) needed for the evaluation of the dualitzy gap
                        
            
            Returns
            -------
            tuple[Vector | None, Vector | None]
                Result described above.
            """
            return (self.x,self.y)

        def dual(self) -> tuple[Vector | None, Vector | None]:
            r"""dual variables of the dual problem (i.e., actually the primal variables) 
            needed for the evaluation of the dualitzy gap
                        
            
            Returns
            -------
            tuple[Vector | None, Vector | None]
                Result described above.
            """
            if hasattr(self,'u'):
                return ((1./self.vol_el)*self.u,(-1./(self.vol_el*self.stepsize))*self.update) 
            else: # at first iterate
                return (self.y,self.x)

        @staticmethod
        def check_applicability(setting: Setting, op_norm: float | None=None)->tuple[dict,dict]:
            r"""Check the requirements of this solver for the supplied setting.
            
            Parameters
            ----------
            setting: Setting
                Problem setting containing the operator, penalty, and data fidelity.
            op_norm: float | None [default: None]
                Operator-norm estimate; None requests numerical estimation.
            
            
            Returns
            -------
            tuple[dict, dict]
                Result described above.
            """
            from regpy.solvers import Setting            
            out = {'info':''}; par = {}            
            if not isinstance(setting,Setting) or not isinstance(setting._primal_setting,TVUniformGridFcts.TV_prox_setting):
                out['info'] += "Chambolle's method is only applicable in the dual of a TV_prox_setting."
            out['applicable'] = out['info'] == ''
            out['rate']=np.nan            
            return out,par 

    @staticmethod
    def _grid_and_extension(domain: VectorSpaceBase):
        """Return the underlying uniform grid and zero-extension operator.

        Nested masks are supported.  A mask on a different kind of measure
        space is deliberately not interpreted as a regular spatial grid.
        """
        from regpy.vecsps.maskable import MaskedVectorSpace

        if isinstance(domain, UniformGridFcts):
            return domain, None
        if not isinstance(domain, MaskedVectorSpace):
            raise TypeError(
                "TV requires a UniformGridFcts or a masked space derived "
                "from UniformGridFcts."
            )

        current = domain
        extension = None
        while isinstance(current, MaskedVectorSpace):
            full = current.get_full_space()
            step = rop.CoordinateProjection(full, current.mask).adjoint
            extension = step if extension is None else step * extension
            current = full
        if not isinstance(current, UniformGridFcts):
            raise TypeError(
                "TV requires a masked space derived from UniformGridFcts."
            )
        return current, extension

    #####################################################################
    def __init__(self, domain: VectorSpaceBase,
                 beta:float =0.,
                 boundary_condition:str='Neum',
                 prox_method:str = 'dFISTA',
                 prox_method_param:dict = {},
                 prox_stop_rule: StopRule | None = None,
                 logging_level: int | str = "INFO",
                 prox_it_logging_level: str = "INFO"
                ) -> None:
        from regpy.stoprules import DualityGapStopping,CountIterations,StopRule

        # On a masked grid, compute spatial derivatives on the original
        # uniform grid after extending missing coefficients by zero.  This
        # preserves the spatial geometry, the value axes (e.g. RGB channels),
        # and the correct TV jump across the support boundary.
        full_grid, self.extension = self._grid_and_extension(domain)
        self.full_grid = full_grid
        full_grad = rop.ForwardFDGradient(
            full_grid, boundary_condition=boundary_condition
        )
        self.grad = (full_grad if self.extension is None
                     else full_grad * self.extension)

        self.beta = beta  # also sets self.func as L1L2 or HuberL2
        # Zero extension is isometric for the standard uniform-grid L2 norm,
        # so the analytic full-grid gradient bound remains valid.  Avoid
        # estimating the norm of a potentially large composition numerically.
        super().__init__(self.func, op=self.grad, op_norm=full_grad.norm(),
                         methods={'eval', 'proximal'},
                         conj_methods={'proximal'})

        self.prox_setting = TVUniformGridFcts.TV_prox_setting(self)
        if prox_method not in self.prox_setting.applicable_methods():
            raise ValueError(Errors.value_error(f"{prox_method} not in list of applicable methods of the TV_pvrox_setting, which is {self.prox_setting.applicable_methods()}."))
        else:
            self.prox_method = prox_method      
        self._fista_zero_data = {}
        if prox_stop_rule is None:
            # The TV proximal objective may be extremely small, especially on
            # masked grids or for small-amplitude iterates.  An absolute
            # duality-gap tolerance such as 1e-4 can therefore certify the
            # initial (nonoptimal) iterate before FISTA has taken a step.
            # Use a scale-independent relative criterion instead.
            prox_stop_rule = (
                DualityGapStopping(
                    tol=1e-12, reltol=0.1, stop_at_infty=False,
                    logging_level=prox_it_logging_level,
                )
                + CountIterations(500, logging_level=prox_it_logging_level)
            )
        elif not isinstance(prox_stop_rule,StopRule):
            raise ValueError(Errors.value_error(f"{prox_stop_rule} is not a StopRule."))
        if not isinstance(prox_method_param,dict):
            raise ValueError(Errors.value_error(f"{prox_method_param} is not a dictionary."))
        else:
            self.prox_method_param = prox_method_param
        self.prox_stop_rule = prox_stop_rule
        self.log.setLevel(logging_level)

    @property
    def beta(self) -> ReadOnlyVector | float | None:
        r"""Parameter in Huber approximation of the L^1-norm. Exact TV functional only for beta==0.
        
        
        Returns
        -------
        ReadOnlyVector | float | None
            Result described above.
        """
        return self._beta
    
    @beta.setter
    def beta(self, beta_val: float) -> None:
        r"""Sets beta as well as the functional that is applied to the gradient (L1L2 if beta == 0 and HuberL2 else)
        
        Parameters
        ----------
        beta_val: float
            Beta val.
        """
        if not isinstance(beta_val,(float,int)) and (beta_val>=0.):
            raise ValueError(Errors.value_error('beta must be a non-negative float'))
        else:
            self._beta = float(beta_val)
        self._beta = beta_val
        self.func = self._grad_func(beta_val)
        if hasattr(self,'prox_setting'):
            self.prox_setting.data_fid = self.func

    def _grad_func(self, beta: float)->Functional:
        r"""Returns the functional applied to the gradient -- L1L2 if beta ==0 and HuberL2 if beta>0
                
        Parameters
        ----------
        beta: float
            Beta.
        
        
        Returns
        -------
        Functional
            Constructed functional.
        """
        if beta==0.:
            func = L1L2(self.grad.codomain)
            func.scalar_func.conj_tol = 1e-5 # increase tolerance to avoid inf's in duality gap evaluations
        else:
            func = HuberL2(self.grad.codomain,sigma=beta)
            func.sigmaeps = np.broadcast_to(1.+1e-5, self.grad.domain.shape) # increase tolerance to avoid inf's in duality gap evaluations
        return func
      
    def _proximal(self, x: ReadOnlyVector, tau: float, out: Vector | None=None, **kwargs: Any) -> Vector:
        # Setting.init also updates penalty.as_data_func(x) because the
        # proximal setting uses incorporate_init_in_penalty=True.
        self.prox_setting.init = x
        self.prox_setting.regpar = 1. / tau
        prox_method_param = self.prox_method_param.copy()
        prox_method_param.update(kwargs)
        if self.prox_method in {'FISTA', 'dFISTA'} and 'data' not in prox_method_param:
            data_space = (
                self.grad.domain
                if self.prox_method == 'dFISTA'
                else self.grad.codomain
            )
            if self.prox_method not in self._fista_zero_data:
                self._fista_zero_data[self.prox_method] = data_space.zeros()
            prox_method_param['data'] = self._fista_zero_data[self.prox_method]
            prox_method_param.setdefault('update_setting', False)
        result = self.prox_setting.run(
            self.prox_method, stoprule=self.prox_stop_rule,
            **prox_method_param
        )[0]
        if out is None:
            return result
        else:
            np.copyto(out,result)
            return out
       
