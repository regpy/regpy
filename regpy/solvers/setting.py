from __future__ import annotations
import math as ma
import numpy as np
from typing import Callable, Any
from copy import deepcopy

from regpy.util import ClassLogger, Errors
from regpy.vecsps import Vector, ReadOnlyVector
from regpy.operators import Operator, ObsDataT
from regpy.hilbert import HilbertSpace
from regpy.functionals.base import  as_functional, Composed
from regpy.functionals import Functional
from regpy.stoprules import (
    StopRule, DualityGapStopping, CountIterations, ErrorMonitor,
)
__all__ = [
    "Setting",
    "get_reg_method_dictionary",
    "get_linear_reg_method_dictionary",
    "get_nonlinear_reg_method_dictionary",
    "get_convex_solver_dictionary",
    "get_non_convex_solver_dictionary",
    "check_Hilbert_applicability",
    "check_Hilbert_applicability_linear",
]

_DEFAULT_STOPRULE = object()

class Setting:
    r"""A *setting* for an inverse problem, used by RegSolvers. A setting always consists at least of

    - a forward operator :math:`F`,
    - a penalty functional :math:`\mathcal{R}` with an associated Hilbert space structure to measure the error, and
    - a data fidelity functional :math:`\mathcal{S}_{g^{\delta}}` with an associated Hilbert space structure to measure the data misfit.

    If a regularization parameter :math:`alpha` is given, this is the setting for the minimization problem 

    .. math::
        \frac{1}{\alpha}\mathcal{S}_{g^{\delta}}(F(f)) + \mathcal{R}(f) = \min!

    If the operator is linear and both functionals are convex, this is the setting of Rockafellar-Fenchel duality 
    --- a rich and algorithmically useful mathematical structure. In this case, the dual setting 
    and primal-dual optimality conditions are provided. 
   
    This class is mostly a container that keeps all of this data in one place and makes sure that all initializations are 
    done correctly.

    A setting with a regularization parameter describes a minimization problem. Solvers approximately compute minimizers, which 
    serve as regularized solutions to the inverse problem. 
    Settings without a regularization parameter indicate prior information through the penalty functional, but do not specify a unique estimator 
    of the solution to the inverse problem. Solver may iterate over regularization
    parameters under the control of a parameter-choice rule, which requires inner minimization solvers. 
    Other iterative regularization methods such as gradient methods use the penalty functional more implicitely, e.g., via its 
    proximal operator. Since such methods are semiconvergent for ill-posed problems, an appropriate :class:`regpy.stoprules.StopRule` is
    essential.

    Parameters
    ----------
    op : regpy.operators.Operator
        The forward operator :math:`F`.
    penalty : regpy.functionals.Functional or regpy.hilbert.HilbertSpace or callable
        The penalty functional  :math:`\mathcal{R}`.
        If a Hilbert space is given, its squared norm, represented by
        :class:`regpy.functionals.SquaredNorm`, is used as the penalty.
        If an `AbstractFunctional`, or more generally any callable is given instead of a `regpy.functionals.Functional`, it is called on the operator's domain to construct a concrete `Functional` instance.
        The penalty need not be a data functional. Its
        :meth:`~regpy.functionals.Functional.reference_vector` provides the
        default initial guess.
    data_fid : regpy.functionals.Functional or regpy.hilbert.HilbertSpace or callable or None
        The data misfit functional :math:`\mathcal{S}_{g^{\delta}}`.
        The cases of Hilbert space and callable instances are treated in analogy to penalty. 
        None is only admissible if codomain_free = True.
    regpar: float or None [default: None]
        regularization parameter  :math:`alpha`. 
        Defaults to None. See explanations above on different meaning of solvers in this case.
    data: Vector or ObsDataT or None [default:None]
        If not None, the data :math:`g^{\delta}` in the data fidelity functional is replaced by data 
        (which often but not necessarily belong to op.codomain). 
    exact_data: Vector or None [default: None]
        Optional exact data used to generate synthetic noisy observations with
        :meth:`add_Gaussian_noise` or :meth:`generate_Poisson_data`. Until one
        of these methods is called, ``exact_data`` does not affect ``data``.
    noise_pars: dict or None [default: None]
        A dictionary containing parameters of the data noise model. 
        This information is required by parameter choice rules.
        If you work with synthetic data, you can use methods of this class to 
        "add noise" to your data 
    init: Vector or None [default: None]
        Initial guess in ``op.domain``. Depending on
        ``incorporate_init_in_penalty``, it is either incorporated into the
        penalty or stored independently. If no initial guess is supplied,
        :attr:`init` returns a writable copy of the penalty's
        :meth:`~regpy.functionals.Functional.reference_vector`.
    incorporate_init_in_penalty: bool [default: True]
        If true, a supplied initial guess is incorporated into the penalty by
        making the penalty a data functional. If false, an independent copy
        is stored in the setting. This option applies only to the ``init``
        argument passed to the constructor and has no effect when ``init`` is
        ``None``. Later assignments to :attr:`init` are stored independently.
    codomain_free: bool [default: False]
        If True, this indicates that vectors in the codomain of the operator should be avoided, e.g., because 
        they are too large to be stored. In this case inversion methods that only rely on 
        op.adjoint_data, op.adjoint_eval, and op.adjoint_derivative, but no evaluations of op and its derivative should be used. 
        The data fidelity term is inactive in this case, and op.data contains raw data, 
        typically not in the codomain, which are backpropagated by op.adjoint_data.
    logging_level: int or str [default: logging.INFO]
        logging level
    explicitDataWarning: bool [default: True]
        If True, a warning will be given if the setting is initialized without explicit data or exact_data. 
    default_linear_regmethod: str [default: 'genTikh']
        Default solver if the setting has a regularization parameter and the operator is linear.    
    default_nonlinear_regmethod: str [default: 'IRN']
        Default solver if the setting has a regularization parameter and the operator is linear.  
    default_inner_solver: str or None [default: None]
        Default inner solver if the setting has a regularization parameter and the Tikhonov functional is convex. 
    default_nl_minimizer: str|None [default: None]
        Name of default method for minimizing non-convex Tikhonov functional. 
    """
    
    log = ClassLogger()

    def __init__(self, 
                 op: Operator, 
                 penalty: Functional|HilbertSpace|Callable, 
                 data_fid:Functional|HilbertSpace|Callable|None,
                 regpar: float|None =None,
                 data: ReadOnlyVector | ObsDataT | None = None, 
                 exact_data: ReadOnlyVector | None = None, 
                 init: ReadOnlyVector | None = None,
                 incorporate_init_in_penalty: bool = True,
                 codomain_free: bool = False,
                 logging_level = "INFO",
                 explicitDataWarning: bool = True,                 
                 default_linear_regmethod: str = 'genTikh', 
                 default_nonlinear_regmethod: str = 'IRN', 
                 default_inner_solver: str|None = None, 
                 default_nl_minimizer: str|None = 'genFISTA',             
                 _primal_setting:Setting | None =None # intended only for internal use in get_dual_setting()                 
                 ) -> None:
        if not isinstance(op,Operator):
            raise TypeError(Errors.not_instance(op,Operator,add_info="Setting requires op to be a RegPy operator."))
        self._op = op
        self._codomain_free = codomain_free
        if not isinstance(incorporate_init_in_penalty, (bool, np.bool_)):
            raise TypeError(Errors.type_error(
                'incorporate_init_in_penalty must be boolean.',
                incorporate_init_in_penalty,
            ))

        self.set_penalty(penalty)

        if init is not None:
            if incorporate_init_in_penalty:
                self.set_penalty(self.penalty.as_data_func(init))
            else:
                self.init = init

        if not codomain_free:
            if data_fid is None:
                raise ValueError(Errors.value_error('data_fid must not be None unless codomain_free is True'))
            self.set_data_fid(data_fid)    
            if (not self.data_fid.is_data_func and data is None and _primal_setting is None):
                if exact_data is None and explicitDataWarning:
                    self.log.warning("Setting does not contain any explicit data.")
                self._data = None
            if self.data_fid.is_data_func:
                self._data=self.data_fid.data #just update internal data, update of data functional not necessary
            if exact_data is not None:
                if not exact_data in op.codomain:
                    raise ValueError(Errors.value_error("exact_data must be in codomain of operator."))
                self._exact_data = exact_data
        elif data_fid is not None:
            _data_fid = as_functional(data_fid, self.op.codomain)
            hilb = _data_fid.h_domain
            if not _data_fid.is_norm_squared_type:
                raise ValueError(Errors.value_error('Can only treat norm_squared_type functionals in the codomain_free case.', _data_fid))
            try:
                self.op.set_gram_factor(hilb)
            except ValueError:
                self.log.warning("The given data fidelity functional cannot be incorporated into the operator's forward-backward methods.")

        if data is not None:
            self.data = data #data and (if not self.codomain_free) data fidelity functional are updated

        self.log.setLevel(logging_level)
        if _primal_setting is not None and not (_primal_setting.is_convex and _primal_setting.is_tikhonov):
            raise ValueError(Errors.value_error("The primal_setting needs to be convex and contain a regularization parameter!"))
        self._primal_setting = _primal_setting
        self.regpar=regpar#The flags are set by setting the regularization parameter

        self._default_linear_regmethod = default_linear_regmethod
        self._default_nonlinear_regmethod = default_nonlinear_regmethod
        self._default_inner_solver = default_inner_solver
        self._default_nl_minimizer = default_nl_minimizer

        self._noise_model_param = {}

    @property
    def op(self)->Operator:
        r"""The operator."""
        return self._op
    
    @property
    def penalty(self)->Functional:
        r"""The penalty functional."""        
        return self._penalty

    def set_penalty(self, penalty)->None :
        self._penalty = as_functional(penalty, self.op.domain)
        self._init_method_dictionary()

    @property
    def init(self) -> Vector:
        r"""Return the initial guess as an independent writable vector.

        An independently stored user initialization is copied. Otherwise, the
        penalty's :meth:`~regpy.functionals.Functional.reference_vector` is
        written into a newly allocated vector. Consequently, modifying the
        returned vector never modifies either the setting or its penalty.
        """
        out = self.op.domain.empty()
        if hasattr(self, '_init'):
            self.op.domain.copyto(out, self._init)
        else:
            self.penalty.reference_vector(out=out)
        return out

    @init.setter
    def init(self, new_init: ReadOnlyVector) -> None:
        self.change_init(new_init=new_init)

    def change_init(self, new_init: ReadOnlyVector) -> None:
        r"""Store a supplied initial guess independently of the penalty."""
        if new_init not in self.op.domain:
            raise TypeError(Errors.not_in_vecsp(
                new_init,
                self.op.domain,
                vec_name='new_init',
                space_name='domain',
            ))
        if not hasattr(self, '_init'):
            self._init = self.op.domain.empty()
        if new_init is not self._init:
            self.op.domain.copyto(self._init,new_init)

    @init.deleter
    def init(self) -> None:
        r"""Discard the stored initial guess and use the penalty reference."""
        self.__dict__.pop('_init', None)

    @property
    def codomain_free(self)->bool:
        r"""If True, vectors in the codomain of the operator should be avoided. """
        return self._codomain_free

    @property
    def data_fid(self) -> Functional:
        r"""The data fidelity functional."""
        return self._data_fid 

    def set_data_fid(self,data_fid: Functional):
        if self.data is not None:
            raise RuntimeError(Errors.runtime_error("data fidelity term cannot be changed if data are set."))
        self._data_fid = as_functional(data_fid, self.op.codomain)
        self._init_method_dictionary()

    @property
    def data(self) -> ReadOnlyVector | ObsDataT | None:
        if not hasattr(self, '_data'):
            return None
        if self._data in self.op.codomain:
            return self.op.codomain.read_only(self._data)
        return self._data
    
    @data.setter
    def data(self, new_data: ObsDataT) -> None:
        self.change_data(new_data=new_data)

    def change_data(self, new_data: ObsDataT) -> None:
        if (new_data is None):
            raise ValueError(Errors.value_error(f"Overwriting data with {None} is not allowed."))
        if not (hasattr(self,"_data") and new_data is self._data):
            if not self.codomain_free:
                if (self.data_fid.is_data_func):
                    self.log.warning("Existing data in data fidelity functional is overwritten.")
                self._data_fid=self.data_fid.as_data_func(new_data)
                self._data = self.data_fid.data
            else:
                self._data = deepcopy(new_data)

    @data.deleter
    def data(self) -> None:
        del self.data_fid.data
        del self._data

    @property
    def exact_data(self)->Vector:
        return self._exact_data
    
    @exact_data.setter
    def exact_data(self, new_exact_data:Vector) -> None:
        self._exact_data = new_exact_data
        del self.data

    @property
    def h_domain(self)->HilbertSpace:
        r"""The Hilbert space associated to penalty functional"""
        return self.penalty.h_domain

    @property   
    def h_codomain(self)->HilbertSpace:
        r"""The Hilbert space associated to data fidelity functional"""
        return self.data_fid.h_domain if not isinstance(self.data_fid,Composed) else self.data_fid.func.h_domain

    @property
    def is_tikhonov(self) -> bool:
        """True if a regularization parameter is set"""
        return self.regpar is not None
    @property
    def is_convex(self) -> bool:
        """True if the penalty and the data fidelity functional are both convex and the operator is linear."""
        return self.op.linear and self.penalty.is_convex and (self.codomain_free or self.data_fid.is_convex)
    @property
    def is_hilbert(self) -> bool:
        """True if penalty and data fidelity are quadratic with Hessian that are positive multiples of the corresponding Gram matrices."""
        return (
            self.penalty.is_norm_squared_type
            and (self.codomain_free or self.data_fid.is_norm_squared_type)
        )

    @property
    def regpar(self) -> float:
        r"""The regularization parameter"""
        return self._regpar

    @regpar.setter
    def regpar(self,new_regpar: float):
        if(new_regpar is not None):
            if not isinstance(new_regpar,(float,int)):
                raise TypeError(Errors.type_error("The regularization parameter need to be a scalar"))
            if new_regpar <= 0:
                raise ValueError(Errors.value_error("The regularization parameter need to be a positive scalar"))
            new_regpar = float(new_regpar)
        self._regpar=new_regpar
        self._init_method_dictionary()

    @regpar.deleter
    def regpar(self) -> None:
        r"""Remove the regularization parameter.

        This turns a primal Tikhonov setting into the corresponding
        non-Tikhonov setting and refreshes its solver dictionary.  Internally,
        a missing regularization parameter is represented by ``None`` so that
        :attr:`is_tikhonov` and later assignments to :attr:`regpar` continue
        to work normally.

        A regularization parameter cannot be removed from a dual setting,
        because such settings are defined only together with their associated
        primal Tikhonov setting.
        """
        if self._primal_setting is not None:
            raise RuntimeError(Errors.runtime_error(
                "The regularization parameter cannot be removed from a dual "
                "setting."
            ))
        self._regpar = None
        self._init_method_dictionary()

    def _init_method_dictionary(self) -> None:
        """Initialize the solver dictionary for one of the three supported setting kinds."""
        if not hasattr(self, '_regpar') or not hasattr(self, '_primal_setting'):
            return
        if self._primal_setting is not None:
            self.__dict__.pop('_methods', None)
            return
        if not self.is_tikhonov:
            self._methods = (
                get_linear_reg_method_dictionary()
                if self.op.linear
                else get_nonlinear_reg_method_dictionary()
            )
            self.__dict__.pop('duality_gap_warning_incr', None)
        elif self.is_convex:
            self._methods = get_convex_solver_dictionary()
            self.duality_gap_warning_incr = 1.
        else: # Non-convex Tikhonov settings
            self._methods = get_non_convex_solver_dictionary()

    ######General convenience methods
    @property
    def noise_model_fixed(self)->bool:
        r""" Flag indicating if a noise model has been fixed. 
        """
        return self._noise_model_param != {}

    def set_Gaussian_white_noise_model(self,relative_noise_level:float=None,absolute_noise_level:float=None,white_noise=True)->None:
        self._noise_model_param = {'model_name':'Gaussian',
                                   'relative_noise_level':relative_noise_level,
                                   'absolute_noise_level':absolute_noise_level,
                                   'white_noise':white_noise
                                   }

    def set_Poisson_noise_model(self,expected_nr_counts=None)->None:
        self._noise_model_param = {'model_name':'Poisson',
                                   'expected_nr_counts':expected_nr_counts}

    def generate_synthetic_data(self)->None:
        r""" Generates synthetic noisy data based on exact_data and some noise model that has to be fixed in advance.
        """
        model = self._noise_model_param
        if not 'model_name' in model:
            raise RuntimeError(f"No noise model has been set, yet. {self._noise_model_param}")
        elif model['model_name'] not in ['Gaussian','Poisson']:
            raise RuntimeError(f"Unknown noise model: {model['model_name']}")

        if model['model_name'] == 'Gaussian':
            self.add_Gaussian_noise(relative_noise_level=model['relative_noise_level'],
                                    absolute_noise_level=model['absolute_noise_level'],
                                    white_noise=model['white_noise'])
        elif model['model_name'] == 'Poisson':
            self.generate_Poisson_data(expected_nr_counts=model['expected_nr_counts'])

    def add_Gaussian_noise(self,relative_noise_level:float=None, absolute_noise_level:float=None,white_noise=True,seed=None) -> None:
        r""" generates Gaussian noise using self.codomain.randn, adds it to exact_data (which must have been provided at initialization of the setting), and sets data to the sum.
        
        Parameters:
        -----------
        relative_noise_level: float | None: optional
        absolute_noise_level: float | None: optional
            Exactly one of these two parameters must be given as a positive float. 
        white_noise: bool: optional
            Defaults to True.
            If true, the noise is generated as white noise with respect to the codomain Hilbert space structure using a Cholesky-type factorization of the gram matrix (which needs to be implemented for the codomain space!); otherwise it is generated by self.codomain.randn.
            If true, relative_noise_level and absolute_noise_level refer to the Hilbert space norm of the covariance operator, and the noise process is asymptotically discretization invariant; otherwise they refer to the norm in the codomain vector space.
        seed: int | None: optional
            seed for random number generator for reproducibility
        """
        if relative_noise_level is None and absolute_noise_level is None:
            raise ValueError(Errors.value_error("Either relative or absolute noise level must be given."))
        if relative_noise_level is not None and absolute_noise_level is not None:
            raise ValueError(Errors.value_error("Your cannot provide both relative or absolute noise level!"))
        if relative_noise_level is not None and not isinstance(relative_noise_level,float) and not relative_noise_level>0:
            raise ValueError(Errors.value_error("ralative_noise_level must be a positive float.")) 
        if absolute_noise_level is not None and not isinstance(absolute_noise_level,float) and not absolute_noise_level>0:
            raise ValueError(Errors.value_error("absolute_noise_level must be a positive float.")) 
        if not isinstance(white_noise,bool):
            raise TypeError(Errors.type_error("white_noise must be a boolean."))
        if not hasattr(self,'_exact_data'):
            raise RuntimeError(Errors.runtime_error('No exact data has been provided at initialization of the setting.'))
        if self.data is not None:
            self.log.warning("Overwriting given data!")
        if white_noise:
            try:
                chol = self.h_codomain.cholesky
            except NotImplementedError:
                raise NotImplementedError("Cholesky factorization is not implemented for the codomain Hilbert space.")
            noise = chol.domain.randn(seed=seed)
            if relative_noise_level is not None:
                noise *= relative_noise_level * self.h_codomain.norm(self._exact_data)
            else:
                noise *= absolute_noise_level
            noise = chol.inverse.adjoint(noise) 
            self._noise_model_param = {'model_name':'Gaussian',
                        'absolute_noise_level':absolute_noise_level
                        if absolute_noise_level is not None else   
                        relative_noise_level * self.h_codomain.norm(self._exact_data)
            }                                   
        else:
            noise = self.op.codomain.randn(seed=seed)
            if relative_noise_level is not None:
                noise *= relative_noise_level*self.h_codomain.norm(self._exact_data)/self.h_codomain.norm(noise)
            else:
                noise *= absolute_noise_level/self.h_codomain.norm(noise)
            self._noise_model_param = {'model_name':'deterministic',
                        'absolute_noise_level':absolute_noise_level
                        if absolute_noise_level is not None else   
                        relative_noise_level * self.h_codomain.norm(self._exact_data)
            }                       

        self.data = self._exact_data + noise 

    def generate_Poisson_data(self, expected_nr_counts=None, seed=None):
        r"""Generate and store Poisson-distributed data.

        Sampling is delegated to :meth:`self.op.codomain.poisson`.
        
        Parameters:
        -----------
        expected_nr_counts: int | None: optional
            Expected total number of counts for the Poisson distribution.
            The Poisson data are rescaled such that the expectation of the data is the exact data. This means that the resulting data are not integer-valued. 
            If None, no scaling is applied, and data are integer-valued.
        seed: int | None: optional
            random seed for reproducibility
        """
        if not hasattr(self,'_exact_data'):
            raise RuntimeError(Errors.runtime_error('No exact data has been provided at initialization of the setting.'))
        if not np.issubdtype(self.op.codomain.dtype, np.floating) and np.all(self._exact_data >=0):
            raise RuntimeError(Errors.runtime_error('Poisson data can only be generated for non-negative real-valued data spaces.'))
        if self.data is not None:
            self.log.warning("Overwriting given data!")
        if expected_nr_counts is not None:
            scale = expected_nr_counts / np.sum(np.array(self._exact_data))
        else:
            scale = 1.0      
        poisson_data = self.op.codomain.poisson(scale*self._exact_data,seed=seed)/scale
        self.data = poisson_data    

    def get_or_update_data(self, 
                           new_data:Vector | ObsDataT | None = None,
                           update:bool = True
                           ) -> Vector | ObsDataT:
        r""" Updates the data stored in the setting or gets these data of they are not provided.

        Parameters
        ----------
        new_data : Vector or ObsDataT or None [default:None]
            New data to be passed to the data functional self.data_fid via the data setter.
        update: bool [default:True]
            If True, the data stored in the setting is updated to new_data. 
        """
        if new_data is not None:
            if update:
                if self.data is not new_data:
                    self.log.warning("Overwriting existing data in setting!")
                    self.data = new_data
            return new_data
        else:
            toret = self.data
            if toret is None:
                raise RuntimeError(Errors.runtime_error("No data has been provided either explicitly or as an argument of the method or in the setting!"))
            return toret

    def get_or_update_initial_guess(self, 
                                    new_init:ReadOnlyVector | None = None,
                                    update:bool = False,
                                    out:Vector | None = None
                                    ) -> Vector:
        r"""Update or retrieve the initial guess as a writable copy.

        If neither ``new_init`` nor a stored initial guess is available, the
        penalty functional's reference vector is used.

        Parameters
        ----------
        new_init : ReadOnlyVector or None [default: None]
            New initial guess in op.domain
        update: bool [default: False]
            If True, the initial guess in the setting is updated if ``new_init`` is not None. 
            Otherwise, the initial guess is only copied to ``out``.
        out: Vector or None [default: None]

        Returns
        --------
        Vector:
            new initial guess (as writable copy)
        """
        out = self.op.domain.empty() if out is None else out
        if out is not None and not out in self.op.domain:
            raise TypeError(Errors.not_in_vecsp(out,self.op.domain,vec_name="out",space_name="domain"))
        if new_init is not None:
            if not new_init in self.op.domain:
                raise TypeError(Errors.not_in_vecsp(new_init,self.op.domain,vec_name="new_init",space_name="domain"))
            source = new_init
            if update:
                if hasattr(self, '_init') and self._init is not new_init:
                    self.log.warning("Overwriting existing initial guess in setting!")
                self.init = new_init
                source = self.init
            if out is not source:
                self.op.domain.copyto(out, source)
        else: # new_init is None
            stored_init = self.init
            if out is not stored_init:
                self.op.domain.copyto(out, stored_init)
        return out

    ######Methods exploiting duality
    def get_dual_setting(self) -> Setting:
        r"""Yields the setting of the dual optimization problem

        .. math::
           \mathcal{R}^\ast(T^\ast p) + \frac{1}{\alpha}\mathcal{S}^\ast(- \alpha p) = \min!

        """
        if(not self.is_tikhonov):
            raise RuntimeError(Errors.generic_message("Incomplete setting: A regularization parameter is required for the computation of a dual setting."))
        if(not self.is_convex):
            raise RuntimeError(Errors.generic_message("The setting has to be convex for the computation of a dual setting."))

        dual_op = self.op.adjoint
        return Setting(
            dual_op,
            self.data_fid.conj.dilation(-self.regpar),
            self.penalty.conj,
            regpar= 1/self.regpar,
            # The reference of S*(-alpha p) is part of the transformed
            # objective, but is generally a poor starting point: for a
            # shifted quadratic data term it is of order data/alpha.  Keep
            # the functional unchanged and store p=0 independently as the
            # initial iterate.  It maps back to the primal penalty reference.
            init=dual_op.domain.zeros(),
            incorporate_init_in_penalty=False,
            _primal_setting = self,
            logging_level=self.log.level
        )

    def dual_to_primal(self,
                       dual: tuple[Vector|None,Vector|None], 
                       own: bool = False
                       ) -> Vector:
        r""" Returns an element of :math:`\partial \mathcal{R}^*(T^*p)` 
        If :math:`p` is a solution to the dual problem and :math:`\partial\mathcal{R}^*` is a singleton, this yields a solution to the primal problem. 
                
        Parameters
        ----------
        dual: tuple of self.op.adjoint.domain and self.op.adjoint.codomain
            tuple of dual variable p and T*p. Either p or T*p must not be None. If T*p is None, it will be 
            computed. Otherwise, p will not be used.
        own: bool [default: False]
            Only relevant for dual settings. If False, the duality relations of the primal setting are used. 
            If true, the duality relations of the dual setting are used. 

        Returns
        -------
        Vector
            An element of the primal space corresponding to the given dual variable.
        """
        if(not self.is_tikhonov):
            raise RuntimeError(Errors.generic_message("Incomplete setting: A regularization parameter is required for the computation of a dual primal mapping."))
        if(not self.is_convex):
            raise RuntimeError(Errors.generic_message("The setting has to be convex for the computation of a dual primal mapping."))
        if not isinstance(dual,tuple) or len(dual)!=2:
            raise TypeError(Errors.type_error("dual must be a tuple of (p,T*p)"))
        if dual[0] is None and dual[1] is None:
            raise ValueError(Errors.value_error("Either p or T*p must be given in dual tuple!"))
        if dual[1] is None:
            dual[1] = self.op.adjoint(dual[0])
        if not dual[1] in self.op.adjoint.codomain:
            raise TypeError(Errors.type_error("T*p not in codomain of adjoint operator!"))
        if self._primal_setting is None or own == True:
            return self.penalty.conj.subgradient(dual[1])
        else:
            return self._primal_setting.primal_to_dual((None,-self.regpar*dual[1]))
            """Note that the dual variables of the dual problem differ by a factor -alpha_d from the primal variables of the primal problem.
            Here alpha_d=1/alpha_p is the regularization parameter of the dual problem, and alpha_p the regularization parameter of the primal problem.
            """
        
    def primal_to_dual(self,
                       primal: tuple[Vector|None,Vector|None], 
                       own: bool =False
                       ) -> Vector:
        r"""
        Returns an element of :math:`(-1/\alpha) \partial \mathcal{S}(Tf)` 
        If :math:`f` is a solution to the primal problem and :math:`\partial \mathcal{S}` is a singleton, this 
        yields a solution to the dual problem. 
    
        Parameters
        ----------------------------
        primal: tuple[Vector|None, Vector|None]
            tuple of primal variable f and Tf. Either f or Tf must not be None. If Tf is None, it will be 
            computed. Otherwise, f will not be used.
        own: bool [default: False]
            Only relevant for dual settings. If False, the duality relations of the primal setting are used. 
            If true, the duality relations of the dual setting are used. 

        Returns
        -------
        Vector
            An element of the dual space corresponding to the given primal variable.
        """
        if(not self.is_tikhonov):
            raise RuntimeError(Errors.generic_message("Incomplete setting: A regularization parameter is required for the computation of a primal dual mapping."))
        if(not self.is_convex):
            raise RuntimeError(Errors.generic_message("The setting has to be convex for the computation of a primal dual mapping."))
        if not isinstance(primal,tuple) or len(primal)!=2:
            raise TypeError(Errors.type_error("primal must be a tuple of (f,Tf)"))
        if primal[0] is None and primal[1] is None:
            raise ValueError(Errors.value_error("Either f or Tf must be given in primal tuple!"))
        if primal[1] is None:
            primal[1] = self.op(primal[0])
        if not primal[1] in self.op.codomain:
            raise TypeError(Errors.type_error("Tf not in codomain of operator!"))
        if self._primal_setting is None or own==True:
            return (-1./self.regpar) * self.data_fid.subgradient(primal[1])
        else:
            return self._primal_setting.dual_to_primal(primal)

    def _complete_primal_dual_tuples(self,
                                     primal: tuple[Vector,Vector|None] | None =None,
                                     dual: tuple[Vector,Vector|None] | None =None
                                     ) -> tuple[tuple[Vector,Vector], tuple[Vector,Vector]]:  
        r"""Completes either the primal or dual tuple by computing the missing operator application.
        If one of the tuples is None, it is computed using the primal_to_dual or dual_to_primal methods.

        Parameters
        ----------
        primal: tuple of setting.op.domain and setting.op.codomain [default: None]
            tuple of primal variable f and Tf. If Tf is None, it will be computed.
        dual: tuple of setting.op.adjoint.domain and setting.op.adjoint.codomain [default: None]
            tuple of dual variable p and T*p. If T*p is None, it will be computed. 

        Returns
        -------
        tuple of tuples of Vectors
            Completed primal and dual tuples, each containing the variable and its operator application.
        """
        if primal is None and dual is None:
            raise ValueError(Errors.value_error("Either a primal or dual tuple need to be given to complete both!"))
        if primal is None:
            if dual[1] is None:
                p = dual[0]
                Tsp = self.op.adjoint(p)
                dual = (p,Tsp)
            f = self.dual_to_primal(dual)
            Tf = self.op(f)
            primal = (f,Tf)
        else:
            f= primal[0]
            if not f in self.op.domain:
                raise TypeError(Errors.type_error("f not in domain of operator!"))
            Tf = self.op(f) if primal[1] is None else primal[1]
            primal = (f,Tf)
        if dual is None:
            p = self.primal_to_dual(primal)
            Tsp = self.op.adjoint(p)
            dual = (p,Tsp)
        else:
            p = dual[0]
            if not p in self.op.adjoint.domain:
                raise TypeError(Errors.type_error("p not in domain of adjoint operator!"))
            Tsp = self.op.adjoint(p) if dual[1]is None else dual[1]
            dual = (p,Tsp)
        return primal,dual

    def duality_gap(self, 
                    primal: tuple[Vector|None,Vector|None] | None =None, 
                    dual: tuple[Vector|None,Vector|None] | None  =None
                    ) -> float:
        r"""Computes the value of the duality gap. 
    
        
        .. math::
            \frac{1}{\alpha}\mathcal{S}_{g^{\delta}}(Tf) + \mathcal{R}(f) - \frac{1}{\alpha} }\mathcal{S}_{g^{\delta}}^\ast(-\alpha p) - \mathcal{R}^\ast(T^\ast p)

        Parameters
        ----------
        primal: tuple[Vector,Vector|None] | None [default: None]
            tuple of primal variable :math:`f` and :math:`Tf`. If :math:`Tf` is None, it will be computed.
        dual: tuple[Vector|None,Vector|None] | None [default: None]
            tuple of dual variable :math:`p` and :math:`T*p`. If :math:`T*p` is None, it will be computed.        
        """
        if(not self.is_tikhonov):
            raise RuntimeError(Errors.generic_message("Incomplete setting: A regularization parameter is required for the computation of the duality gap."))
        if not self.is_convex:
            raise RuntimeError(Errors.not_linear_op(self.op,add_info="The duality gap can only be computed for convex settings with linear operators!"))

        if self._primal_setting is not None and primal is not None:
            # A primal variable of the transformed dual setting is precisely
            # a dual variable of the original setting.  Evaluate the gap
            # there directly.  This is algebraically equivalent to computing
            # the transformed gap and multiplying it by
            # self.regpar**2, but avoids magnifying cancellation errors by
            # alpha_primal**(-2).
            return self._primal_setting.duality_gap(dual=primal)

        (f,Tf),(p,Tsp) = self._complete_primal_dual_tuples(primal,dual)
        alpha = self.regpar

        dat = 1./alpha * self.data_fid(Tf)
        pen = self.penalty(f)
        ddat = self.penalty.conj(Tsp)
        dpen = 1./alpha * self.data_fid.conj(-alpha*p)
        ares = ma.fabs(dat)+ma.fabs(pen)+ma.fabs(ddat)+ma.fabs(dpen) 
        if not ma.isfinite(ares):
            self.log.warning('duality gap infinite: R(..)={:.3e}, S(..)={:.3e}, S*(..)={:.3e}, R*(..)={:.3e}'.format(pen,dat,dpen,ddat))
            return ma.inf
        res = dat+pen+ddat+dpen
        if ares/res>1e10:
            self.log.warning('estimated loss of rel. accuracy in duality gap by cancellation: {:.3e}'.format(ares/res))
        elif ares/res>1e5*self.duality_gap_warning_incr:
            self.log.debug('estimated loss of rel. accuracy in duality gap by cancellation: {:.3e}'.format(ares/res))
            self.duality_gap_warning_incr *=10.

        return (res
                if self._primal_setting is None else
                res * alpha 
                # For dual settings alpha=alpha_dual = 1/alpha_primal.
                # Since the dual objective functional is multiplied by alpha_primal to match the 
                # form of the primal objective functional, a factor 1/alpha_primal appears. 
        ) 
    
    def violation_optimality_cond(self, 
                                  primal: tuple[Vector|None,Vector|None] | None =None,
                                  dual: tuple[Vector|None,Vector|None] | None =None,
                                  deriv: Operator | None = None,
                                  ) -> tuple[float, float] | float:
        r"""Returns the degree to which an iterate violates the optimality conditions for a minimum of the Tikhonov functional :math:`\frac{1}{\alpha}\mathcal{S}(f) + \mathcal{R}(f)`. 

        For convex Tikhonov functionals (self.is_convex): 
        A pair :math:`(f,p)` of a primal point :math:`f` and a dual point :math:`p\ 
        is a saddle point of 
        :math:`-<Tf,p> + \mathcal{R}(f)-\frac{1}{\alpha}\mathcal{S}^*(-\alpha p) `
        (and hence :math:`f` a primal and :math:`p` a dual solution) iff

        .. math::
            Tf \in \partial \mathcal{S}^\ast(-\alpha p), \qquad T^\ast p \in \partial \mathcal{R}(f).

        The violations are returned as a pair of distances of the left-hand sides to the respective 
        subdifferentials. 

        For non-convex Tikhonov functionals (not self.is_convex):
        The violation of the optimality condition

        .. math::
            \frac{-1}{\alpha} F'[f]^*\mathcal{S}_{g^{\delta}}'(F(f)) \in \partial \mathcal{R}(f)
 
        is returned as single distance to the subdifferential. 

        Parameters
        ---------------------------
        primal: tuple[Vector|None,Vector|None] | None [default: None]
            tuple of primal variable :math:`f` and :math:`Tf`. If :math:`Tf` is None, it will be computed.
        dual: tuple[Vector|None,Vector|None] | None [default: None]
            tuple of dual variable :math:`p` and :math:`T*p`. If :math:`T*p` is None, it will be computed. 
        If one of the tuples is None, it is computed using the primal_to_dual or dual_to_primal methods.

        Returns
        -------
        tuple of floats or float
            Distance(s) to the subdifferential(s) in the optimality condition(s).
        """
        if(not self.is_tikhonov):
            raise RuntimeError(Errors.generic_message("Incomplete setting: A regularization parameter is required for this check."))
        if not "dist_subdiff" in self.penalty.methods:
            raise RuntimeError(Errors.generic_message("Need dist_subdiff method of penalty functional."))
        if self.is_convex:
            if deriv is not None:
                raise ValueError(Errors.value_error(
                    "deriv is only used for nonconvex settings.", deriv
                ))
            if not "dist_subdiff" in self.data_fid.conj.methods:
                raise RuntimeError(Errors.generic_message("Need dist_subdiff method of conjugate data fidelity functional."))                            
            (f,Tf),(p,Tsp) = self._complete_primal_dual_tuples(primal,dual)

            alpha = self.regpar
            return (1./alpha)*self.data_fid.conj.dist_subdiff(Tf,(-alpha)*p), \
               self.penalty.dist_subdiff(Tsp,f) 
        else:
            if not "subgradient" in self.data_fid.methods:
                raise RuntimeError(Errors.generic_message("Need subgradient method of data fidelity functional."))
            if dual is not None:
                raise ValueError(Errors.value_error(
                    "dual is only used for convex settings.", dual
                ))
            if not isinstance(primal, tuple) or len(primal) != 2:
                raise ValueError(Errors.value_error(
                    "primal must be a tuple of length two.", primal
                ))
            f, Tf = primal
            if not f in self.op.domain:
                raise TypeError(Errors.type_error(
                    "primal[0] must be in operator domain.", f
                ))
            if Tf is not None and not Tf in self.op.codomain:
                raise TypeError(Errors.type_error(
                    "primal[1] must be None or in operator codomain.", Tf
                ))
            if deriv is not None and not (
                    isinstance(deriv, Operator) and deriv.linear):
                raise TypeError(Errors.type_error(
                    "deriv must be a linear operator.", deriv
                ))
            if deriv is None:
                evaluated, deriv = self.op.linearize(f)
                if Tf is None:
                    Tf = evaluated
            elif Tf is None:
                Tf = self.op(f)
            p = self.data_fid.subgradient(Tf)
            Fstar_p = deriv.adjoint(p)
            Fstar_p *= (-1/self.regpar)
            return self.penalty.dist_subdiff(Fstar_p,f)

    def add_method(self,
                   method_abbrv:str,
                   class_name:type,
                   primal:bool|None=None,
                   full_method_name:str=None
                   )->None:
        """ Add a new method to the method dictionary
        
        Parameters:
        ----------------
        method_abbrv:str
            Abbreviation of the new method by which it is called.
        class_name:type
            Class of the new method
        primal:bool or None
            For convex Tikhonov settings, flag if the method is applied to the primal
            or dual problem; None defaults to True. It is not used for settings
            without a regularization parameter.
        full_method_name:str | None, optional
            Full, more descriptive name of the method. Defaults to None 
        """
        if not self.is_tikhonov and primal is False:
            raise ValueError(Errors.value_error(
                'A setting without a regularization parameter has no dual '
                'setting to which a method could be applied.'
            ))
        if not callable(getattr(class_name, 'check_applicability', None)):
            raise TypeError(Errors.type_error(
                'Solver classes in a method dictionary must implement '
                'check_applicability.', class_name
            ))
        if not hasattr(self,'_methods'):
            self._init_method_dictionary()
        method = {'class': class_name, 'full': full_method_name}
        if self.is_tikhonov:
            method['primal'] = True if primal is None else primal
        self._methods[method_abbrv] = method

    def evaluate_methods(self, method_names: list[str] | None = None)->None:
        """Evaluates which methods are applicable to the current setting. 
        This is achieved by calling method.check_applicability(self), which also provide information on guaranteed rates.

        Parameters
        ----------
        method_names: list of strings or None [default:None]
            List of names of methods to be evaluated. If None, all methods are evaluated.   
        """

        if method_names is None:
            method_names = list(self._methods)
        else:
            method_names = list(method_names)
            for method_name in method_names:
                if not method_name in self._methods:
                    raise ValueError(f'Unknown method name {method_name}. Known methods are {self._methods.keys()}.')
        # Only convex Tikhonov solvers use the global operator norm for their
        # convergence-rate estimates. It may not exist for nonlinear methods.
        op_norm = self.op.norm() if method_names and self.is_tikhonov and self.is_convex else None
        for method_name in method_names:
            method = self._methods[method_name]
            if self.is_tikhonov:
                primal = method['primal']
                target_setting = self if primal else self.get_dual_setting()
            else:
                target_setting = self
            out,_ = (method['class'].check_applicability(target_setting, op_norm=op_norm)
                     if op_norm is not None else
                     method['class'].check_applicability(target_setting)
                    )
            method.pop('rate', None)
            if (self.is_tikhonov and not primal
                    and not 'subgradient' in self.penalty.conj.methods):
                method['info'] = ('' if out['applicable'] else out['info']) + 'Missing subgradient of conjugate penalty.'
                method['applicable'] = False
            else:
                method['applicable'] = out['applicable']
                method['info'] = out['info']
                if (self.is_tikhonov and out['applicable']
                        and 'rate' in out):
                    method['rate'] = out['rate']

    def applicable_methods(self)->dict[str,dict]:
        """Return the methods that can be applied to this setting."""
        if any('applicable' not in self._methods[name] for name in self._methods.keys()):
            self.evaluate_methods()
        return {name:method for name, method in self._methods.items() if method['applicable']}
        
    def display_all_methods(self, full_names:bool=True)->None:
        """Display all registered methods and their applicability."""
        self.evaluate_methods()
        print('Applicable methods: (default =',self.get_default_method(),')\n')
        for name,method in self.applicable_methods().items():
            full_name = method.get('full')
            suffix = (
                'linear rate: {:.3e}'.format(method['rate'])
                if 'rate' in method else ''
            )
            print(name, (f' ({full_name}): ' if full_names and full_name else ''),
                  method['info'], suffix)
        print('\n Non-applicable methods:\n')
        for name,method in self._methods.items(): 
            if method['applicable']==False:
                full_name = method.get('full')
                print(name, (f' ({full_name}): ' if full_names and full_name else ''),
                      method['info'])

    def set_method_parameters(self,
                              method_name:str,
                              params:dict|None=None)->None:
        if method_name not in self._methods:
            raise RuntimeError(Errors.runtime_error(f"Unknown method name {method_name}. Known methods are {self._methods}. Use add_method() to add a new method."))    
        self._methods[method_name]['parameters'] = (
            {} if params is None else dict(params)
        )

    def select_best_method(self) -> str:
        """Returns the name of the applicable method with the best convergence rate predicted by theory 
        and the convexity and Lipschitz parameters of the data and penalty functional.
        (Since comparisons of first and second order methods are difficult, we only choose among first 
        order methods, and to achieve this, we set convergence rates of second order method >1.)
        """
        if not self.is_tikhonov:
            raise RuntimeError(Errors.runtime_error(
                'Convergence-rate estimates are not available for '
                'non-Tikhonov regularization methods. Please specify '
                'method_name explicitly.'
            ))
        d = self.applicable_methods()
        rated = {name: method for name, method in d.items() if 'rate' in method}
        if not rated:
            raise RuntimeError(Errors.runtime_error(
                'No convergence-rate comparison is available for the '
                'applicable methods. Please specify method_name explicitly.'
            ))
        best_method_name = min(
            rated, key=lambda name: np.abs(rated[name]['rate'])
        )
        self.log.info('Choose '+best_method_name+' as best method.')
        return best_method_name

    def set_stoprule(self,
                          rule:StopRule,
                          method_names:list[str]|None=None) -> None:
        """Sets a StopRule for an optimization method.
        
        Parameters
        ----------
        rule: StopRule
            the stoprule
        method_names: list[str] or None [default: None]
            keys of the methods. In the default case the StopRule is attached to all applicable solvers. 
        """
        if not isinstance(rule,StopRule):
            raise TypeError(f"rule must be of class StopRule. Got{rule}.")
        if method_names is None:
            method_names = list(self.applicable_methods())
        if (not isinstance(method_names,list)
                or not all(isinstance(method_name,str) for method_name in method_names)):
            raise ValueError(Errors.value_error('method_names must be list of strings',method_names))
        for method_name in method_names:
            if method_name not in self._methods.keys():
                raise ValueError(f"{method_name} is unknown method key.")
            # Keep the configured rule separate from the rule used by the
            # latest run.  The latter accumulates histories and may contain
            # run-local additions such as an iteration limit or ErrorMonitor.
            configured_rule = rule.copy_and_reset()
            self._methods[method_name]['configured_stoprule'] = configured_rule
            self._methods[method_name]['stoprule'] = configured_rule.copy_and_reset()

    def get_stopping_rule(self, method_name: str) -> StopRule:
        """Retrieves a stoprule that has run an optimization method 
        (e.g. to view statistics or (intermediate) solutions)
        
        Parameters
        ----------
        method_name: str
            Key of the method
        
        Returns
        -------
        StopRule
        """
        if not method_name in self._methods.keys():
            raise ValueError(f"{method_name} is unknown method key.")
        if 'stoprule' not in self._methods[method_name]:
            raise RuntimeError(f'Method {method_name} has no StopRule.')
        else:
            return self._methods[method_name]['stoprule']   

    def get_default_method(self):
        if not self.is_tikhonov:
            return (self._default_linear_regmethod
                            if self.op.linear else
                            self._default_nonlinear_regmethod
            )
        else:
            if self._default_inner_solver is None and self.is_convex: 
                self._default_inner_solver = self.select_best_method()
            return (
                self._default_inner_solver
                if self.is_convex else
                self._default_nl_minimizer
            )

    def run(self,
            method_name: str|None = None,
            max_it: int | None = None,
            stoprule: StopRule|object = _DEFAULT_STOPRULE,
            ref_sol: (ReadOnlyVector
                        | tuple[ReadOnlyVector | None, ReadOnlyVector]
                        | None) = None,
            dist_x: Callable[[tuple[ReadOnlyVector, ReadOnlyVector]], float] | None = None,
            dist_y: Callable[[tuple[ReadOnlyVector, ReadOnlyVector]], float] | None = None,
            **kwargs) -> tuple[Vector, Vector]:
        """Runs a given method for the setting. If no method name is given, the "best" method is selected by select_best_method() if possible.
        
        Parameters
        ----------
        method_name: str or None [default: None] 
            Key of the method to be run in the methods dictionary self._methods (can be displayed by display_all_methods())
            If None the "best" method is selected by select_best_method().
        max_it: int [default: 1000]
            maximal number of iterations
        stoprule: StopRule | None
            Stopping rule. An explicit value overrides a rule configured by
            :meth:`set_stoprule` for this run only. By default, a configured
            rule is used if present; otherwise convex Tikhonov methods use a
            duality-gap rule combined with 1000 maximum iterations and other
            methods use 1000 maximum iterations. If None, a previously stored
            stopping rule is required and used.
        ref_sol: ReadOnlyVector or tuple[ReadOnlyVector | None, ReadOnlyVector], optional
            Reference solution used to monitor iteration errors. A single
            vector monitors ``x``. A pair ``(x, y)`` monitors both values; use
            ``(None, y)`` to monitor only ``y``. The monitor never stops the
            iteration.
        dist_x, dist_y: callable, optional
            Custom distances called as ``dist((current, reference))`` by the
            error monitor. If omitted, the corresponding Hilbert-space norm is
            used when available.
        logging_level: int or str, optional
            Logging level for the solver and the stopping-rule tree used by
            this run, including iteration limits and error monitors. Defaults
            to ``self.log.level``. An explicit value takes precedence over
            registered method parameters and configured rule levels.
        **kwargs: dict
            Arguments to be passed to the method.

        Returns
        -------
            x,y: x is the minimizer of the appproximate solution and y its value under the operator.         
        """
        if method_name is None:
            method_name = self.get_default_method()
        if not method_name in self._methods:
            raise ValueError(Errors.value_error(f"{method_name} not in list of methods: {self._methods}."))
        themethod= self._methods[method_name]
        if not 'applicable' in themethod:
            self.evaluate_methods([method_name]) 
        if themethod['applicable'] == False:
            raise RuntimeError(f'{method_name} is not applicable in this setting: '+themethod['info'])
        logging_level = kwargs.get('logging_level', self.log.level)
        if 'parameters' in themethod:
            kwargs.update(themethod['parameters'])
        kwargs['logging_level'] = logging_level

        if self.is_tikhonov:
            primal = themethod['primal']
            thesetting = self if primal else self.get_dual_setting()
        else:
            thesetting = self
        if max_it is not None:
            if not isinstance(max_it, (int, np.integer)):
                raise TypeError(Errors.type_error(
                    'max_it must be an integer or None.', max_it
                ))
            if max_it < 0:
                raise ValueError(Errors.value_error(
                    'max_it must be nonnegative.', max_it
                ))
            max_it = int(max_it)

        configured_stoprule = themethod.get('configured_stoprule')
        if stoprule is _DEFAULT_STOPRULE and configured_stoprule is not None:
            stoprule = configured_stoprule.copy_and_reset()
        elif stoprule is None:
            stored_stoprule = configured_stoprule or themethod.get('stoprule')
            if stored_stoprule is None:
                raise RuntimeError(
                    'No StopRule given either as argument of run or set '
                    'explicitly by set_stoprule.'
                )
            stoprule = stored_stoprule.copy_and_reset()
        elif stoprule is _DEFAULT_STOPRULE:
            default_max_it = 1000 if max_it is None else max_it
            iteration_limit = CountIterations(
                default_max_it, timing=True, logging_level=logging_level
            )
            if self.is_tikhonov and self.is_convex:
                stoprule = (
                    DualityGapStopping(tol=1e-10, logging_level=logging_level)
                    + iteration_limit
                )
            else:
                stoprule = iteration_limit
            # The requested limit is already part of the new default rule.
            max_it = None
        elif not isinstance(stoprule, StopRule):
            raise TypeError(Errors.type_error(
                'stoprule must be a StopRule or None.', stoprule
            ))

        if max_it is not None:
            stoprule = stoprule + CountIterations(
                max_it, timing=True, logging_level=logging_level
            )
        if ref_sol is not None:
            monitor_ref_sol = ref_sol
            monitor_dist_x = dist_x
            monitor_dist_y = dist_y

            if self.is_tikhonov and not primal:
                # The solver's x and y are (p, T*p), whereas Setting.run()
                # returns the corresponding primal pair (f, Tf).  Convert
                # both the reference and the monitored iterates so that all
                # methods report errors in the same primal variables.
                if isinstance(ref_sol, tuple):
                    ref_x, ref_y = ref_sol
                else:
                    ref_x, ref_y = ref_sol, None

                ref_Tx = self.op(ref_x) if ref_y is None else ref_y
                ref_p = self.primal_to_dual((ref_x, ref_Tx))
                ref_Tsp = self.op.adjoint(ref_p)

                if ref_x is not None:
                    primal_dist_x = (
                        dist_x
                        if dist_x is not None
                        else ErrorMonitor._norm_distance(self.h_domain.norm)
                    )

                    def monitor_dist_x(pair):
                        current_p, _ = pair
                        current_Tsp = self.op.adjoint(current_p)
                        current_x = self.dual_to_primal(
                            (current_p, current_Tsp)
                        )
                        return primal_dist_x((current_x, ref_x))

                if ref_y is not None:
                    primal_dist_y = (
                        dist_y
                        if dist_y is not None
                        else ErrorMonitor._norm_distance(
                            self.h_codomain.norm
                        )
                    )

                    def monitor_dist_y(pair):
                        current_Tsp, _ = pair
                        current_x = self.penalty.conj.subgradient(
                            current_Tsp
                        )
                        current_y = self.op(current_x)
                        return primal_dist_y((current_y, ref_y))

                monitor_ref_sol = (
                    (ref_p if ref_x is not None else None, ref_Tsp)
                    if ref_y is not None
                    else ref_p
                )

            stoprule = stoprule + ErrorMonitor(
                monitor_ref_sol,
                dist_x=monitor_dist_x,
                dist_y=monitor_dist_y,
                logging_level=logging_level,
            )
        elif dist_x is not None or dist_y is not None:
            raise ValueError(Errors.value_error(
                "dist_x and dist_y require ref_sol."
            ))
        # Store the actual rule used by this run so that its histories remain
        # available through get_stopping_rule().  Do not call set_stoprule()
        # here: doing so would overwrite the independently configured rule.
        themethod['stoprule'] = stoprule.copy_and_reset()
        
        solver = themethod['class'](thesetting,**kwargs)
        # Configure the executed copy, including nested combined rules.
        # Instance logger overrides require the instance-local ClassLogger.
        pending_rules = [themethod['stoprule']]
        seen_rules = set()
        while pending_rules:
            rule = pending_rules.pop()
            if id(rule) in seen_rules:
                continue
            seen_rules.add(id(rule))
            rule.log = solver.log.getChild(type(rule).__name__)
            rule.log.setLevel(logging_level)
            pending_rules.extend(getattr(rule, 'rules', ()))
        x,y = solver.run(themethod['stoprule'])
        
        if self.is_tikhonov and not primal:
            x_star,y_star = x,y
            x = self.dual_to_primal((x_star,y_star))
            y = self.op(x)
        return x,y

def get_linear_reg_method_dictionary() -> dict:
    """Return regularization methods intended for linear operators.

    ``FISTA`` is available without a regularization parameter when the
    setting penalty is of squared-norm type or is a null functional. In that
    mode the penalty determines the domain Hilbert geometry but is not part of
    the minimized objective.
    """
    from .linear import (
        CGNE,
        FISTA,
        Landweber as linLandweber,
        Richardson,
        NuMethod,
        NuMethodSelfAdjoint,
        TikhonovAlphaGrid
    )
    from .nonlinear import GeneralizedTikhonov
    from .linear.spectral import Spectral, Spectral_SelfAdjoint
    return {
            'FISTA':   {'class': FISTA,
                        'primal': True,
                        'full': 'Fast Iterative Thresholding for linear problems'
                       },
            'linLW':    {'class': linLandweber, 
                         'full': 'Landweber iteration for linear problems'
                        },
            'Richardson':{'class': Richardson,
                          'full': 'relaxed Richardson method'
                        },
            'nu':       {'class': NuMethod,
                         'full': 'nu-method'
                        },
            'nu_sa':    {'class': NuMethodSelfAdjoint,
                         'full': 'nu-method for self-adjoint operators'
                        },
            'CGNE':     {'class':CGNE, 
                         'full': 'Conjugate Gradient method applied to the normal equation'
                        },
            'TikhOnGrid':{'class': TikhonovAlphaGrid,
                          'full': 'Tikhonov regularization applied for a grid of regularization parameters (special case of genTikh)'
                         },
            'genTikh':   {'class': GeneralizedTikhonov,
                          'full': 'Generalized Tikhonov regularization applied for a grid of regularization parameters'
                        },
            'Spectral': {'class': Spectral,
                         'full': 'spectral regularization by an SVD filter'
                        },
            'Spectral_SelfAdjoint': {
                         'class': Spectral_SelfAdjoint,
                         'full': 'spectral regularization of a self-adjoint operator'
                        },
    }


def get_nonlinear_reg_method_dictionary() -> dict:
    """Return regularization methods intended for nonlinear operators.

    Linear operators may also satisfy the applicability conditions of these
    methods. They are deliberately omitted from the default dictionary of a
    linear setting because the specialized linear methods are preferable.
    """
    from .nonlinear import (
        IrgnmCG,
        Landweber as nlLandweber,
        FISTA as genFISTA,
        LevenbergMarquardt,
        NewtonCG,
        NewtonCGFrozen,
        IterativelyRegularizedNewton
    )
    return {
            'genFISTA': {'class': genFISTA,
                         'primal': True,
                         'full': 'Fast Iterative Thresholding for nonlinear problems'
                        },
            'nlLW':     {'class': nlLandweber,
                         'full': 'Landweber iteration for nonlinear problems'
                        },
            'IRGNM_CG':     {'class': IrgnmCG, 
                         'full': 'Iteratively Regularized Gauss-Newton Method with inner CG solver'
                        },
            'LevMar':     {'class': LevenbergMarquardt, 
                         'full': 'Levenberg-Marquardt method'
                        },
            'NewtonCG':     {'class': NewtonCG, 
                         'full': 'Newton method with inner CGNE regularized by early stopping'
                        },
            'NewtCGFroz':     {'class': NewtonCGFrozen, 
                         'full': 'Newton-CG method with derivative frozen at initial guess'
                        },
            'IRN' :      {'class': IterativelyRegularizedNewton,
                          'full': 'Iteratively regularized Newton methods (generalized IRGNM)'
                          }
    }


def get_reg_method_dictionary() -> dict:
    """Return all non-Tikhonov regularization methods.

    This compatibility helper combines the linear and nonlinear registries.
    New code should normally use :func:`get_linear_reg_method_dictionary` or
    :func:`get_nonlinear_reg_method_dictionary`. A :class:`Setting` selects
    between those two automatically and never uses this combined dictionary.
    """
    return {
        **get_linear_reg_method_dictionary(),
        **get_nonlinear_reg_method_dictionary(),
    }

def get_convex_solver_dictionary() -> dict:
    """Return the available solvers for convex Tikhonov settings."""
    from .linear import (
        ADMM,
        FISTA,
        ForwardBackwardSplitting,
        PDHG,
        SemiSmoothNewton,
        TikhonovCG,
    )
    method_dict={
            'TikhCG': {'class':TikhonovCG, 'primal': True, 'full':'conjugate gradient method applied to normal equation'},
            'dual_TikhCG': {'class':TikhonovCG, 'primal': False, 'full':'conjugate gradient method applied to dual normal equation'},
            'FB': {'class':ForwardBackwardSplitting, 'primal': True, 'full':'Forward Backward Splitting applied to primal problem'},
            'dual_FB': {'class':ForwardBackwardSplitting, 'primal': False, 'full': 'Forward Backward Splitting applied to primal problem'},
            'FISTA': {'class':FISTA, 'primal': True, 'full': 'Fast Iterative Thresholding applied to primal problem'}, 
            'dual_FISTA': {'class':FISTA, 'primal': False, 'full': 'Fast Iterative Thresholding applied to dual problem'},
            'PDHG': {'class':PDHG, 'primal': True, 'full': 'Primal-Dual Hybrid Gradient Method applied to primal problem'},
            'dual_PDHG': {'class':PDHG, 'primal': False, 'full': 'Primal-Dual Hybrid Gradient Method applied to dual problem'},
            'ADMM': {'class':ADMM, 'primal': True, 'full': 'Alternating Direction Method of Multipliers' },
            'SSNewton': {'class':SemiSmoothNewton, 'primal': True, 'full': 'Semismooth Newton method'},
            'dual_SSNewton': {'class':SemiSmoothNewton, 'primal': False, 'full': 'Semismooth Newton method applied to dual problem'}
        }
    return method_dict

def get_non_convex_solver_dictionary() -> dict:
    """Return the available solvers for non-convex Tikhonov settings."""
    from .nonlinear import FISTA as genFISTA
    from .nonlinear import ForwardBackwardSplitting as genFB

    method_dict={
            'genFB': {'class':genFB, 'primal': True, 'full':'generalized forward-backward splitting'},
            'genFISTA': {'class':genFISTA, 'primal': True, 'full':'Fast Iterative Thresholding for non-convex problems'},
    }
    return method_dict

def check_Hilbert_applicability(setting: Setting,
                                user_data: ReadOnlyVector | ObsDataT | None = None,
                                user_init: ReadOnlyVector | None = None,
                                regularization_parameter_required: bool = False,
                                check_is_norm_squared_type: bool = True,
                                **kwargs
                                ) -> tuple[dict, dict]:
    """Check whether a Hilbert-space regularization method is applicable.

    Parameters
    ----------
    setting : Setting
        Setting to check.
    user_data : ReadOnlyVector or ObsDataT or None, optional
        Data supplied directly to the solver. For a codomain-free setting,
        these may be arbitrary observational data accepted by
        :meth:`Operator.adjoint_data`.
    user_init : ReadOnlyVector or None, optional
        Initial guess supplied directly to the solver.
    regularization_parameter_required : bool, optional
        Whether the method requires a setting with a regularization
        parameter. If false, the setting must not have one.
    check_is_norm_squared_type : bool, optional
        If true, require the data-fidelity and penalty terms to be shifted
        multiples of squared Hilbert norms. Otherwise, only quadraticity is
        required.
        
    Returns
    -------
    out : dict
        ``out['applicable']`` states whether the method is applicable. If it
        is false, ``out['info']`` gives the reasons.
    par : dict
        Resolved parameters. ``par['data']`` is ``user_data`` when provided.
        Otherwise it is the setting's stored observational data for a
        codomain-free setting, possibly ``None``, or the reference vector of
        the quadratic data-fidelity term as returned by its
        :meth:`~regpy.functionals.Functional.reference_vector`.
        ``par['init']`` is ``user_init`` when provided, otherwise the
        setting's stored initial guess or the penalty's reference vector.
    """
    out = {'info':''}; par = {} 
    if check_is_norm_squared_type:
        if not setting.penalty.is_norm_squared_type:
            out['info'] += 'Penalty functional is not of squared Hilbert-norm type. '
        if not (setting.codomain_free or setting.data_fid.is_norm_squared_type):
            out['info'] += 'Data fidelity functional is not of squared Hilbert-norm type. '
    else:
        if not setting.penalty.is_quadratic:
            out['info'] += 'Penalty functional is not quadratic. '
        if not (setting.codomain_free or setting.data_fid.is_quadratic):
            out['info'] += 'Data fidelity term is not quadratic. '
    if setting.is_tikhonov and (not regularization_parameter_required):
        out['info'] += 'Setting has a specified regularization parameter. '
    if (not setting.is_tikhonov) and regularization_parameter_required:
        out['info'] += 'Setting has no specified regularization parameter. '    
    if setting.codomain_free or setting.data_fid.is_quadratic:
        if user_data is not None:
            if not setting.codomain_free and user_data not in setting.op.codomain:
                raise TypeError(Errors.not_in_vecsp(
                    user_data,
                    setting.op.codomain,
                    vec_name='data',
                    space_name='codomain',
                ))
            par['data'] = user_data
        elif setting.codomain_free:
            # Missing data do not make a method structurally inapplicable:
            # they may still be supplied when constructing the solver.
            par['data'] = setting.data
        else:
            par['data'] = setting.data_fid.reference_vector()

    if user_init is None and setting.penalty.is_quadratic:
        par['init'] = setting.init
    else:
        par['init'] = user_init     
    if setting.is_tikhonov:
        par['regpar']= setting.regpar
    
    out['applicable'] = out['info'] == ''
    return out, par

def check_Hilbert_applicability_linear(setting:Setting,
                                        user_data: ReadOnlyVector | ObsDataT | None= None, 
                                        user_init: ReadOnlyVector | None =None,
                                        regularization_parameter_required: bool = False,
                                        check_is_norm_squared_type: bool = True,
                                        **kwargs
                                    )->tuple[dict,dict] :
    """ Checks if regularization methods for linear operators in Hilbert spaces are applicable.

    Parameters and return values as of check_Hilbert_applicability. 
    """
    out, par = check_Hilbert_applicability(setting,
                                           user_data=user_data,
                                           user_init=user_init,
                                           regularization_parameter_required=regularization_parameter_required,
                                           check_is_norm_squared_type = check_is_norm_squared_type
                                           )
    if not setting.op.linear:
        out['info'] += 'Operator is not linear. '
    out['applicable'] = out['info'] == ''
    return out, par
