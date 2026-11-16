from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from regpy.vecsps.base import TupleVector

from collections import defaultdict
from types import NotImplementedType
from typing import Any, Callable, Iterator
from copy import copy
from math import inf
import logging

import numpy as np
from numpy import isscalar

from regpy import operators, util, vecsps, hilbert
from regpy.operators import Operator, Zero, ObsDataT
from regpy.vecsps import VectorSpaceBase,NumPyVectorSpace
from regpy.vecsps.base import Scalar, Vector, ReadOnlyVector
from regpy.hilbert import HilbertSpace
from regpy.util import Errors

__all__ = ["as_functional", "AbstractFunctional", "Functional", "LinearFunctional", "LinearCombination", "Composed","SquaredNorm", "VerticalShift", "HorizontalShiftDilation", "FunctionalOnDirectSum"]

class NotInEssentialDomainError(Exception):
    r"""
    Raised if value of the functional is inf at given argument. In this case the subdifferential is empty. 
    """
    pass


class NotTwiceDifferentiableError(Exception):
    r"""
    Raised if hessian is called at an argument where a functional is not twice differentiable. 
    """
    pass


class AbstractFunctionalBase:
    r"""Class representing abstract functionals without reference to a concrete implementation.

    Abstract functionals do not have elements, properties or any other structure, their sole purpose is
    to pick the proper concrete implementation for a given vector space.
    """

    log = util.ClassLogger()

    def __mul__(self, other: object) -> AbstractFunctionalBase | NotImplementedType:
        if isscalar(other) and other == 1:
            return self
        elif isinstance(other, operators.Operator):
            return AbstractComposed(self, other)
        return NotImplemented

    def __rmul__(self, other: object) -> AbstractFunctionalBase | NotImplementedType:
        if isscalar(other):
            if other == 1:
                return self
            elif util.is_real_dtype(other):
                return AbstractLinearCombination((other, self))
        return NotImplemented

    def __truediv__(self, other: Scalar) -> AbstractFunctionalBase:
        return (1 / other) * self

    def __add__(self, other: object) -> AbstractFunctionalBase | NotImplementedType:
        if isinstance(other, AbstractFunctional):
            return AbstractLinearCombination(self, other)
        elif isscalar(other):
            return AbstractVerticalShift(self, other)
        return NotImplemented

    def __radd__(self, other: object) -> AbstractFunctionalBase | NotImplementedType:
        return self + other

    def __sub__(self, other: object) -> AbstractFunctionalBase | NotImplementedType:
        return self + (-other)

    def __rsub__(self, other: object) -> AbstractFunctionalBase | NotImplementedType:
        return (-self) + other

    def __neg__(self) -> AbstractFunctionalBase:
        return (-1) * self

    def __pos__(self) -> AbstractFunctionalBase:
        return self


class AbstractFunctional(AbstractFunctionalBase):
    r"""An abstract functional that can be called on a vector space to get the corresponding
    concrete implementation.
    
    AbstractFunctionals provides two kinds of functionality:
    
     * A decorator method `register(vecsp_type)` that can be used to declare some class or function
       as the concrete implementation of this abstract functional for vector spaces of type `vecsp_type`
       or subclasses thereof, e.g.:
     * AbstractFunctionals are callable. Calling them on a vector space and arbitrary optional
       keyword arguments finds the corresponding concrete `regpy.functionals.Functional` among all
       registered implementations. If there are implementations for multiple base classes of the
       vector space type, the most specific one will be chosen. The chosen implementation will
       then be called with the vector space and the keyword arguments, and the result will be
       returned.
    
    .. highlight:: python
    .. code-block:: python
    
        @TV.register(vecsps.UniformGridFcts)
        class TVUniformGridFcts(HilbertSpace):
            ...
    
    If called without a vector space as positional argument, it returns a new abstract functional
    with all passed keyword arguments remembered as defaults.
    
    Parameters
    ----------
    name: str
        A name for this abstract functional. Currently, this is only used in error messages, when no
        implementation was found for some vector space.
    """

    def __init__(self, name: str) -> None:
        self._registry = {}
        self.name = name
        self.args = {}

    def register(
        self,
        vecsp_type: type[VectorSpaceBase] | type[HilbertSpace],
        impl: Callable[..., Functional | NotImplementedType] | None = None,
    ) -> Callable[[Callable[..., Functional]], Callable[..., Functional]] | None:
        r"""Register an implementation for a vector-space class.
        
        When ``impl`` is omitted, return a decorator that performs the
        registration.
        
        Parameters
        ----------
        vecsp_type: type[VectorSpaceBase] | type[HilbertSpace]
            Vector Space on which the functional should be registered.
        impl: Callable[..., Functional | NotImplementedType] | None [default: None]
            Concrete implementation to register.
        
        Returns
        -------
        Callable[[Callable[..., Functional]], Callable[..., Functional]] | None
            ``None`` for direct registration; otherwise, a registration
            decorator.
        """
        if impl is not None:
            self._registry.setdefault(vecsp_type, []).append(impl)
            self.__doc__ += "-"*125+ "\n" + f"--- Implementation for {vecsp_type.__name__} is given by {impl.__name__} with the following documentation ---" +"\n" + f"{impl.__doc__}" +"\n" + "-"*125
        else:
            def decorator(
                i: Callable[..., Functional],
            ) -> Callable[..., Functional]:
                self.register(vecsp_type, i)
                return i
            return decorator

    def __call__(
        self,
        vecsp: VectorSpaceBase | HilbertSpace | None = None,
        **kwargs: Any,
    ) -> AbstractFunctional | Functional:
        if vecsp is None:
            clone = copy(self)
            clone.args = copy(self.args)
            clone.args.update(kwargs)
            return clone
        for cls in type(vecsp).mro():
            if not isinstance(vecsp,(vecsps.VectorSpaceBase, hilbert.HilbertSpace)):
                raise TypeError(util.Errors.type_error("The vecsp of an Abstract functional can be either a RegPy vector space or Hilbert space!"))
            try:
                impls = self._registry[cls]
            except KeyError:
                continue
            kws = copy(self.args)
            kws.update(kwargs)
            for impl in impls:
                result = impl(vecsp, **kws)
                if result is NotImplemented:
                    continue
                if isinstance(result, Functional):
                    return result
                else:
                    raise RuntimeError(util.Errors.not_instance(result,Functional,"AbstractFunctionals called on some vector space have to give some Functional."))
        raise NotImplementedError(
            '{} not implemented on {}'.format(self.name, vecsp)
        )


class AbstractLinearCombination(AbstractFunctional):
    r"""Linear combination of abstract functionals. 
    
    Parameters
    ----------
    *args: AbstractFunctional | tuple[float, AbstractFunctional]
        List of coefficients and functionals to be taken as linear combinations.
    """
    def __init__(
        self,
        *args: AbstractFunctional | tuple[float, AbstractFunctional],
    ) -> None:
        coeff_for_func = defaultdict(lambda: 0)
        for arg in args:
            if isinstance(arg, tuple):
                coeff, func = arg
            else:
                coeff, func = 1, arg
            if not isinstance(func, AbstractFunctional) or not isinstance(coeff,(int,float)):
                raise ValueError(util.Errors.value_error(f"""
        The AbstractLinearCombination only takes a list of arbitrary items provided either tuples 
        (coeff,func) which are a real number and functional or a only a functional. However, you gave:
        [{";  ".join(f"({arg})" for arg in args)}]"""))
            if isinstance(func, type(self)):
                for c, f in zip(func.coeffs, func.funcs):
                    coeff_for_func[f] += coeff * c
            else:
                coeff_for_func[func] += coeff
        self.coeffs = []
        """List of all coefficients
        """
        self.funcs = []
        """List of all functionals. 
        """
        for func, coeff in coeff_for_func.items():
            self.coeffs.append(coeff)
            self.funcs.append(func)

    def __call__(self, vecsp: VectorSpaceBase, **kwargs: Any) -> Functional:
        if kwargs is not None and len(kwargs) != 0:
            raise ValueError(util.Errors.value_error("""
    An AbstractLinearCombination of functionals cannot process generic keyword arguments. Please modify the specific 
    AbstractFunctional by either editing it before of modifying it by calling the item. 
    That is for example to modify the k-th functional use CombinedFunctional[k](arg = ...)."""))
        if not isinstance(vecsp,vecsps.VectorSpaceBase):
            raise ValueError(util.Errors.not_instance(vecsp,vecsps.VectorSpaceBase))
        return LinearCombination(
            *((w,func(vecsp)) for w, func in zip(self.coeffs, self.funcs))
            )

    def __getitem__(
        self, item: int | slice
    ) -> tuple[float, AbstractFunctional] | tuple[list[float], list[AbstractFunctional]]:
        return self.coeffs[item], self.funcs[item]

    def __iter__(self) -> Iterator[tuple[float, AbstractFunctional]]:
        return iter(zip(self.coeffs,self.funcs))

class AbstractVerticalShift(AbstractFunctional):
    r"""Abstract analogue to `VerticalShift` class. Shifting a functional by some offset. Should not be used directly but rather by adding some scalar to the functional. 
    
    Parameters
    ----------
    func: AbstractFunctional
        Functional to be offset.
    offset: float
        Offset added to the evaluation of the functional.
    """
    def __init__(self, func: AbstractFunctional, offset: float) -> None:
        if not isinstance(func, AbstractFunctional) or not isinstance(offset,(int,float)):
            raise ValueError(util.Errors.value_error(f""" 
        The AbstractVerticalShift only takes two arguments one AbstractFunctional 
        and one offset that is a scalar. However, you gave:
            func = {func},
            offset = {offset}."""))
        super().__init__(func.name)
        self.func = func
        """Functional to be offset.
        """
        self.offset = offset
        """Offset added to the evaluation of the functional.
        """

    def __call__(
        self, vecsp: VectorSpaceBase | None = None, **kwargs: Any
    ) -> Functional | None:
        if vecsp is None:
            self.func = self.func(**kwargs)
        elif not isinstance(vecsp,vecsps.VectorSpaceBase):
            raise ValueError(util.Errors.not_instance(vecsp,vecsps.VectorSpaceBase))
        else:
            return VerticalShift(func=self.func(vecsp=vecsp,**kwargs),offset=self.offset)
    
class AbstractComposed(AbstractFunctional):
    r"""Abstract analogue to `Composed`. Composition of an operator with a functional :math:`F\circ O`. This should not be called
    directly but rather used by multiplying the `AbstractFunctional` object with an `regpy.operators.Operator`.
    
    Parameters
    ----------
    func: AbstractFunctional
        Functional to be composed with.
    op: Operator
        Operator to be composed with.
    """
    def __init__(self, func: AbstractFunctional, op: Operator) -> None:
        if not isinstance(func, AbstractFunctional):
            raise ValueError(util.Errors.not_instance(func,AbstractFunctional))
        if not isinstance(op,operators.Operator):
            raise ValueError(util.Errors.not_instance(op,operators.Operator))
        super().__init__(op.domain)
        if isinstance(func, type(self)):
            op = func.op * op
            func = func.func
        self.func = func
        """Functional that is composed with an Operator. 
        """
        self.op = op
        """Operator composed that is composed with a functional. 
        """

    def __call__(
        self, vecsp: VectorSpaceBase | None = None, **kwargs: Any
    ) -> Functional | None:
        if vecsp is None:
            self.func = self.func(**kwargs)
        elif not isinstance(vecsp,vecsps.VectorSpaceBase):
            raise ValueError(util.Errors.not_instance(vecsp,vecsps.VectorSpaceBase))
        else:
            return Composed(func=self.func(vecsp),op=self.op)


class Functional:
    r"""Base class for implementation of functionals. Subclasses should at least implement the 
        `_eval` :  evaluating the functional
    and 
        `_subgradient` or `_linearize` : returning a subgradient at `x`.
    
    The evaluation of a specific functional on some element of the `domain` can be done by
    simply calling the functional on that element. 
    
    Functionals can be added by taking `LinearCombination` of them. The `domain` has to be the
    same for each functional. 
    
    They can also be multiplied by scalars or vector of their respective `domain`or multiplied by 
    `regpy.operators.Operator`. This leads to a functional that is composed with the operator
    :math:`F\circ O` where :math:`F` is the functional and :math:`O` some operator. Multiplying by a scalar
    results in a composition with the `PtwMultiplication` operator.
    
    Parameters
    ----------
    domain: VectorSpaceBase
        The underlying vector space for the function space on which it is defined.
    h_domain: HilbertSpace | None [default: None]
        The underlying Hilbert space. The proximal mapping, the parameter of strong convexity, 
        and the Lipschitz constant are defined with respect to this Hilbert space.
        In the default case `L2(domain)` is used.
    is_linear: bool [default: False]
        If true, the functional should be linear.
    is_quadratic: bool [default: False]
        If true, the functional should be quadratic.
    conj_is_quadratic: bool | None [default: None]
        The corresponding boolean attribute indicates if the conjugate functional is quadratic.
        If None, the same value as is_quadratic will be used.
    is_separable: bool [default: False]
        If true, the functional should be the sum of functionals acting on only one component of the input vector.
        In this case, the parameters
    is_convex: bool [default: True]
        If true, the functional should be convex.
    convexity_param: float [default: 0.0]
        parameter of strong convexity of the functional. 
        0 if the functional is not strongly convex.
    Lipschitz: float [default: inf]
        Lipschitz continuity constant of the gradient.  
        math.inf the gradient is not Lipschitz continuous.
    dom_l: Vector | Scalar | None [default: None]
        should not be None, and they should specify the essential domain of the functional by 
        :math:`\{x in domain: dom_l<=x<=dom_u\}`, 
        and the essential domain of the conjugate functional (which is then also separable) by 
        :math:`\{xstar in domain: conj_dom_l<=xstar <= conj_dom_u\}`.
        In case of open domains, the boundaries should be shifted in the order of machine precision.
    dom_u: Vector | Scalar | None [default: None]
        should not be None, and they should specify the essential domain of the functional by 
        :math:`\{x in domain: dom_l<=x<=dom_u\}`, 
        and the essential domain of the conjugate functional (which is then also separable) by 
        :math:`\{xstar in domain: conj_dom_l<=xstar <= conj_dom_u\}`.
        In case of open domains, the boundaries should be shifted in the order of machine precision.
    conj_dom_l: Vector | Scalar | None [default: None]
        should not be None, and they should specify the essential domain of the functional by 
        :math:`\{x in domain: dom_l<=x<=dom_u\}`, 
        and the essential domain of the conjugate functional (which is then also separable) by 
        :math:`\{xstar in domain: conj_dom_l<=xstar <= conj_dom_u\}`.
        In case of open domains, the boundaries should be shifted in the order of machine precision.
    conj_dom_u: Vector | Scalar | None [default: None]
        should not be None, and they should specify the essential domain of the functional by 
        :math:`\{x in domain: dom_l<=x<=dom_u\}`, 
        and the essential domain of the conjugate functional (which is then also separable) by 
        :math:`\{xstar in domain: conj_dom_l<=xstar <= conj_dom_u\}`.
        In case of open domains, the boundaries should be shifted in the order of machine precision.
    methods: set[str] [default: set()]
        names of the methods implemented by a given Functional instance.
        Subset of {'eval', 'subgradient', 'hessian', 'proximal', 'dist_subdiff'}
    conj_methods: set[str] [default: set()]
        Names of the implemented functional operations.
    is_data_func: bool [default: False]
        If true, the functional acts as a "data functional". In this case it must have a `data` property with a corresponding setter.
        Often data is a vector in domain, and the functional attains its infimum at this point.  
        But this is not always the case -- think of the realization of a point process defining
        a Kullback-Leibler functional, for example. 
        This property is also used to incorporate an initial guess in penalty functionals.
    """

    log = util.ClassLogger()

    def __init__(self, 
                 domain:VectorSpaceBase, 
                 h_domain:HilbertSpace | None = None,
                 is_linear:bool = False,
                 is_quadratic:bool = False,  
                 conj_is_quadratic:bool|None = None,               
                 is_separable:bool = False,
                 is_convex:bool = True,
                 convexity_param:float=0.,
                 Lipschitz:float = inf,
                 dom_l: Vector | Scalar | None = None,
                 dom_u: Vector | Scalar | None = None,
                 conj_dom_l: Vector | Scalar | None = None,
                 conj_dom_u: Vector | Scalar | None = None,
                 methods: set[str] = set(), conj_methods: set[str] = set(),
                 is_data_func:bool = False
                 ) -> None:
        if not isinstance(domain, vecsps.VectorSpaceBase):
            raise TypeError(f'domain must be an instance of VectorSpaceBase. Got {domain}')
        self.domain = domain
        """The underlying vector space."""
        self.h_domain = hilbert.as_hilbert_space(h_domain,domain) or hilbert.L2(domain)
        """The underlying Hilbert space."""

        if not isinstance(convexity_param, (float, np.floating,int,np.integer)) and convexity_param>=0:
            raise ValueError(f'convexity_param must be a scalar, nonnegative float. Got {convexity_param}.')
        self._convexity_param = np.float64(convexity_param)
        """parameter of strong convexity of the functional."""
        if not isinstance(Lipschitz, (float, np.floating,int,np.integer)) and Lipschitz>=0:
            raise ValueError(f'Lipschitz must be a scalar, nonnegative float. Got {Lipschitz}.')
        self._Lipschitz = np.float64(Lipschitz)
        """Lipschitz continuity constant of the gradient."""

        if not isinstance(is_linear,(bool, np.bool_)):
            raise TypeError(Errors.type_error('is_linear must be boolean.',is_linear))
        self._is_linear = is_linear
        """boolean indicating if the functional is linear"""
        if not isinstance(is_quadratic,(bool, np.bool_)):
            raise TypeError(Errors.type_error('is_quadratic must be boolean.',is_quadratic))
        self._is_quadratic = is_quadratic
        """boolean indicating if the functional is quadratic"""
        if not isinstance(conj_is_quadratic,(bool, np.bool_,type(None))):
            raise TypeError(Errors.type_error('conj_is_quadratic must be boolean or None.',conj_is_quadratic))
        self._conj_is_quadratic = is_quadratic if conj_is_quadratic is None else conj_is_quadratic
        """boolean indicating if the conjugate functional is quadratic"""
        if not isinstance(is_separable,(bool, np.bool_)):
            raise TypeError(Errors.type_error('is_separable must be boolean.',is_separable))
        self._is_separable = is_separable
        """boolean indicating if the functional is separable."""
        if not isinstance(is_convex,(bool, np.bool_)):
            raise TypeError(Errors.type_error('is_convex must be boolean.',is_convex))
        self._is_convex = is_convex
        """boolean indicating if the functional is convex."""


        if is_separable:
            if isinstance(dom_l,np.ndarray) and isinstance(dom_u,np.ndarray) and np.any(dom_l>dom_u):
                raise ValueError('dom_l must be smaller or equal to dom_u.')
            if conj_dom_l is not None  and isinstance(conj_dom_l,np.ndarray) \
                and conj_dom_u is not None and isinstance(conj_dom_u, np.ndarray) \
                and np.any(conj_dom_l>conj_dom_u):
                raise ValueError(util.Errors.value_error(f'conj_dom_l must be smaller or equal conj_dom_u. Was given conj_dom_l = {conj_dom_l} and conj_dom_u = {conj_dom_u}',self,"__init__"))
        self._dom_l, self._dom_u, self._conj_dom_l, self._conj_dom_u = dom_l, dom_u, conj_dom_l, conj_dom_u
        """vectors indicating the essential domain of the functional and its conjugate"""

        if not methods <= {'eval','subgradient','hessian','proximal','dist_subdiff'}:
            raise ValueError(f"Given methods set {methods} contains inadmissable elements.")
        else:
            self._methods  =methods
        if not conj_methods <= {'eval','subgradient','hessian','proximal','dist_subdiff'}:
            raise ValueError(f"Given methods set {methods} contains inadmissable elements.")
        else:
            self._conj_methods = conj_methods

        self.is_data_func = is_data_func


    def _validate_out(self, out: Vector | None) -> Vector | None:
        r"""Validate an optional output vector and return it unchanged.
        
        Parameters
        ----------
        out: Vector | None
            Writable output buffer. If None, allocate a new result.
        
        
        Returns
        -------
        Vector | None
            Result described above.
        """
        if out is not None and out not in self.domain:
            raise ValueError(util.Errors.not_in_vecsp(
                out, self.domain, vec_name="out", space_name="domain",
                add_info=f"The output vector for {self} must belong to its domain."
            ))
        return out

    def _copy_vector(
        self, x: ReadOnlyVector, out: Vector | None = None
    ) -> Vector:
        r"""Return ``out`` containing ``x``, allocating only if necessary.
        
        Parameters
        ----------
        x: ReadOnlyVector
            Input vector; its coefficients are only read.
        out: Vector | None [default: None]
            Writable output buffer. If None, allocate a new result.
        
        
        Returns
        -------
        Vector
            Writable result vector.
        """
        if out is None:
            out = self.domain.empty()
        if out is not x and self._shares_memory(out, x):
            source = self.domain.empty()
            self.domain.copyto(source, x)
            x = source
        self.domain.copyto(out, x)
        return out

    @staticmethod
    def _shares_memory(x: ReadOnlyVector, y: ReadOnlyVector) -> bool:
        r"""Whether writing to one vector can modify the other.
        
        Parameters
        ----------
        x: ReadOnlyVector
            Input vector; its coefficients are only read.
        y: ReadOnlyVector
            Input vector; its coefficients are only read.
        
        
        Returns
        -------
        bool
            Whether the stated condition is satisfied.
        """
        if x is y:
            return True
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            return np.shares_memory(x, y)
        if isinstance(x, vecsps.TupleVector):
            return any(Functional._shares_memory(x_i, y) for x_i in x)
        if isinstance(y, vecsps.TupleVector):
            return any(Functional._shares_memory(x, y_i) for y_i in y)
        return False

    def __call__(self, x: ReadOnlyVector) -> float:
        if x not in self.domain:
            raise ValueError(util.Errors.not_in_vecsp(x,self.domain,space_name="domain", add_info=f"Not able to evaluate {self} on the given vector."))
        try:
            y = self._eval(x)
        except NotImplementedError:
            y, _ = self._linearize(x)
        if not isinstance(y, (int,float)):
            raise RuntimeError(util.Errors.not_instance(x,float, add_info=f"The evaluation of the functional {self} did not return a float or int."+"\n\t"+f"x = {x}"))
        return y

    def linearize(self, x: ReadOnlyVector) -> tuple[float, Vector]:
        r"""Bounds the functional from below by a linear functional at `x` given by the value at that point and a subgradient v such that
        
        .. math::
            F(x+ h) \geq  F(x) + vdot(v,h) for all h
        
        Requires the implementation of either `_subgradient` or `_linearize`.
        
        Parameters
        ----------
        x: ReadOnlyVector
            Element at which will be linearized
        
        Return
        ------
        y 
            Value of :math:`F(x)`.
        grad : in self.domain
            Subgradient of :math:`F` at :math:`x`.        
        
        Returns
        -------
        tuple[float, Vector]
            Result described above.
        """
        if x not in self.domain:
            raise ValueError(util.Errors.not_in_vecsp(x,self.domain,space_name="domain", add_info=f"Not able to linearize {self} on the given vector."))
        try:
            y, grad = self._linearize(x)
        except NotImplementedError:
            y = self._eval(x)
            grad = self._subgradient(x)
        if not isinstance(y, (int,float)):
            raise RuntimeError(util.Errors.not_instance(y,float, add_info=f"The evaluation of the functional {self} did not return a float or int."+"\n\t" + f"x = {x}"))
        if grad not in self.domain:
            raise RuntimeError(util.Errors.not_in_vecsp(grad,self.domain,vec_name="gradient",space_name="domain", add_info=f"The computation of the gradient of functional {self} did not return an element in the domain." ))
        return y, grad

    def subgradient(
        self, x: ReadOnlyVector, out: Vector | None = None, **subgradient_par: Any
    ) -> Vector:
        r"""Return a subgradient of the functional at ``x``.
        
        If ``out`` is supplied, the result is written into that vector and the
        same object is returned. ``out`` may coincide with ``x``.
        
        Parameters
        ----------
        x: ReadOnlyVector
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
        if x not in self.domain:
            raise ValueError(util.Errors.not_in_vecsp(
                x, self.domain, space_name="domain",
                add_info=f"Not able to compute subgradient of {self} on the given vector."
            ))
        self._validate_out(out)
        try:
            grad = self._subgradient(x, out=out, **subgradient_par)
        except NotImplementedError:
            _, grad = self._linearize(x)
        if out is not None:
            grad = self._copy_vector(grad, out=out)
        if grad not in self.domain:
            raise RuntimeError(util.Errors.not_in_vecsp(
                grad, self.domain, vec_name="gradient", space_name="domain",
                add_info=f"The computation of the gradient of functional {self} did not return an element in the domain."
            ))
        return grad

    def reference_vector(self, out: Vector | None = None) -> Vector:
        r"""Return a feasible reference vector for the functional.
        
        The reference vector is intended, in particular, as an initial guess
        for iterative methods. Ideally, it is a global minimizer of the
        functional. As a minimal requirement, it is guaranteed to be feasible, meaning
        that the functional has a finite value at the returned vector.
        
        The default implementation first evaluates
        
        .. code-block:: python
        
            self.conj.subgradient(self.domain.zeros(read_only=True))
        
        If this conjugate subgradient is available, the result is a global
        minimizer. If it is unavailable, or if zero is outside the essential
        domain of the conjugate, the proximal map is evaluated at zero with
        parameter ``tau=1``. Notice that zero being outside the effective
        domain of the conjugate is equivalent to the primal functional being
        unbounded below. An implemented proximal map always returns a feasible
        point. Only if the proximal map is unavailable does the default
        implementation try the zero vector itself. If none of these strategies
        produces a feasible vector, :class:`NotInEssentialDomainError` is
        raised.
        
        Derived classes may override :meth:`_reference_vector` if the default
        implementation is not applicable or not appropriate. 
        
        Parameters
        ----------
        out: Vector | None [default: None]
            If supplied, the result is written into this vector and the same
            object is returned.
        
        Returns
        -------
        Vector
            A vector in :attr:`domain` at which the functional is finite.
        
        Raises
        ------
        ValueError
            If ``out`` does not belong to :attr:`domain`.
        RuntimeError
            If an overridden :meth:`_reference_vector` returns an object
            outside :attr:`domain`.
        NotInEssentialDomainError
            If no feasible reference vector can be constructed, or if an
            overridden :meth:`_reference_vector` returns an infeasible vector.
        """
        self._validate_out(out)
        reference = self._reference_vector(out=out)
        if out is not None:
            reference = self._copy_vector(reference, out=out)
        if reference not in self.domain:
            raise RuntimeError(util.Errors.not_in_vecsp(
                reference, self.domain, vec_name="reference", space_name="domain",
                add_info=(
                    f"The reference vector of functional {self} does not "
                    "belong to its domain."
                ),
            ))
        try:
            value = self(reference)
        except NotInEssentialDomainError as error:
            raise NotInEssentialDomainError(
                f"The reference vector returned by {self} is not feasible."
            ) from error
        if not np.isfinite(value):
            raise NotInEssentialDomainError(
                f"The reference vector returned by {self} is not feasible: "
                f"the functional value is {value}."
            )
        return reference

    def conj_reference_vector(self, out: Vector | None = None) -> Vector:
        r"""Return a feasible reference vector for the conjugate functional.
        
        This method is the conjugate counterpart of :meth:`reference_vector`.
        Ideally, it returns a global minimizer of the conjugate functional; at
        minimum, the conjugate functional is finite at the returned vector.
        
        The default implementation first evaluates
        
        .. code-block:: python
        
            self.subgradient(self.domain.zeros(read_only=True))
        
        If this is unavailable or zero is outside the primal essential domain,
        it evaluates the conjugate proximal map at zero with parameter
        ``tau=1``. Only if that proximal map is unavailable does it return zero,
        provided zero is feasible for the conjugate functional. Derived classes
        may override :meth:`_conj_reference_vector`.
        
        Parameters
        ----------
        out: Vector | None [default: None]
            If supplied, the result is written into this vector and the same
            object is returned.
        
        Returns
        -------
        Vector
            A vector in :attr:`domain` at which the conjugate functional is
            finite.
        
        Raises
        ------
        ValueError
            If ``out`` does not belong to :attr:`domain`.
        RuntimeError
            If an overridden :meth:`_conj_reference_vector` returns an object
            outside :attr:`domain`.
        NotInEssentialDomainError
            If no feasible conjugate reference vector can be constructed, or
            if an overridden :meth:`_conj_reference_vector` returns an
            infeasible vector.
        """
        self._validate_out(out)
        reference = self._conj_reference_vector(out=out)
        if out is not None:
            reference = self._copy_vector(reference, out=out)
        if reference not in self.domain:
            raise RuntimeError(util.Errors.not_in_vecsp(
                reference, self.domain, vec_name="conjugate reference",
                space_name="domain",
                add_info=(
                    f"The conjugate reference vector of functional {self} "
                    "does not belong to its domain."
                ),
            ))
        try:
            value = self.conj(reference)
        except NotInEssentialDomainError as error:
            raise NotInEssentialDomainError(
                f"The conjugate reference vector returned by {self} is not "
                "feasible."
            ) from error
        if not np.isfinite(value):
            raise NotInEssentialDomainError(
                f"The conjugate reference vector returned by {self} is not "
                f"feasible: the conjugate functional value is {value}."
            )
        return reference
    
    def dist_subdiff(
        self, vstar: ReadOnlyVector, x: ReadOnlyVector, best_approx: Vector | None = None
    ) -> float:
        r"""Returns the distance of a vector :math:`v^*` to the subdifferential :math:`\partial F(x)` at x with respect 
        to the dual norm.
        Needs to be re-implemented for functionals which are not Gateaux differentiable.
        
        Parameters
        ----------
        vstar: ReadOnlyVector
            Vector of which the distance is to be determined.
        x: ReadOnlyVector
            Point at which the subdifferential is evaluated
        best_approx: Vector | None [default: None]
            Pre-allocated output vector. If provided, write the best
            approximation to vstar in the subdifferential at x into it.
            The default implementation uses ``subgradient(x)``; nonsmooth
            functionals need to override this method to project onto the
            whole subdifferential. The output may alias either input.
        
        Returns
        -------
        float
            Distance in the dual Hilbert norm. Providing best_approx does
            not change the return type or the distance.
        """
        self._validate_out(best_approx)
        proj_vstar = self.domain.empty()
        self.subgradient(x, out=proj_vstar)
        distance = self.h_domain.dual_space().norm(vstar - proj_vstar)
        if best_approx is not None:
            self._copy_vector(proj_vstar, out=best_approx)
        return distance

    def hessian(
        self, x: ReadOnlyVector, recursion_safeguard: bool = False
    ) -> operators.Operator:
        r"""The hessian of the functional at `x` as an `regpy.operators.Operator` mapping form the 
        functionals `domain` to it self. It is defined by 
        
        .. math::
            F(x+h) = F(x) + (\nabla F)(x)^T h + \frac{1}{2} h^T Hess F(x) h + \mathcal{o}(\|h\|^2)
        
        Require either the implementation of _hessian or of _hessian_conj and _subgradient
        
        Parameters
        ----------
        x: ReadOnlyVector
            Input vector; its coefficients are only read.
        recursion_safeguard: bool [default: False]
            Recursion safeguard.
        
        Returns
        -------
        `h` : `regpy.operators.Operator`
            Hessian operator at the point `x`.
        """
        if x not in self.domain:
            raise ValueError(util.Errors.not_in_vecsp(x,self.domain,space_name="domain", add_info=f"Not able to compute hessian of {self} on the given vector."))
        try:
            h = self._hessian(x)
        except NotImplementedError:
            if recursion_safeguard:
                raise NotImplementedError("Neither hessian nor conj_hessian are implemented.")
            else:
                h = self.conj_hessian(self.subgradient(x),recursion_safeguard=True).inverse
        if isinstance(h, operators.Operator) and h.linear and h.domain == h.codomain == self.domain:
            return h
        else:
            raise RuntimeError(util.Errors.not_instance(h,operators.Operator,add_info=f"The computed Hessian of {self} is either nonlinear or its domain or codomain are not identical to the domain of the functional."))

    def _bregman_reference(self, y: ReadOnlyVector, ystar: ReadOnlyVector | None) -> tuple[ReadOnlyVector, Any, float]:
        r"""Validate and copy a Bregman base point and its dual subgradient.
        
        Parameters
        ----------
        y: ReadOnlyVector
            Input vector; its coefficients are only read.
        ystar: ReadOnlyVector | None
            Input vector; its coefficients are only read.
        
        
        Returns
        -------
        tuple[ReadOnlyVector, Any, float]
            Result described above.
        """
        if y not in self.domain:
            raise ValueError('y must belong to the functional domain.')
        y = self._copy_vector(y)
        value = self(y)
        if not np.isfinite(value):
            raise NotInEssentialDomainError('The Bregman base point must have finite functional value.')
        if ystar is None:
            ystar = self.subgradient(y)
        else:
            if ystar not in self.domain:
                raise ValueError('ystar must belong to the functional domain (dual coordinates).')
            distance = self.dist_subdiff(ystar, y)
            tolerance = 64 * np.finfo(float).eps * max(
                1., self.h_domain.dual_space().norm(ystar))
            if not np.isfinite(distance) or distance > tolerance:
                raise ValueError(Errors.value_error('ystar must belong to the subdifferential at y.',
                                                    (self.dist_subdiff,tolerance)))
        return y, self._copy_vector(ystar), float(np.real(value))

    def bregman_func(self, y: ReadOnlyVector, ystar: ReadOnlyVector | None = None) -> Functional:
        r"""Construct the Bregman functional based at y.
        
        Returns the functional
        
        .. math::
            x \mapsto F(x)-F(y)-\langle y^*,x-y\rangle.
        
        y must have finite functional value. ystar is a dual covector in
        the subdifferential at y; if omitted, subgradient(y) is used. The
        pairing is the real part of domain.vdot, not a Hilbert inner product.
        The generic result is a LinearCombination with an affine correction.
        Base-point and subgradient vectors are copied. The original F is
        retained as a component and should not subsequently be modified.
        
        The same interface is available on F.conj.
        
        Parameters
        ----------
        y: ReadOnlyVector
            Input vector; its coefficients are only read.
        ystar: ReadOnlyVector | None [default: None]
            Input vector; its coefficients are only read.
        
        
        Returns
        -------
        Functional
            Constructed functional.
        """
        y, ystar, value = self._bregman_reference(y, ystar)
        linear = LinearFunctional(-ystar, domain=self.domain,
                                  h_domain=self.h_domain,
                                  gradient_in_dual_space=True)
        offset = float(np.real(self.domain.vdot(ystar, y))) - value
        return LinearCombination(self, linear, offset=offset)


    def conj_bregman_func(self, y: ReadOnlyVector, ystar: ReadOnlyVector | None = None) -> Functional:
        r"""Construct the conjugate's Bregman functional (overridable hook).
        
        Here y is a point in dual coordinates and ystar is a primal element
        of the conjugate's subdifferential at y.
        
        Parameters
        ----------
        y: ReadOnlyVector
            Input vector; its coefficients are only read.
        ystar: ReadOnlyVector | None [default: None]
            Input vector; its coefficients are only read.
        
        
        Returns
        -------
        Functional
            Constructed functional.
        """
        return Functional.bregman_func(self.conj, y, ystar)


    def bregman(self,x:ReadOnlyVector, y:ReadOnlyVector | None, ystar:ReadOnlyVector | None=None) -> float:
        r"""Computes the Bregman divergence of the functional `F`for two points `x` and `y` 
        with respect to a subgradient :math:`y^* \in \partial F(y)` given by 
        
        .. math::
            \Delta^{y^*}(x,y) := F(x) - F(y) - \langle y^*,x-y\rangle       
        
        The second argument `y` can be None if `ystar` is provided. In this case the Bregman 
        divergence is evaluated by the equivalent formula
        
         .. math::
            \Delta^{y^*}(x,y) := F(x) + F^*(y^*) - \langle y^*,x\rangle  
        
        Parameters
        ----------
        x: ReadOnlyVector
            Input vector; its coefficients are only read.
        y: ReadOnlyVector | None
            Input vector; its coefficients are only read.
        ystar: ReadOnlyVector | None [default: None]
            Input vector; its coefficients are only read.
        
        
        Returns
        -------
        float
            Computed scalar value.
        """

        if x not in self.domain:
            raise ValueError(util.Errors.not_in_vecsp(
                x, self.domain, space_name="domain",
                add_info=f"First argument of bregman cannot be handled by {self}."
            ))
        if y is None:
            if ystar is None:
                raise ValueError("bregman requires y or ystar.")
            if ystar not in self.domain:
                raise ValueError(util.Errors.not_in_vecsp(
                    ystar, self.domain, space_name="domain"
                ))
            return float(np.real(self(x) + self.conj(ystar)
                                 - self.domain.vdot(ystar, x)))
        if y not in self.domain:
            raise ValueError(util.Errors.not_in_vecsp(
                y, self.domain, space_name="domain",
                add_info=f"Second argument of bregman cannot be handled by {self}."
            ))
        if ystar is None:
            ystar = self.subgradient(y)  
        else: 
            if ystar not in self.domain:
                raise ValueError(util.Errors.not_in_vecsp(
                    ystar, self.domain, space_name="domain",
                    add_info=f"Given subgradient of bregman cannot be handled by {self}."
                ))
            dist = self.dist_subdiff(ystar,y)
            if dist>0:
                raise ValueError(util.Errors.value_error('Given subgradient not in subdifferential',(dist,ystar,y)))

        return self(x) - self(y) - self.domain.vdot(ystar,x-y)

    def conj_bregman(self, xstar: ReadOnlyVector, ystar: ReadOnlyVector | None,
                     y: ReadOnlyVector | None = None) -> float:
        r"""A separate implementation for the conjugate is included to allow replacement
        by derived classes.
        
        Parameters
        ----------
        xstar: ReadOnlyVector
            Input vector; its coefficients are only read.
        ystar: ReadOnlyVector | None
            Input vector; its coefficients are only read.
        y: ReadOnlyVector | None [default: None]
            Input vector; its coefficients are only read.
        
        
        Returns
        -------
        float
            Computed scalar value.
        """
        return Functional.bregman(self.conj, xstar, ystar, y)

    def symm_bregman(self,x:ReadOnlyVector, y:ReadOnlyVector,
                     xstar:ReadOnlyVector | None = None, ystar:ReadOnlyVector | None = None) -> float:
        r"""Computes the symmetric Bregman divergence of the functional `F` for two points 
        `x` and `y` with respect to subgradients :math:`xstar \in \partial F(x)` 
        and :math:`ystar \in \partial F(y)` given by 
        
        .. math::
            \Delta_{symm}^{x^*,y^*}(x,y)  F(x) - F(x0) 
                         = \Delta^{y^*}(x,y) + \Delta^{x^*}(y,x)
                         = \langle x^*-y^*,x-y\rangle.       
        
        The latter formula is used for evaluation. 
                         
        Parameters
        ----------
        x: ReadOnlyVector
            Input vector; its coefficients are only read.
        y: ReadOnlyVector
            Input vector; its coefficients are only read.
        xstar: ReadOnlyVector | None [default: None]
            Input vector; its coefficients are only read.
        ystar: ReadOnlyVector | None [default: None]
            Input vector; its coefficients are only read.
        
        
        Returns
        -------
        float
            Computed scalar value.
        """
        if x not in self.domain:
            raise ValueError(util.Errors.not_in_vecsp(
                x, self.domain, space_name="domain",
                add_info=f"First argument of symm_bregman cannot be handled by {self}."
            ))
        if y not in self.domain:
            raise ValueError(util.Errors.not_in_vecsp(
                y, self.domain, space_name="domain",
                add_info=f"Second argument of symm_bregman cannot be handled by {self}."
            ))
        if xstar is None:
            xstar = self.subgradient(x)  
        else: 
            if xstar not in self.domain:
                raise ValueError(util.Errors.not_in_vecsp(
                    xstar, self.domain, space_name="domain",
                    add_info=f"Given subgradient of bregman cannot be handled by {self}."
                ))
            dist = self.dist_subdiff(xstar,x)
            if dist>0:
                raise ValueError(util.Errors.value_error('Given subgradient not in subdifferential',(dist,xstar,x)))
        if ystar is None:
            ystar = self.subgradient(y)  
        else: 
            if ystar not in self.domain:
                raise ValueError(util.Errors.not_in_vecsp(
                    ystar, self.domain, space_name="domain",
                    add_info=f"Given subgradient of bregman cannot be handled by {self}."
                ))
            dist = self.dist_subdiff(ystar,y)
            if dist>0:
                raise ValueError(util.Errors.value_error('Given subgradient not in subdifferential',(dist,ystar,y)))

        return self.domain.vdot(xstar-ystar,x-y)

    def gradient_as_operator(self) -> operators.Operator:
        r"""Return the functional's gradient as a nonlinear operator.
        
        The returned operator maps ``self.domain`` to itself. Its evaluation
        at :math:`x` is :meth:`subgradient`, and the derivative at :math:`x`
        applies the Hessian returned by :meth:`hessian`. Since Hessians are
        symmetric, derivatives of the returned operator are marked as self-adjoint.
        
        Returns
        -------
        operators.Operator
            An operator representing :math:`x \mapsto \partial F(x)`.
        
        Raises
        ------
        NotImplementedError
            If either ``subgradient`` or ``hessian`` is not listed in
            :attr:`methods`.
        """
        required = {'subgradient', 'hessian'}
        missing = required - self.methods
        if missing:
            raise NotImplementedError(util.Errors.generic_message(
                "gradient_as_operator requires the functional methods "
                f"subgradient and hessian. Missing: {', '.join(sorted(missing))}."
            ))
        return _GradientAsOperator(self)

    def conj_subgradient(
        self, xstar: ReadOnlyVector, out: Vector | None = None, **subgradient_par: Any
    ) -> Vector:
        r"""Return a subgradient of the conjugate functional.
        
        If ``out`` is supplied, the result is written into that vector and the
        same object is returned. ``out`` may coincide with ``xstar``.
        
        Parameters
        ----------
        xstar: ReadOnlyVector
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
        if xstar not in self.domain:
            raise ValueError(util.Errors.not_in_vecsp(
                xstar, self.domain, space_name="domain",
                add_info=f"Not able to compute subgradient of conjugate of {self} on the given vector."
            ))
        self._validate_out(out)
        try:
            grad = self._conj_subgradient(xstar, out=out, **subgradient_par)
        except NotImplementedError:
            try:
                _, grad = self._conj_linearize(xstar)
            except (NotInEssentialDomainError, NotImplementedError) as error:
                raise error
        if out is not None:
            grad = self._copy_vector(grad, out=out)
        if grad not in self.domain:
            raise RuntimeError(util.Errors.not_in_vecsp(
                grad, self.domain, vec_name="gradient", space_name="domain",
                add_info=f"The computation of the gradient of conjugate functional {self} did not return an element in the domain."
            ))
        return grad

    def _conj_dist_subdiff(
        self, v: ReadOnlyVector, xstar: ReadOnlyVector, best_approx: Vector | None = None
    ) -> float:
        r"""Returns the distance of a vector :math:`v` to the subdifferential :math:`\partial F^*(x^*)` at :math:`x^*`.
        Needs to be re-implemented for functionals whose conjugate is not Gateaux differentiable.
        
        Parameters
        ----------
        v: ReadOnlyVector
            Vector of which the distance is to be determined.
        xstar: ReadOnlyVector
            Point at which the subdifferential is evaluated
        best_approx: Vector | None [default: None]
            Pre-allocated output for the closest point to v in the
            conjugate subdifferential. The default is
            ``conj_subgradient(xstar)``. May alias either input.
        
        Returns
        -------
        float
            Distance in the primal Hilbert norm.
        """
        self._validate_out(best_approx)
        proj_v = self.domain.empty()
        self.conj_subgradient(xstar, out=proj_v)
        distance = self.h_domain.norm(v - proj_v)
        if best_approx is not None:
            self._copy_vector(proj_v, out=best_approx)
        return distance

    def conj_hessian(
        self, xstar: ReadOnlyVector, recursion_safeguard: bool = False
    ) -> operators.Operator:
        r"""The hessian of the functional. Should not be called directly, but via self.conj.hessian.
                
        Parameters
        ----------
        xstar: ReadOnlyVector
            Input vector; its coefficients are only read.
        recursion_safeguard: bool [default: False]
            Recursion safeguard.
        
        
        Returns
        -------
        operators.Operator
            Result described above.
        """
        if xstar not in self.domain:
            raise ValueError(util.Errors.not_in_vecsp(xstar,self.domain,space_name="domain", add_info=f"Not able to compute hessian of conjugate of {self} on the given vector."))
        try:
            h = self._conj_hessian(xstar)
        except NotImplementedError:
            if recursion_safeguard:
                raise NotImplementedError("Neither hessian nor conj_hessian are implemented.")
            else:
                h = self.hessian(self.conj_subgradient(xstar),recursion_safeguard=True).inverse
        if isinstance(h, operators.Operator) and h.linear and h.domain == h.codomain == self.domain:
            return h
        else:
            raise RuntimeError(util.Errors.not_instance(h,operators.Operator,add_info=f"Computing the Hessian of conjugate of {self} return a possibly nonlinear operator with non matching domain and codomain that is not identical to the domain of the functional."))

    def conj_linearize(self, xstar: ReadOnlyVector) -> tuple[float, Vector]:
        r"""Linearizes the conjugate functional :math:`F^*`. Should not be called directly, but via self.conj.linearize
        
        Parameters
        ----------
        xstar: ReadOnlyVector
            Input vector; its coefficients are only read.
        
        
        Returns
        -------
        tuple[float, Vector]
            Result described above.
        """
        if xstar not in self.domain:
            raise ValueError(util.Errors.not_in_vecsp(xstar,self.domain,space_name="domain", add_info=f"Not able to compute linearize of conjugate of {self} on the given vector."))
        try:
            y, grad = self._conj_linearize(xstar)
        except NotImplementedError:
            y = self._conj(xstar)
            grad = self._conj_subgradient(xstar)
        if not isinstance(y, (int,float)):
            raise RuntimeError(util.Errors.not_instance(y,float, add_info=f"The evaluation of the conjugate functional of {self} for \n x = {xstar} \n did not return a float or int."))
        if grad not in self.domain:
            raise RuntimeError(util.Errors.not_in_vecsp(grad,self.domain,vec_name="gradient",space_name="domain", add_info=f"The computation of the gradient of conjugate of functional {self} did not return an element in the domain." ))
        return y, grad

    def proximal(
        self,
        x: ReadOnlyVector,
        tau: float,
        recursion_safeguard: bool = False,
        out: Vector | None = None,
        **proximal_par: Any,
    ) -> Vector:
        r"""Proximal operator 
        
        .. math::
            \mathrm{prox}_{\tau F}(x)=\arg \min _{v\in {\mathcal {X}}}(F(v)+{\frac{1}{2\tau}}\Vert v-x\Vert_{\mathcal {X}}^{2}).
        
        Requires either an implementation of `_proximal` or of `_subgradient` and `_conj_proximal`.
        
        Parameters
        ----------
        x: ReadOnlyVector
            Vector in the respective domain. Point at which to compute proximal.
        tau: float
            Regularization parameter for the proximal.
        recursion_safeguard: bool [default: False]
            Recursion safeguard.
        out: Vector | None [default: None]
            If supplied, the result is written into that vector and the same object 
            is returned. ``out`` may coincide with ``x``
        **proximal_par: Any
            Proximal par.
        
        Returns
        -------
        proximal : `self.domain`
            the computed proximal at :math:`x` with parameter :math:`\tau`.
        """
        if x not in self.domain:
            raise ValueError(util.Errors.not_in_vecsp(
                x, self.domain, space_name="domain",
                add_info=f"Not able to compute proximal of {self} on the given vector."
            ))
        self._validate_out(out)
        try:
            proximal = self._proximal(x, tau, out=out, **proximal_par)
        except NotImplementedError:
            # evaluation by Moreau's identity
            if recursion_safeguard:
                raise NotImplementedError("Neither proximal nor conj_proximal are implemented.")
            gram = self.h_domain.gram
            gram_inv = self.h_domain.gram_inv
            dual_arg = self.domain.empty()
            gram(x, out=dual_arg)
            dual_arg *= 1. / tau
            self.conj_proximal(
                dual_arg, 1 / tau, recursion_safeguard=True,
                out=dual_arg, **proximal_par
            )
            if out is None or self._shares_memory(out, x):
                proximal = gram_inv(dual_arg, out=dual_arg)
            else:
                proximal = gram_inv(dual_arg, out=out)
            proximal *= -tau
            proximal += x
        if out is not None:
            proximal = self._copy_vector(proximal, out=out)
        if proximal not in self.domain:
            raise ValueError(util.Errors.not_in_vecsp(
                proximal, self.domain, space_name="domain",
                add_info=f"The proximal of the functional {self} did not return an element in the domain.\n\tx = {x}"
            ))
        return proximal

    def conj_proximal(
        self,
        xstar: ReadOnlyVector,
        tau: float,
        recursion_safeguard: bool = False,
        out: Vector | None = None,
        **proximal_par: Any,
    ) -> Vector:
        r"""Evaluate the proximal mapping of the conjugate functional.
        
        If ``out`` is supplied, the result is written into that vector and the
        same object is returned. ``out`` may coincide with ``xstar``.
        
        Parameters
        ----------
        xstar: ReadOnlyVector
            Input vector; its coefficients are only read.
        tau: float
            Proximal or iteration step size.
        recursion_safeguard: bool [default: False]
            Recursion safeguard.
        out: Vector | None [default: None]
            Writable output buffer. If None, allocate a new result.
        **proximal_par: Any
            Proximal par.
        
        
        Returns
        -------
        Vector
            Writable result vector.
        """
        if xstar not in self.domain:
            raise ValueError(util.Errors.not_in_vecsp(
                xstar, self.domain, space_name="domain",
                add_info=f"Not able to compute proximal of the conjugate of {self} on the given vector."
            ))
        self._validate_out(out)
        try:
            proximal = self._conj_proximal(xstar, tau, out=out, **proximal_par)
        except NotImplementedError:
            if recursion_safeguard:
                raise NotImplementedError("Neither proximal nor conj_proximal are implemented.")
            gram = self.h_domain.gram
            gram_inv = self.h_domain.gram_inv
            primal_arg = self._copy_vector(xstar)
            primal_arg *= 1. / tau
            gram_inv(primal_arg, out=primal_arg)
            self.proximal(
                primal_arg, 1 / tau, recursion_safeguard=True,
                out=primal_arg, **proximal_par
            )
            if out is None or self._shares_memory(out, xstar):
                proximal = gram(primal_arg, out=primal_arg)
            else:
                proximal = gram(primal_arg, out=out)
            proximal *= -tau
            proximal += xstar
        if out is not None:
            proximal = self._copy_vector(proximal, out=out)
        if proximal not in self.domain:
            raise ValueError(util.Errors.not_in_vecsp(
                proximal, self.domain, space_name="domain",
                add_info=f"The proximal of the conjugate functional {self} did not return an element in the domain.\n\txstar = {xstar}"
            ))
        return proximal
    
    def as_data_func(self, 
                     data: ObsDataT | None = None
                     ) -> Functional:
        r"""Returns a modified version of the given functional as a data functional. 
        If self is already a data functional, the method returns self with an updated data vector. 
        
        Otherwise, self is made a data functional by shifting it by the given data vector. 
        None results in the zero vector. 
        Derived classes (e.g., KullbackLeibler) replace this default implementation.
        
        Parameters
        ----------
        data: ObsDataT | None [default: None]
            The data. None results in the zero vector.
        
        Returns
        --------------
        Functional
        """
        if self.is_data_func:
            if data is not None:
                self.data = data
            return self
        else:     
            return HorizontalShiftDilation(self, data = data)

    def shift(
        self, v: ReadOnlyVector | None = None, data_shift: ReadOnlyVector | None = None
    ) -> Functional:
        r"""Returns the functional :math:`x\mapsto F(x-v-data)` 
        
        Parameters
        ----------
        v: ReadOnlyVector | None [default: None]
            Input vector; its coefficients are only read.
        data_shift: ReadOnlyVector | None [default: None]
            Input vector; its coefficients are only read.
        
        
        Returns
        -------
        Functional
            Constructed functional.
        """
        return HorizontalShiftDilation(self,shift=v,data=data_shift)
    
    def dilation(self, a: float) -> Functional:
        r"""Returns the functional :math:`x\mapsto F(ax)` 
        
        Parameters
        ----------
        a: float
            A.
        
        
        Returns
        -------
        Functional
            Constructed functional.
        """
        return HorizontalShiftDilation(self,dilation=a)

    @property
    def Lipschitz(self)->float:
        r"""The Lipschitz constant of the derivative of the functional.
                
        
        Returns
        -------
        float
            Computed scalar value.
        """
        return self._Lipschitz

    @property
    def convexity_param(self)->float:
        r"""The convexity parameter of the functional.
                
        
        Returns
        -------
        float
            Computed scalar value.
        """
        return self._convexity_param

    @property
    def is_strictly_convex(self) -> bool:
        r"""Whether strict convexity on the essential domain is certified.
        
        The default uses positive strong-convexity parameter. Subclasses may
        certify strict convexity even when no positive uniform bound exists.
        False means that strict convexity is not certified by this property.
        
        
        Returns
        -------
        bool
            Whether the stated condition is satisfied.
        """
        return bool(self.convexity_param > 0)

    @property
    def conj_is_strictly_convex(self) -> bool:
        r"""Whether the conjugate is strictly convex on its essential domain.
        
        By default this is certified by a finite Lipschitz constant of the
        primal gradient. Subclasses may provide a sharper criterion.
        
        
        Returns
        -------
        bool
            Whether the stated condition is satisfied.
        """
        return bool(np.isfinite(self.Lipschitz))

    @property
    def is_convex(self)->bool:
        r"""Flag if the functional is convex.
                
        
        Returns
        -------
        bool
            Whether the stated condition is satisfied.
        """
        return self._is_convex

    @property
    def is_linear(self)->bool:
        r"""Flag if the functional is linear.
                
        
        Returns
        -------
        bool
            Whether the stated condition is satisfied.
        """
        return self._is_linear

    @property
    def is_quadratic(self)->bool:
        r"""Flag if the functional is quadratic.
                
        
        Returns
        -------
        bool
            Whether the stated condition is satisfied.
        """
        return self._is_quadratic

    @property
    def conj_is_quadratic(self)->bool:
        r"""Flag if the conjugate functional is quadratic.
                
        
        Returns
        -------
        bool
            Whether the stated condition is satisfied.
        """
        return self._conj_is_quadratic

    @property
    def is_norm_squared_type(self)->bool:
        r"""Flag if the functional behaves like a SquaredNorm functional for which the hessian is a positive multiple of the Gram matrix. 
        
        
        Returns
        -------
        bool
            Whether the stated condition is satisfied.
        """
        return False

    @property
    def is_separable(self)->bool:
        r"""Flag if the functional is separable, i.e. :math:`F(x)=sum_i f_i(x_i)`
                
        
        Returns
        -------
        bool
            Whether the stated condition is satisfied.
        """
        return self._is_separable

    @property
    def dom_l(self) -> Vector | Scalar | None:
        r"""For a separable functional, this is the pointwise lower boundary of the essential domain.
                
        
        Returns
        -------
        Vector | Scalar | None
            Result described above.
        """
        if not self.is_separable:
            raise RuntimeError(util.Errors.runtime_error('Functional not separable, therefore dom_l is not defined.'))
        elif isinstance(self._dom_l,(int,float,np.floating,np.integer)) and isinstance(self.domain,NumPyVectorSpace):
            return np.broadcast_to(float(self._dom_l),self.domain.shape)
        else:
            return self._dom_l

    @property
    def dom_u(self) -> Vector | Scalar | None:
        r"""For a separable functional, this is the pointwise upper boundary of the essential domain.
                
        
        Returns
        -------
        Vector | Scalar | None
            Result described above.
        """        
        if not self.is_separable:
            raise RuntimeError(util.Errors.runtime_error('Functional not separable, therefore dom_u is not defined.'))        
        elif isinstance(self._dom_l,(int,float,np.floating,np.integer)) and isinstance(self.domain,NumPyVectorSpace):
            return np.broadcast_to(float(self._dom_u),self.domain.shape)
        else:        
            return self._dom_u

    @property
    def conj_dom_l(self) -> Vector | Scalar | None:
        r"""For a separable functional, this is the pointwise lower boundary of the essential domain of the conjugate functional.
                
        
        Returns
        -------
        Vector | Scalar | None
            Result described above.
        """        
        if not self.is_separable:
            raise RuntimeError(util.Errors.runtime_error('Functional not separable, therefore conj_dom_l is not defined.'))                
        elif isinstance(self._conj_dom_l,(int,float,np.floating,np.integer))  and isinstance(self.domain,NumPyVectorSpace):
            return np.broadcast_to(float(self._conj_dom_l),self.domain.shape)
        else:
            return self._conj_dom_l

    @property
    def conj_dom_u(self) -> Vector | Scalar | None:
        r"""For a separable functional, this is the pointwise upper boundary of the essential domain of the conjugate functional.
                
        
        Returns
        -------
        Vector | Scalar | None
            Result described above.
        """            
        if not self.is_separable:
            raise RuntimeError(util.Errors.runtime_error('Functional not separable, therefore conj_dom_u is not defined.'))                
        elif isinstance(self._conj_dom_u,(int,float,np.floating,np.integer)) and isinstance(self.domain,NumPyVectorSpace):
            return np.broadcast_to(float(self._conj_dom_u),self.domain.shape)
        else:
            return self._conj_dom_u

    def _eval(self, x: ReadOnlyVector) -> float:
        raise NotImplementedError

    def _linearize(self, x: ReadOnlyVector) -> tuple[float, Vector]:
        raise NotImplementedError

    def _subgradient(
        self, x: ReadOnlyVector, out: Vector | None = None, **subgradient_par: Any
    ) -> Vector:
        raise NotImplementedError

    def _reference_vector(self, out: Vector | None = None) -> Vector:
        r"""Default implementation used by :meth:`reference_vector`.
        
        Parameters
        ----------
        out: Vector | None [default: None]
            Writable output buffer. If None, allocate a new result.
        
        
        Returns
        -------
        Vector
            Writable result vector.
        """
        zero = self.domain.zeros(read_only=True)
        try:
            return self.conj.subgradient(zero, out=out)
        except (NotImplementedError, NotInEssentialDomainError):
            pass

        try:
            return self.proximal(zero, 1.0, out=out)
        except NotImplementedError:
            pass

        try:
            if np.isfinite(self(zero)):
                return self._copy_vector(zero, out=out)
        except NotInEssentialDomainError:
            pass

        raise NotImplementedError(
            f"Could not construct a feasible reference vector for {self}: "
            "the conjugate subgradient at zero is unavailable or undefined, "
            "the proximal map is unavailable, and zero is not feasible."
        )

    def _conj_reference_vector(self, out: Vector | None = None) -> Vector:
        r"""Default implementation used by :meth:`conj_reference_vector`.
        
        Parameters
        ----------
        out: Vector | None [default: None]
            Writable output buffer. If None, allocate a new result.
        
        
        Returns
        -------
        Vector
            Writable result vector.
        """
        zero = self.domain.zeros(read_only=True)
        try:
            return self.subgradient(zero, out=out)
        except (NotImplementedError, NotInEssentialDomainError):
            pass

        try:
            return self.conj_proximal(zero, 1.0, out=out)
        except NotImplementedError:
            pass

        try:
            if np.isfinite(self.conj(zero)):
                return self._copy_vector(zero, out=out)
        except NotInEssentialDomainError:
            pass

        raise NotInEssentialDomainError(
            f"Could not construct a feasible conjugate reference vector for "
            f"{self}: the primal subgradient at zero is unavailable or "
            "undefined, the conjugate proximal map is unavailable, and zero "
            "is not feasible for the conjugate functional."
        )

    def _hessian(self, x: ReadOnlyVector) -> operators.Operator:
        raise NotImplementedError
    
    def _conj(self, xstar: ReadOnlyVector) -> float:
        raise NotImplementedError

    def _conj_linearize(self, xstar: ReadOnlyVector) -> tuple[float, Vector]:
        raise NotImplementedError
    
    def _conj_subgradient(
        self, xstar: ReadOnlyVector, out: Vector | None = None, **subgradient_par: Any
    ) -> Vector:
        raise NotImplementedError

    def _conj_hessian(self, xstar: ReadOnlyVector) -> operators.Operator:
        raise NotImplementedError

    def _proximal(
        self, x: ReadOnlyVector, tau: float, out: Vector | None = None, **proximal_par: Any
    ) -> Vector:
        raise NotImplementedError

    def _conj_proximal(
        self,
        xstar: ReadOnlyVector,
        tau: float,
        out: Vector | None = None,
        **proximal_par: Any,
    ) -> Vector:
        raise NotImplementedError

    def __mul__(self, other: object) -> Functional | NotImplementedType:
        if isscalar(other) and other == 1:
            return self
        elif isinstance(other, operators.Operator):
            return Composed(self, other)
        elif other in self.domain:
            return self * operators.PtwMultiplication(self.domain, other)
        return NotImplemented

    def __rmul__(self, other: object) -> Functional | NotImplementedType:
        if isscalar(other):
            if other == 1:
                return self
            elif util.is_real_dtype(other):
                return LinearCombination((other, self))
        return NotImplemented

    def __truediv__(self, other: Scalar) -> Functional:
        return (1 / other) * self

    def __add__(self, other: object) -> Functional | NotImplementedType:
        if isinstance(other, Functional):
            return LinearCombination(self, other)
        elif isscalar(other):
            return self if other==0 else VerticalShift(self, other)
        return NotImplemented

    def __radd__(self, other: object) -> Functional | NotImplementedType:
        return self+other

    def __sub__(self, other: object) -> Functional | NotImplementedType:
        return self + (-other)

    def __rsub__(self, other: object) -> Functional | NotImplementedType:
        return (-self) + other

    def __neg__(self) -> Functional:
        return (-1) * self

    def __pos__(self) -> Functional:
        return self

    @property
    def methods(self) -> set[str]:
        r"""Set of strings of the names of the methods are available for a given Functional instance. 
                
        
        Returns
        -------
        set[str]
            Result described above.
        """
        return self._methods

    @util.memoized_property
    def conj(self) -> Functional:
        r"""For linear operators, this is the adjoint as a linear `regpy.operators.Operator`
        instance. Will only be computed on demand and saved for subsequent invocations.
        
        Returns
        -------
        Functional
            The adjoint as an `regpy.operators.Operator` instance.
        """
        return Conj(self)


class _GradientAsOperator(operators.Operator):
    r"""Operator wrapper for :meth:`Functional.gradient_as_operator`.
    
    Parameters
    ----------
    func: Functional
        Functional to evaluate or transform.
    """

    def __init__(self, func: Functional) -> None:
        self.func = func
        self._hessian: operators.Operator | None = None
        super().__init__(
            func.domain,
            func.domain,
            linear=False,
            self_adjoint=True,
        )

    def _eval(
        self, x: ReadOnlyVector, differentiate: bool = False, **kwargs: Any
    ) -> Vector:
        if differentiate:
            self._hessian = self.func.hessian(x)
        return self.func.subgradient(x, **kwargs)

    def _ieval(
        self,
        x: ReadOnlyVector,
        out: Vector,
        differentiate: bool = False,
        **kwargs: Any,
    ) -> Vector:
        if differentiate:
            # Compute the Hessian before writing to out, since out may be x.
            self._hessian = self.func.hessian(x)
        return self.func.subgradient(x, out=out, **kwargs)

    def _derivative(self, x: ReadOnlyVector, **kwargs: Any) -> Vector:
        if self._hessian is None:
            raise RuntimeError(util.Errors.runtime_error(
                "The gradient operator must be linearized before its "
                "derivative can be evaluated."
            ))
        return self._hessian(x, **kwargs)

    def _iderivative(
        self, x: ReadOnlyVector, out: Vector, **kwargs: Any
    ) -> Vector:
        if self._hessian is None:
            raise RuntimeError(util.Errors.runtime_error(
                "The gradient operator must be linearized before its "
                "derivative can be evaluated."
            ))
        return self._hessian(x, out=out, **kwargs)

    def __repr__(self) -> str:
        return util.make_repr(self, self.func)


class Conj(Functional):
    r"""An proxy class wrapping a functional. Calling it will evaluate the functional's
    conj method. This class should not be instantiated directly, but rather through the
    `Functional.conj` property of a functional.
    
    Parameters
    ----------
    func: Functional
        Functional to evaluate or transform.
    """

    def __init__(self, func: Functional) -> None:
        self.func = func
        if not func.is_convex:
            self.log.warning("Taking conjugate of a non-convex functional. The biconjugate will not coincide with the primal functional.")
        """The underlying functional."""
        super().__init__(func.domain, h_domain = func.h_domain.dual_space(),
                         is_separable = func.is_separable,
                         is_convex = True
                         )         

    def _eval(self, x: ReadOnlyVector) -> float:
        return self.func._conj(x)
    # def __call__(self, x):
    #     return self.func._conj(x)

    def _conj(self, x: ReadOnlyVector) -> float:
        return self.func._eval(x)
    
    def _subgradient(
        self, x: ReadOnlyVector, out: Vector | None = None, **subgradient_par: Any
    ) -> Vector:
        return self.func.conj_subgradient(x, out=out, **subgradient_par)

    def _reference_vector(self, out: Vector | None = None) -> Vector:
        return self.func.conj_reference_vector(out=out)

    def dist_subdiff(
        self, v: ReadOnlyVector, x: ReadOnlyVector, best_approx: ReadOnlyVector | None = None
    ) -> float:
        r"""Dist subdiff.
        
        Parameters
        ----------
        v: ReadOnlyVector
            Input vector; its coefficients are only read.
        x: ReadOnlyVector
            Input vector; its coefficients are only read.
        best_approx: ReadOnlyVector | None [default: None]
            Input vector; its coefficients are only read.
        
        
        Returns
        -------
        float
            Computed scalar value.
        """
        return self.func._conj_dist_subdiff(v, x, best_approx=best_approx)

    def _conj_subgradient(
        self, x: ReadOnlyVector, out: Vector | None = None, **subgradient_par: Any
    ) -> Vector:
        return self.func.subgradient(x, out=out, **subgradient_par)

    def _conj_reference_vector(self, out: Vector | None = None) -> Vector:
        return self.func.reference_vector(out=out)

    def _conj_dist_subdiff(
        self, v: ReadOnlyVector, x: ReadOnlyVector, best_approx: ReadOnlyVector | None = None
    ) -> float:
        return self.func.dist_subdiff(v, x, best_approx=best_approx)
    
    def _hessian(self, x: ReadOnlyVector) -> operators.Operator:
        return self.func.conj_hessian(x)
    
    def _conj_hessian(self, x: ReadOnlyVector) -> operators.Operator:
        return self.func.hessian(x)
    
    def _proximal(
        self,
        x: ReadOnlyVector,
        tau: float,
        out: Vector | None = None,
        **proximal_par: Any,
    ) -> Vector:
        return self.func.conj_proximal(x, tau, out=out, **proximal_par)
    
    def _conj_proximal(
        self,
        x: ReadOnlyVector,
        tau: float,
        out: Vector | None = None,
        **proximal_par: Any,
    ) -> Vector:
        return self.func.proximal(x, tau, out=out, **proximal_par)

    def bregman_func(self, y: ReadOnlyVector, ystar: ReadOnlyVector | None = None) -> Functional:
        r"""Delegate construction to the primal's conjugate-specific hook.
        
        Parameters
        ----------
        y: ReadOnlyVector
            Input vector; its coefficients are only read.
        ystar: ReadOnlyVector | None [default: None]
            Input vector; its coefficients are only read.
        
        
        Returns
        -------
        Functional
            Constructed functional.
        """
        return self.func.conj_bregman_func(y, ystar)


    def bregman(self, x: ReadOnlyVector, y: ReadOnlyVector | None,
                ystar: ReadOnlyVector | None = None) -> float:
        r"""Bregman.
        
        Parameters
        ----------
        x: ReadOnlyVector
            Input vector; its coefficients are only read.
        y: ReadOnlyVector | None
            Input vector; its coefficients are only read.
        ystar: ReadOnlyVector | None [default: None]
            Input vector; its coefficients are only read.
        
        
        Returns
        -------
        float
            Computed scalar value.
        """
        return self.func.conj_bregman(x, y, ystar)

    @property
    def conj(self)-> Functional:
        r"""Conj.
        
        
        Returns
        -------
        Functional
            Constructed functional.
        """
        return self.func
    
    @property
    def methods(self) -> set[str]:
        r"""Methods.
        
        
        Returns
        -------
        set[str]
            Result described above.
        """
        return self.func._conj_methods

    @property
    def Lipschitz(self)->float:
        r"""Lipschitz.
        
        
        Returns
        -------
        float
            Computed scalar value.
        """
        return 1/self.func.convexity_param if self.func.convexity_param>0 else inf

    @property
    def convexity_param(self)->float:
        r"""Convexity param.
        
        
        Returns
        -------
        float
            Computed scalar value.
        """
        return 1/self.func.Lipschitz if self.func.Lipschitz>0 else inf

    @property
    def is_strictly_convex(self) -> bool:
        r"""Whether the functional is strictly convex.
        
        
        Returns
        -------
        bool
            Whether the stated condition is satisfied.
        """
        return bool(self.func.conj_is_strictly_convex)

    @property
    def conj_is_strictly_convex(self) -> bool:
        r"""Conj is strictly convex.
        
        
        Returns
        -------
        bool
            Whether the stated condition is satisfied.
        """
        return bool(self.func.is_strictly_convex)

    @property
    def is_quadratic(self) -> bool:
        r"""Is quadratic.
        
        
        Returns
        -------
        bool
            Whether the stated condition is satisfied.
        """
        return self.func.conj_is_quadratic

    @property
    def is_norm_squared_type(self) -> bool:
        r"""Whether the functional is norm squared type.
        
        
        Returns
        -------
        bool
            Whether the stated condition is satisfied.
        """
        return super().is_norm_squared_type
    
    @property
    def conj_is_quadratic(self) -> bool:
        r"""Conj is quadratic.
        
        
        Returns
        -------
        bool
            Whether the stated condition is satisfied.
        """
        return self.func.is_quadratic
    
    @property
    def dom_l(self) -> Vector | Scalar | None:
        r"""Dom l.
        
        
        Returns
        -------
        Vector | Scalar | None
            Result described above.
        """
        return self.func.conj_dom_l

    @property
    def dom_u(self) -> Vector | Scalar | None:
        r"""Dom u.
        
        
        Returns
        -------
        Vector | Scalar | None
            Result described above.
        """
        return self.func.conj_dom_u

    @property
    def conj_dom_l(self) -> Vector | Scalar | None:
        r"""Conj dom l.
        
        
        Returns
        -------
        Vector | Scalar | None
            Result described above.
        """
        return self.func.dom_l

    @property
    def conj_dom_u(self) -> Vector | Scalar | None:
        r"""Conj dom u.
        
        
        Returns
        -------
        Vector | Scalar | None
            Result described above.
        """
        return self.func.dom_u

    def __repr__(self) -> str:
        return util.make_repr(self, self.func)


class LinearFunctional(Functional):
    r"""Linear functionals
    Linear functional given by
    
    .. math::
        F(x) = \langel a, x\rangle
    
    The operators `__add__` , `__iadd__` , `__mul__` , `__imul__` with `LinearFunctional`\s and 
    scalars as other arguments, rsp, are overwritten to yield the expected `LinearFunctional`\s. 
    
    Parameters
    ----------
    gradient: ReadOnlyVector
        The gradient of the linear functional. :math:`a=gradient` if gradient_in_dual_space == True
    domain: vecsps.VectorSpaceBase | None [default: None]
        The `regpy.vecsps.VectorSpaceBase` on which the functional is defined
    h_domain: hilbert.HilbertSpace | None [default: None]
        Hilbert space for proximity operator
    gradient_in_dual_space: bool [default: False]
        If false, the argument gradient is considered as an element of the primal space, 
        and :math:`a = h_domain.gram(gradient)`.
    """
    def __init__(
        self,
        gradient: ReadOnlyVector,
        domain: vecsps.VectorSpaceBase | None = None,
        h_domain: hilbert.HilbertSpace | None = None,
        gradient_in_dual_space: bool = False,
    ) -> None:
        if domain is None and isinstance(gradient,np.ndarray):
            domain = vecsps.NumPyVectorSpace(shape=gradient.shape,dtype=float)
        elif gradient not in domain:
            raise ValueError(util.Errors.not_in_vecsp(gradient,domain,vec_name="gradient",space_name="domain"))
        if h_domain is None:
            h_domain = hilbert.as_hilbert_space(h_domain,domain) or hilbert.L2(domain)
        elif not isinstance(h_domain,hilbert.HilbertSpace) and h_domain.vecsp == domain:
            raise ValueError(util.Errors.not_instance(h_domain,hilbert.HilbertSpace,add_info="The given h_domain has to be a HilbertSpace with matching domain to the domain."))
        if gradient_in_dual_space:
            self._gradient = gradient
        else:
            self._gradient = domain.empty()
            h_domain.gram(gradient, out=self._gradient)
        super().__init__(domain=domain,h_domain=h_domain,
                         is_linear=True, is_quadratic=True, conj_is_quadratic=False,
                         Lipschitz = 0,
                         is_separable=True,
                         is_convex = True,
                         dom_l=-inf, dom_u = inf,
                         conj_dom_l = self._gradient, conj_dom_u = self._gradient,
                         methods = {'eval','subgradient','hessian','proximal','dist_subdiff'},
                         conj_methods= {'eval','subgradient','proximal','dist_subdiff'}
                         ) 

    def _eval(self, x: ReadOnlyVector) -> float:
        return self.domain.vdot(self._gradient,x).real

    @property
    def gradient(self) -> Vector:
        r"""Gradient.
        
        
        Returns
        -------
        Vector
            Writable result vector.
        """
        gradient = self.domain.empty()
        self.domain.copyto(gradient, self._gradient)
        return gradient

    def _subgradient(
        self, x: ReadOnlyVector, out: Vector | None = None, **subgradient_par: Any
    ) -> Vector:
        if out is None:
            gradient = self.domain.empty()
            self.domain.copyto(gradient, self._gradient)
            return gradient
        return self._copy_vector(self._gradient, out=out)

    def _hessian(self, x: ReadOnlyVector) -> operators.Operator:
        return operators.Zero(self.domain)

    def _conj(self, x_star: ReadOnlyVector) -> float:
        return 0 if self.domain.norm(x_star- self._gradient)==0 else inf

    def _conj_subgradient(
        self, xstar: ReadOnlyVector, out: Vector | None = None, **subgradient_par: Any
    ) -> Vector:
        if self.domain.norm(xstar - self._gradient) == 0:
            if out is None:
                return self.domain.zeros()
            self.domain.fill(out, 0)
            return out
        raise NotInEssentialDomainError('LinearFunctional.conj')

    def _conj_dist_subdiff(
        self, v: ReadOnlyVector, xstar: ReadOnlyVector, best_approx: Vector | None = None
    ) -> float:
        self._validate_out(best_approx)
        if self.domain.norm(xstar - self._gradient) == 0:
            if best_approx is not None:
                self._copy_vector(v, out=best_approx)
            return 0.
        else:
            raise NotInEssentialDomainError('LinearFunctional.conj')          

    def _proximal(
        self,
        x: ReadOnlyVector,
        tau: float,
        out: Vector | None = None,
        **proximal_par: Any,
    ) -> Vector:
        gradient = self._gradient
        if out is not None and self._shares_memory(out, gradient):
            gradient = self._copy_vector(gradient)
        result = self._copy_vector(x, out=out)
        scaled_gradient = self._copy_vector(gradient)
        scaled_gradient *= tau
        result -= scaled_gradient
        return result

    def _conj_proximal(
        self,
        xstar: ReadOnlyVector,
        tau: float,
        out: Vector | None = None,
        **proximal_par: Any,
    ) -> Vector:
        if out is None:
            gradient = self.domain.empty()
            self.domain.copyto(gradient, self._gradient)
            return gradient
        return self._copy_vector(self._gradient, out=out)

    def dilation(self, a: float) -> LinearFunctional:
        r"""Dilation.
        
        Parameters
        ----------
        a: float
            A.
        
        
        Returns
        -------
        LinearFunctional
            Result described above.
        """
        gradient = self.gradient
        gradient *= a
        return LinearFunctional(
            gradient, domain=self.domain, h_domain=self.h_domain,
            gradient_in_dual_space=True
        )
    
    def shift(
        self, v: ReadOnlyVector | None = None, data_shift: ReadOnlyVector | None = None
    ) -> Functional:
        r"""Shift.
        
        Parameters
        ----------
        v: ReadOnlyVector | None [default: None]
            Input vector; its coefficients are only read.
        data_shift: ReadOnlyVector | None [default: None]
            Input vector; its coefficients are only read.
        
        
        Returns
        -------
        Functional
            Constructed functional.
        """
        if(data_shift is None):
            return self - self.domain.vdot(self._gradient,v).real
        #Linear functional cannot be data functional at the moment
        return HorizontalShiftDilation(self-self.domain.vdot(self._gradient,v).real,data=data_shift)

    def __add__(self, other: object) -> Functional | NotImplementedType:
        # Import at call time to avoid a base/integral module import cycle.
        from .integral_functionals import Quadratic
        if isinstance(other, Quadratic):
            return other + self
        if isinstance(other,LinearFunctional):
            gradient = self.gradient
            gradient += other._gradient
            return LinearFunctional(
                gradient, domain=self.domain, h_domain=self.h_domain,
                gradient_in_dual_space=True
            )
        elif other in self.domain:
            gradient = self.gradient
            gradient += other
            return LinearFunctional(
                gradient, domain=self.domain, h_domain=self.h_domain,
                gradient_in_dual_space=True
            )
        elif isinstance(other,SquaredNorm):
            return other+self
        else:
            return super().__add__(other)

    def __iadd__(self, other: object) -> LinearFunctional | NotImplementedType:
        if isinstance(other, LinearFunctional):
            self._gradient += other._gradient
            return self
        else:
            return NotImplemented
        
    def __rmul__(self, other: object) -> LinearFunctional | NotImplementedType:
        if isscalar(other):
            gradient = self.gradient
            gradient *= other
            return LinearFunctional(
                gradient, domain=self.domain, h_domain=self.h_domain,
                gradient_in_dual_space=True
            )
        else:
            return NotImplemented

    def __imul__(self, other: object) -> LinearFunctional | NotImplementedType:
        if isscalar(other):
            self._gradient *= other
            return self
        else:
            return NotImplemented

class NullFunctional(LinearFunctional):
    r"""Functional which is identically zero
    
    Parameters
    domain: VectorSpaceBase
    
    Parameters
    ----------
    domain: VectorSpaceBase
        Domain vector space.
    """
    def __init__(self, domain: VectorSpaceBase) -> None:
        zeros = np.broadcast_to([0.],domain.shape) if isinstance(domain,NumPyVectorSpace) else domain.zeros()
        super().__init__(zeros,domain=domain)

    def _proximal(self, x: ReadOnlyVector, tau: float, out: Vector | None = None, **proximal_par: Any) -> Vector:
        out = self.domain.empty() if out is None else out
        if out is not x:
            self.domain.copyto(out,x)
        return out

class SquaredNorm(Functional):
    r"""Functionals of the form 
    
    .. math::
        \mathcal{F}(x) = \frac{a}{2}\|x\|_X^2 +\langle b,x\rangle_X + c
    
    Here the linear term represents an inner product in the Hilbert space, not a pairing with the dual space.
    
    The operators `__add__` , `__iadd__` , `__mul__` , `__imul__` with `SquaredNorm`\s, `LinearFunctional`\s and scalars 
    as other arguments are overwritten to yield the expected `SquaredNorm`\s. 
    
    Parameters
    ----------
    h_space: HilbertSpace
        The underlying Hilbert space.
    a: float [default: 1.0]
        coefficient of quadratic term
    b: ReadOnlyVector | None [default: None]
        Coefficient of linear term -- in this case, it must be In h_space.domain. In the default case be is set to 0.
    c: float [default: 0.0]
        constant term
    shift: ReadOnlyVector | None [default: None]
        If not None, then it must be in h_space.domain, and we must have b is None and c==0. 
        In this case the functional is initialized as :math:`\mathcal{F}(x) = \frac{a}{2}\|x-shift-data\|^2`.
    data: ReadOnlyVector | None [default: None]
        If not None, then we must have b is None and c==0. 
        In this case the functional is initialized as :math:`\mathcal{F}(x) = \frac{a}{2}\|x-shift-data\|^2`.
    """

    def __init__(
        self,
        h_space: HilbertSpace,
        a: float = 1.0,
        b: ReadOnlyVector | None = None,
        c: float = 0.0,
        shift: ReadOnlyVector | None = None,
        data: ReadOnlyVector | None = None,
    ) -> None:
        from regpy.operators import PtwMultiplication, Identity
        from regpy.operators.base import LinearCombination as LinComb 
        if  isinstance(h_space.gram, (PtwMultiplication,Identity)):
            separable = True
        elif isinstance(h_space.gram,LinComb):
            separable = (len(h_space.gram.ops)==1 and isinstance(h_space.gram.ops[0],(PtwMultiplication,Identity)))
        else:
            separable = False    
        super().__init__(h_space.vecsp,h_domain=h_space, 
                        is_linear = (a==0 and shift is None and c==0),
                        is_quadratic = True,
                        is_separable = separable,
                        dom_l = -np.inf,
                        dom_u = np.inf,
                        methods = {'eval','subgradient','hessian','proximal','dist_subdiff'},
                        conj_methods= {'eval','subgradient','hessian','proximal','dist_subdiff'}
                        )
        if not isinstance(a,(float,int)): raise ValueError(util.Errors.not_instance(a,float,add_info="for SquaredNorm `a` has to be a scalar!"))
        self._a=float(a)
        if shift is None:
            if b is None:
                if isinstance(self.domain,vecsps.NumPyVectorSpace):
                    self._b = np.broadcast_to(np.zeros(()),self.domain.shape)
                else:
                    self._b = self.domain.zeros()
            elif b not in self.domain: 
                raise ValueError(util.Errors.not_in_vecsp(b,self.domain,add_info=f"b not in domain of functional {self}."))
            else:
                self._b = b
            if not isinstance(c,(float,int)): raise ValueError(util.Errors.not_instance(c,float,add_info="for SquaredNorm `c` has to be a scalar!"))
            self._c = float(c)
        else:
            if b is not None or c!=0:
                raise ValueError(util.Errors.generic_message("If shift is given b and c cannot be set."))
            if shift in self.domain:
                self._b = -self.a*shift
                self._c = (self.a/2.) * self.h_domain.norm(shift)**2
            else:
                raise ValueError(util.Errors.not_in_vecsp(shift,self.domain, add_info=f"Shift not in domain of functional {self}.")) 
        self.data = data

    @property
    def dom_l(self) -> Vector | Scalar:
        """Pointwise lower bound of the full effective domain."""
        if isinstance(self.domain, NumPyVectorSpace):
            return np.broadcast_to(-np.inf, self.domain.shape)
        return -np.inf * self.domain.ones()

    @property
    def dom_u(self) -> Vector | Scalar:
        """Pointwise upper bound of the full effective domain."""
        if isinstance(self.domain, NumPyVectorSpace):
            return np.broadcast_to(np.inf, self.domain.shape)
        return np.inf * self.domain.ones()

    @property
    def gram(self)->Operator:
        r"""Return the Gram operator representing the Hilbert inner product.
        
        
        Returns
        -------
        Operator
            Constructed operator.
        """
        return self.h_domain.gram

    @property
    def gram_inv(self) -> Operator | None:
        r"""Gram inv.
        
        
        Returns
        -------
        Operator | None
            Result described above.
        """
        try:
            gram_inv = self.h_domain.gram_inv
        except NotImplementedError:
            gram_inv = None
            self.log.warning("The inverse of the gram operator of h_space is not implemented. ")
        return gram_inv

    @util.memoized_property
    def a(self)->float:
        r"""A.
        
        
        Returns
        -------
        float
            Computed scalar value.
        """
        return self._a

    @util.memoized_property
    def b(self) -> Vector:
        r"""B.
        
        
        Returns
        -------
        Vector
            Writable result vector.
        """
        if self.is_data_func:
            return self._b-self._a*self.data
        else:
            return self._b

    @util.memoized_property
    def c(self)->float:
        r"""C.
        
        
        Returns
        -------
        float
            Computed scalar value.
        """
        if self.is_data_func:
            return self._c - self.h_domain.inner(self._b,self.data).real + (self._a/2)* self.h_domain.norm(self.data)**2
        else:
            return self._c

    @property
    def data(self) -> ReadOnlyVector:
        r"""Data.
        
        
        Returns
        -------
        ReadOnlyVector
            Read-only vector view.
        """
        return self.domain.read_only(self._data)
    
    @data.setter
    def data(self, new_data: ObsDataT | None) -> None:
        r"""Data.
        
        Parameters
        ----------
        new_data: ObsDataT | None
            New data.
        """
        if new_data is None:
            self.is_data_func = False
            del self.a; del self.b; del self.c
        elif new_data in self.domain:
            self.is_data_func = True
            self._data = new_data
            del self.a; del self.b; del self.c
        else:
            raise ValueError(util.Errors.not_in_vecsp(new_data,self.domain,vec_name="new data vector",space_name="domain of functional"))
        
    @data.deleter
    def data(self) -> None:
        r"""Data.
        """
        if self.is_data_func:
            del self._data
            del self.a; del self.b; del self.c
            self.is_data_func = False

    @property
    def Lipschitz(self) -> float:
        r"""Lipschitz.
        
        
        Returns
        -------
        float
            Computed scalar value.
        """
        return np.abs(self.a)
    
    @property
    def convexity_param(self) -> float:
        r"""Convexity param.
        
        
        Returns
        -------
        float
            Computed scalar value.
        """
        return max([self.a,0])

    @property
    def is_convex(self) -> bool:
        r"""Whether the functional is convex.
        
        
        Returns
        -------
        bool
            Whether the stated condition is satisfied.
        """
        return (self.a>=0)

    @property
    def conj_is_quadratic(self) -> bool:
        r"""Conj is quadratic.
        
        
        Returns
        -------
        bool
            Whether the stated condition is satisfied.
        """
        return (self.a>0) 

    @property
    def is_norm_squared_type(self) -> bool: 
        r"""Whether the functional is norm squared type.
        
        
        Returns
        -------
        bool
            Whether the stated condition is satisfied.
        """
        return self.a>0 

    @property
    def conj_dom_l(self) -> Vector | Scalar | None:
        r"""Conj dom l.
        
        
        Returns
        -------
        Vector | Scalar | None
            Result described above.
        """
        self._conj_dom_l=None
        if self.is_separable:
            self._conj_dom_l = -np.inf if self.a>0 else self.h_space.gram(self.b)
        return super().conj_dom_l        

    @property
    def conj_dom_u(self) -> Vector | Scalar | None:
        r"""Conj dom u.
        
        
        Returns
        -------
        Vector | Scalar | None
            Result described above.
        """
        self._conj_dom_u=None
        if self.is_separable:
            self._conj_dom_u = np.inf if self.a>0 else self.h_space.gram(self.b)
        return super().conj_dom_u            

    def as_data_func(self, data: ObsDataT) -> SquaredNorm:
        r"""As data func.
        
        Parameters
        ----------
        data: ObsDataT
            Observed data.
        
        
        Returns
        -------
        SquaredNorm
            Result described above.
        """
        self.data=data
        return self

    def bregman_func(self, y: ReadOnlyVector, ystar: ReadOnlyVector | None = None) -> Functional:
        r"""Return a SquaredNorm representing a/2 * ||x-y||_H**2.
        
        For convex SquaredNorms, affine coefficients and stored data cancel.
        
        Parameters
        ----------
        y: ReadOnlyVector
            Input vector; its coefficients are only read.
        ystar: ReadOnlyVector | None [default: None]
            Input vector; its coefficients are only read.
        
        
        Returns
        -------
        Functional
            Constructed functional.
        """
        if not self.is_convex:
            return super().bregman_func(y, ystar)
        y, _, _ = self._bregman_reference(y, ystar)
        return SquaredNorm(self.h_domain, a=self.a, shift=y)


    def conj_bregman_func(self, y: ReadOnlyVector, ystar: ReadOnlyVector | None = None) -> Functional:
        r"""Use the dual Hilbert norm for the conjugate of a positive quadratic.
        
        Parameters
        ----------
        y: ReadOnlyVector
            Input vector; its coefficients are only read.
        ystar: ReadOnlyVector | None [default: None]
            Input vector; its coefficients are only read.
        
        
        Returns
        -------
        Functional
            Constructed functional.
        """
        if not self.is_convex or self.a <= 0:
            return super().conj_bregman_func(y, ystar)
        y, _, _ = self.conj._bregman_reference(y, ystar)
        return SquaredNorm(self.h_domain.dual_space(), a=1. / self.a, shift=y)


    def _eval(self, x: ReadOnlyVector) -> float:
        return (self.a/2.) * self.h_domain.inner(x,x).real  + self.h_domain.inner(self.b,x).real + self.c
    
    def _subgradient(
        self, x: ReadOnlyVector, out: Vector | None = None, **subgradient_par: Any
    ) -> Vector:
        b = self.b
        if out is not None and self._shares_memory(out, b):
            b = self._copy_vector(b)
        if self.a == 0:
            argument = self._copy_vector(b, out=out)
        else:
            argument = self._copy_vector(x, out=out)
            argument *= self.a
            argument += b
        return self.gram(argument, out=argument)
    
    def _hessian(self, x: ReadOnlyVector) -> operators.Operator:
        return self.a * self.gram
    
    def _proximal(
        self,
        z: ReadOnlyVector,
        tau: float,
        out: Vector | None = None,
        **proximal_par: Any,
    ) -> Vector:
        if self.a <= 0:
            raise NotImplementedError(util.Errors.generic_message(
                f"The prox operator for the SquaredNorm functional {self} is not implemented for a = {self.a}. a<=0"
            ))
        b = self.b
        if out is not None and self._shares_memory(out, b):
            b = self._copy_vector(b)
        result = self._copy_vector(z, out=out)
        scaled_b = self._copy_vector(b)
        scaled_b *= tau
        result -= scaled_b
        result *= 1. / (tau * self.a + 1)
        return result
    
    def _conj(self, xstar: ReadOnlyVector) -> float:
        bstar = self.domain.empty()
        self.gram(self.b, out=bstar)
        residual = self._copy_vector(xstar)
        residual -= bstar
        if self.a>0:
            if self.gram_inv is None:
                raise RuntimeError("The inverse of the gram operator is not implemented. Thus not allowing an application of the conjugate functional.")
            self.gram_inv(residual, out=bstar)
            return self.h_domain.vecsp.vdot(
                residual, bstar
            ).real / (2. * self.a) - self.c
        elif self.a==0:
            eps = 1e-10
            return -self.c if self.domain.norm(residual)<=eps*(self.domain.norm(xstar)+eps) else inf
        else:
            return -inf

    def _conj_subgradient(
        self, xstar: ReadOnlyVector, out: Vector | None = None, **subgradient_par: Any
    ) -> Vector:
        argument = self.domain.empty()
        self.gram(self.b, out=argument)
        argument *= -1
        argument += xstar
        if self.a > 0:
            if self.gram_inv is None:
                raise RuntimeError("The inverse of the gram operator is not implemented. Thus not allowing an application of the conjugate subgradient functional.")
            result = self.gram_inv(
                argument, out=argument if out is None else out
            )
            result *= 1. / self.a
            return result
        if self.a == 0:
            if out is None:
                return self.domain.zeros()
            self.domain.fill(out, 0)
            return out
        raise NotInEssentialDomainError

    def _conj_dist_subdiff(
        self, v: ReadOnlyVector, xstar: ReadOnlyVector, best_approx: Vector | None = None
    ) -> float:
        self._validate_out(best_approx)
        if self.a==0:
            if self.domain.norm(xstar - self.gram(self.b)) == 0:
                if best_approx is not None:
                    self._copy_vector(v, out=best_approx)
                return 0.
            else:
                raise NotInEssentialDomainError('SquaredNorm.conj')
        elif self.a <0:
            raise RuntimeError("Can't evaluate conjugate and its subdifferential if a<0.")
        else: 
            return super()._conj_dist_subdiff(v, xstar, best_approx=best_approx)

    def _conj_hessian(self, xstar: ReadOnlyVector) -> operators.Operator:
        if self.a>0:
            if self.gram_inv is None:
                raise RuntimeError("The inverse of the gram operator is not implemented. Thus not allowing an application of the conjugate hessian functional.")
            return (1./self.a) * self.gram_inv
        else:
            return NotTwiceDifferentiableError
    
    def _conj_proximal(
        self,
        zstar: ReadOnlyVector,
        tau: float,
        out: Vector | None = None,
        **proximal_par: Any,
    ) -> Vector:
        if self.a <= 0:
            raise NotImplementedError(util.Errors.generic_message(
                f"The prox operator for the conjugate of the SquaredNorm functional {self} is not implemented for a = {self.a}. it has to satisfy a>0"
            ))
        bstar = self.domain.empty()
        self.gram(self.b, out=bstar)
        result = self._copy_vector(zstar, out=out)
        result -= bstar
        result *= 1. / (1. + tau / self.a)
        result += bstar
        return result

    def dilation(self, dil: float) -> SquaredNorm:
        r"""Dilation.
        
        Parameters
        ----------
        dil: float
            Dil.
        
        
        Returns
        -------
        SquaredNorm
            Result described above.
        """
        return SquaredNorm(self.h_domain,
                               a = dil**2 *self.a,
                               b = dil*self.b,
                               c = self.c 
                               )

    def shift(
        self, v: ReadOnlyVector | None = None, data_shift: ReadOnlyVector | None = None
    ) -> SquaredNorm:
        r"""Return x -> self(x - v - data_shift), preserving existing data.
        
        Neither the supplied shifts nor the original functional are modified.
        The result remains a SquaredNorm with intrinsic coefficients and data
        stored separately.
        
        Parameters
        ----------
        v: ReadOnlyVector | None [default: None]
            Input vector; its coefficients are only read.
        data_shift: ReadOnlyVector | None [default: None]
            Input vector; its coefficients are only read.
        
        
        Returns
        -------
        SquaredNorm
            Result described above.
        """
        if v is not None and v not in self.domain:
            raise ValueError(util.Errors.not_in_vecsp(
                v, self.domain, vec_name="shift vector", space_name="domain of functional"))
        if data_shift is not None and data_shift not in self.domain:
            raise ValueError(util.Errors.not_in_vecsp(
                data_shift, self.domain, vec_name="shift data vector", space_name="domain of functional"))
        data = self._copy_vector(self.data) if self.is_data_func else None
        if data_shift is not None:
            if data is None:
                data = self._copy_vector(data_shift)
            else:
                data += data_shift
        if v is None:
            return SquaredNorm(self.h_domain, a=self._a,
                               b=self._copy_vector(self._b), c=self._c, data=data)
        return SquaredNorm(
            self.h_domain, a=self._a, b=self._b - self._a * v,
            c=self._c - self.h_domain.inner(self._b, v).real
              + (self._a / 2) * self.h_domain.norm(v)**2,
            data=data,
        )

    def _reference_vector(self, out=None):
        if self.a > 0:
            return self._copy_vector((-1./ self.a) * self.b, out=out)
        return super()._reference_vector(out=out)    

    def __add__(self, other: object) -> Functional | NotImplementedType:
        if isinstance(other, SquaredNorm) and other.h_domain==self.h_domain:
            return SquaredNorm(self.h_domain,
                               a = self.a+other.a,
                               b = self.b+other.b,
                               c = self.c+other.c 
                               )
        elif isinstance(other,LinearFunctional) and other.h_domain==self.h_domain:
            if self.gram_inv is None:
                raise RuntimeError("The inverse of the gram operator is not implemented. Thus not allowing an addition with a LinearFunctional.")
            b = self._copy_vector(self.b)
            dual_gradient = self.domain.empty()
            self.gram_inv(other._gradient, out=dual_gradient)
            b += dual_gradient
            return SquaredNorm(self.h_domain,
                               a = self.a,
                               b = b,
                               c = self.c 
                               )
        elif isscalar(other):
            return SquaredNorm(self.h_domain,
                               a = self.a,
                               b = self.b,
                               c = self.c+other 
                               )
        return super().__add__(other)

    def __iadd__(self, other: object) -> SquaredNorm | NotImplementedType:
        if isinstance(other, SquaredNorm) and other.h_domain==self.h_domain:
            self._a += other.a
            self._b += other.b
            self._c += other.c
            del self.a; del self.b; del self.c
            return self
        elif isinstance(other,LinearFunctional) and other.h_domain==self.h_domain:
            if self.gram_inv is None:
                raise RuntimeError("The inverse of the gram operator is not implemented. Thus not allowing an addition with a LinearFunctional.")
            dual_gradient = self.domain.empty()
            self.gram_inv(other._gradient, out=dual_gradient)
            self._b += dual_gradient
            del self.b
            return self
        elif isscalar(other):
            self._c += other
            del self.c
            return self
        return NotImplemented

    def __rmul__(self, other: object) -> SquaredNorm | NotImplementedType:
        if isscalar(other):
            return SquaredNorm(self.h_domain,
                               a = other*self.a,
                               b = other*self.b,
                               c = other*self.c 
                               )
        else:
            return NotImplemented

    def __imul__(self, other: object) -> SquaredNorm | NotImplementedType:
        if isscalar(other):
            self._a *=other
            self._b *=other
            self._c *=other
            del self.a; del self.b; del self.c
            return self
        return NotImplemented


class LinearCombination(Functional):
    r"""A flattened sum ``offset + linear_part + sum(c_i * funcs[i])``.
    
    Parameters
    ----------
    offset: float [default: 0.0]
        Finite additive constant, default zero.
    *args: Functional | tuple[float, Functional]
        Functionals with optional real coefficients. Nested combinations
        (including VerticalShift) are flattened and their weighted offsets
        accumulated. Repeated functionals are combined and zero terms removed.
        At least one functional is required to determine the domain.
    
    Attributes
    ----------
    funcs, coeffs : list
        Nonlinear components and their nonzero coefficients only.
    linear_part : LinearFunctional
        Sum of all linear components, stored as a dual covector. It is the
        zero functional when no linear components are present.
    offset : float
        Accumulated additive constant.
    
    Notes
    -----
    For ``G(x) = a*F(x) + <b,x> + c`` with a > 0, the conjugate is
    ``G*(p) = a*F*((p-b)/a) - c``. Conjugate operations and proximal maps
    delegate to F when available. Purely affine sums are handled directly.
    With several nonlinear terms, evaluation and derivatives are available
    when implemented by all terms; no general proximal/conjugate formula exist. 
    Negative nonlinear coefficients retain primal evaluation and
    derivatives, but do not enable these convex conjugate/proximal formulas.
    The Hilbert geometry is that of the first nonlinear term (or the first
    argument for affine sums). A proximal map inherited from a nonlinear
    term is therefore evaluated in that term's own metric.
    """

    def __init__(self, *args: Functional | tuple[float, Functional],
                 offset: float = 0.0) -> None:
        if not isinstance(offset, (int, float, np.integer, np.floating)) or not np.isfinite(offset):
            raise ValueError('offset must be a finite real scalar.')
        if not args:
            raise ValueError('At least one functional is required to determine the domain.')
        terms = defaultdict(float)
        domain = None
        first = None
        total_offset = float(offset)

        def collect(coeff: ReadOnlyVector, func: Functional) -> None:
            r"""Collect.
            
            Parameters
            ----------
            coeff: ReadOnlyVector
                Input vector; its coefficients are only read.
            func: Functional
                Functional to evaluate or transform.
            """
            nonlocal domain, first, total_offset
            if not isinstance(func, Functional):
                raise TypeError('Each component must be a Functional.')
            if not isinstance(coeff, (int, float, np.integer, np.floating)) or not np.isfinite(coeff):
                raise ValueError('Coefficients must be finite real scalars.')
            if domain is None:
                domain, first = func.domain, func
            elif func.domain != domain:
                raise ValueError('All functionals must have the same domain.')
            if coeff == 0:
                return
            if isinstance(func, LinearCombination):
                total_offset += coeff * func.offset
                collect(coeff, func.linear_part)
                for c, f in zip(func.coeffs, func.funcs):
                    collect(coeff * c, f)
            else:
                terms[func] += coeff

        for arg in args:
            if isinstance(arg, tuple):
                if len(arg) != 2:
                    raise ValueError('Weighted components must be (coefficient, functional) pairs.')
                collect(*arg)
            else:
                collect(1., arg)
        self.offset = float(total_offset)
        self.funcs, self.coeffs = [], []
        grad = domain.zeros()
        for func, coeff in terms.items():
            if coeff == 0:
                continue
            if func.is_linear:
                grad += coeff * func.subgradient(domain.zeros())
            else:
                self.funcs.append(func)
                self.coeffs.append(float(coeff))
        h_domain = self.funcs[0].h_domain if self.funcs else first.h_domain
        self.linear_part = LinearFunctional(
            grad, domain=domain, h_domain=h_domain, gradient_in_dual_space=True)
        # Retained as a compatibility alias; it is a covector, not a Hilbert gradient.
        self.grad_sum = grad
        self.linear_table = [False] * len(self.funcs)
        if self.funcs:
            methods = set.intersection(*(set(f.methods) for f in self.funcs))
            methods -= {'proximal', 'dist_subdiff'}
            conj_methods = set()
            if len(self.funcs) == 1 and self.coeffs[0] > 0 and self.funcs[0].is_convex:
                methods |= self.funcs[0].methods & {'proximal', 'dist_subdiff'}
                conj_methods = set(self.funcs[0].conj.methods)
        else:
            methods = {'eval', 'subgradient', 'hessian', 'proximal', 'dist_subdiff'}
            conj_methods = {'eval', 'subgradient', 'proximal', 'dist_subdiff'}
        super().__init__(
            domain, h_domain=h_domain,
            is_linear=not self.funcs and self.offset == 0,
            is_separable=all(f.is_separable for f in self.funcs),
            methods=methods, conj_methods=conj_methods,
        )

    def _eval(self, x: ReadOnlyVector, **kwargs: Any) -> float:
        return (self.offset + self.linear_part(x)
                + sum(c * f(x, **kwargs) for c, f in zip(self.coeffs, self.funcs)))

    def _linearize(self, x: ReadOnlyVector, **kwargs: Any) -> tuple[float, Vector]:
        value = self.offset + self.linear_part(x)
        gradient = self._copy_vector(self.grad_sum)
        for c, f in zip(self.coeffs, self.funcs):
            val, grad = f.linearize(x, **kwargs)
            value += c * val
            gradient += c * grad
        return value, gradient

    def _subgradient(self, x: ReadOnlyVector, out: Vector | None = None,
                     **subgradient_par: Any) -> Vector:
        # Accumulate separately so out may alias x or a stored covector.
        gradient = self._copy_vector(self.grad_sum)
        for c, f in zip(self.coeffs, self.funcs):
            gradient += c * f.subgradient(x, **subgradient_par)
        return self._copy_vector(gradient, out=out)

    def _hessian(self, x: ReadOnlyVector, **kwargs: Any) -> operators.Operator:
        if not self.funcs:
            return operators.Zero(self.domain)
        if len(self.funcs) == 1:
            return self.coeffs[0] * self.funcs[0].hessian(x, **kwargs)
        from regpy.operators.base import LinearCombination as OperatorLinearCombination
        return OperatorLinearCombination(
            *((c, f.hessian(x, **kwargs)) for c, f in zip(self.coeffs, self.funcs)))

    def _single_convex_term(self) -> None | tuple[Any, Any]:
        if not self.funcs:
            return None
        if len(self.funcs) != 1 or self.coeffs[0] <= 0 or not self.funcs[0].is_convex:
            raise NotImplementedError('Requires at most one nonlinear term with positive coefficient and convex functional.')
        return self.coeffs[0], self.funcs[0]

    def _conj_argument(self, xstar: ReadOnlyVector, coeff: ReadOnlyVector) -> Vector:
        argument = self._copy_vector(xstar)
        argument -= self.grad_sum
        argument /= coeff
        return argument

    def _proximal(self, x: ReadOnlyVector, tau: float, out: Vector | None = None,
                  **proximal_par: Any) -> Vector:
        term = self._single_convex_term()
        argument = self._copy_vector(x)
        argument -= tau * self.h_domain.gram_inv(self.grad_sum)
        if term is None:
            return self._copy_vector(argument, out=out)
        coeff, func = term
        return func.proximal(argument, coeff * tau, out=out, **proximal_par)

    def _conj(self, xstar: ReadOnlyVector, **kwargs: Any) -> float:
        term = self._single_convex_term()
        if term is None:
            return -self.offset if self.domain.norm(xstar - self.grad_sum) == 0 else inf
        coeff, func = term
        return coeff * func.conj(self._conj_argument(xstar, coeff), **kwargs) - self.offset

    def _conj_subgradient(self, xstar: ReadOnlyVector, out: Vector | None = None,
                          **subgradient_par: Any) -> Vector:
        term = self._single_convex_term()
        if term is None:
            return self.linear_part.conj_subgradient(xstar, out=out, **subgradient_par)
        coeff, func = term
        return func.conj_subgradient(self._conj_argument(xstar, coeff),
                                    out=out, **subgradient_par)

    def _conj_hessian(self, xstar: ReadOnlyVector, **kwargs: Any) -> operators.Operator:
        term = self._single_convex_term()
        if term is None:
            raise NotTwiceDifferentiableError('The conjugate of an affine functional is a singleton indicator.')
        coeff, func = term
        return (1. / coeff) * func.conj.hessian(self._conj_argument(xstar, coeff), **kwargs)

    def _conj_proximal(self, xstar: ReadOnlyVector, tau: float,
                       out: Vector | None = None, **proximal_par: Any) -> Vector:
        term = self._single_convex_term()
        if term is None:
            return self._copy_vector(self.grad_sum, out=out)
        coeff, func = term
        argument = self._conj_argument(xstar, coeff)
        result = func.conj_proximal(argument, tau / coeff, **proximal_par)
        # Copy before scaling; underlying functionals may return shared storage.
        result = self._copy_vector(result)
        result *= coeff
        result += self.grad_sum
        return self._copy_vector(result, out=out)

    def dist_subdiff(
        self, vstar: ReadOnlyVector, x: ReadOnlyVector, best_approx: Vector | None = None,
        **kwargs: Any,
    ) -> float:
        r"""Distance to a*subdiff F(x)+b; optionally write a*projection+b.
        
        Parameters
        ----------
        vstar: ReadOnlyVector
            Input vector; its coefficients are only read.
        x: ReadOnlyVector
            Input vector; its coefficients are only read.
        best_approx: Vector | None [default: None]
            Writable vector used by this operation.
        **kwargs: Any
            Additional keyword arguments forwarded to the underlying implementation.
        
        
        Returns
        -------
        float
            Computed scalar value.
        """
        self._validate_out(best_approx)
        if not self.funcs:
            distance = self.h_domain.dual_space().norm(vstar - self.grad_sum)
            if best_approx is not None:
                self._copy_vector(self.grad_sum, out=best_approx)
            return distance
        term = self._single_convex_term()
        coeff, func = term
        shift = self.grad_sum
        if best_approx is not None and self._shares_memory(best_approx, shift):
            shift = self._copy_vector(shift)
        distance = coeff * func.dist_subdiff(
            self._conj_argument(vstar, coeff), x,
            best_approx=best_approx, **kwargs)
        if best_approx is not None:
            best_approx *= coeff
            best_approx += shift
        return distance

    def _conj_dist_subdiff(
        self, v: ReadOnlyVector, xstar: ReadOnlyVector, best_approx: ReadOnlyVector | None = None,
        **kwargs: Any,
    ) -> float:
        r"""Distance and optional projection onto subdiff F*((xstar-b)/a).
        
        Parameters
        ----------
        v: ReadOnlyVector
            Input vector; its coefficients are only read.
        xstar: ReadOnlyVector
            Input vector; its coefficients are only read.
        best_approx: ReadOnlyVector | None [default: None]
            Input vector; its coefficients are only read.
        **kwargs: Any
            Additional keyword arguments forwarded to the underlying implementation.
        
        
        Returns
        -------
        float
            Computed scalar value.
        """
        self._validate_out(best_approx)
        term = self._single_convex_term()
        if term is None:
            return self.linear_part._conj_dist_subdiff(
                v, xstar, best_approx=best_approx, **kwargs)
        coeff, func = term
        return func.conj.dist_subdiff(
            v, self._conj_argument(xstar, coeff),
            best_approx=best_approx, **kwargs)

    @property
    def Lipschitz(self) -> float:
        r"""Lipschitz.
        
        
        Returns
        -------
        float
            Computed scalar value.
        """
        return sum(abs(c) * f.Lipschitz for c, f in zip(self.coeffs, self.funcs))

    def _convexity_bound(self) -> float:
        return sum(c * (f.convexity_param if c > 0 else f.Lipschitz)
                   for c, f in zip(self.coeffs, self.funcs))

    @property
    def convexity_param(self) -> float:
        r"""Convexity param.
        
        
        Returns
        -------
        float
            Computed scalar value.
        """
        return max(0., self._convexity_bound()) if self.is_convex else 0.

    @property
    def is_convex(self) -> bool:
        r"""Whether the functional is convex.
        
        
        Returns
        -------
        bool
            Whether the stated condition is satisfied.
        """
        return bool(all(f.is_convex for f in self.funcs) and self._convexity_bound() >= 0)

    @property
    def is_strictly_convex(self) -> bool:
        r"""Whether the functional is strictly convex.
        
        
        Returns
        -------
        bool
            Whether the stated condition is satisfied.
        """
        return bool(self.convexity_param > 0 or (
            self.is_convex and all(c > 0 for c in self.coeffs)
            and any(f.is_strictly_convex for f in self.funcs)))

    @property
    def conj_is_strictly_convex(self) -> bool:
        r"""Conj is strictly convex.
        
        
        Returns
        -------
        bool
            Whether the stated condition is satisfied.
        """
        if len(self.funcs) == 1 and self.coeffs[0] > 0:
            return bool(self.funcs[0].conj_is_strictly_convex)
        return bool(np.isfinite(self.Lipschitz) and self.is_convex)

    @property
    def is_quadratic(self) -> bool:
        r"""Is quadratic.
        
        
        Returns
        -------
        bool
            Whether the stated condition is satisfied.
        """
        return all(f.is_quadratic for f in self.funcs)

    @property
    def conj_is_quadratic(self) -> bool:
        r"""Conj is quadratic.
        
        
        Returns
        -------
        bool
            Whether the stated condition is satisfied.
        """
        return bool(len(self.funcs) == 1 and self.coeffs[0] > 0
                    and self.funcs[0].conj_is_quadratic)

    @property
    def is_norm_squared_type(self) -> bool:
        r"""Whether the functional is norm squared type.
        
        
        Returns
        -------
        bool
            Whether the stated condition is satisfied.
        """
        return bool(self.convexity_param > 0
                    and all(f.is_norm_squared_type for f in self.funcs))

    def _domain_bound(self, name: str, lower: bool) -> ReadOnlyVector | float | None:
        if not self.is_separable:
            return None
        values = [getattr(f, name) for f in self.funcs]
        if any(v is None for v in values):
            return None
        if not values:
            return -inf if lower else inf
        bound = values[0]
        # Use vector-space operations for tuple and other non-NumPy vectors.
        for value in values[1:]:
            if np.isscalar(bound) and np.isscalar(value):
                bound = max(bound, value) if lower else min(bound, value)
            else:
                b = bound * self.domain.ones() if np.isscalar(bound) else bound
                v = value * self.domain.ones() if np.isscalar(value) else value
                if not hasattr(self.domain, 'geq_mask'):
                    return None
                bound = self._copy_vector(b)
                mask = self.domain.geq_mask(v, b) if lower else self.domain.geq_mask(b, v)
                self.domain.copyto(bound, v, where=mask)
        return bound

    @property
    def dom_l(self) -> ReadOnlyVector | float | None:
        r"""Dom l.
        
        
        Returns
        -------
        ReadOnlyVector | float | None
            Result described above.
        """
        return self._domain_bound('dom_l', True)

    @property
    def dom_u(self) -> ReadOnlyVector | float | None:
        r"""Dom u.
        
        
        Returns
        -------
        ReadOnlyVector | float | None
            Result described above.
        """
        return self._domain_bound('dom_u', False)

    def _conjugate_bound(self, name: str) -> ReadOnlyVector | float | None:
        if not self.is_separable:
            return None
        if not self.funcs:
            return self._copy_vector(self.grad_sum)
        if len(self.funcs) != 1 or self.coeffs[0] <= 0:
            return None
        bound = getattr(self.funcs[0], name)
        if bound is None:
            return None
        if np.isscalar(bound):
            bound = bound * self.domain.ones()
        return self.grad_sum + self.coeffs[0] * bound

    @property
    def conj_dom_l(self) -> ReadOnlyVector | float | None:
        r"""Conj dom l.
        
        
        Returns
        -------
        ReadOnlyVector | float | None
            Result described above.
        """
        return self._conjugate_bound('conj_dom_l')

    @property
    def conj_dom_u(self) -> ReadOnlyVector | float | None:
        r"""Conj dom u.
        
        
        Returns
        -------
        ReadOnlyVector | float | None
            Result described above.
        """
        return self._conjugate_bound('conj_dom_u')


class VerticalShift(LinearCombination):
    r"""Add a constant to a functional using LinearCombination's flat storage.
    
    Nested offsets are accumulated by the parent constructor. For a simple
    functional this stores one nonlinear component (or one linear part), and
    the supplied offset. Evaluation and linearization include the offset;
    conjugate evaluation subtracts it. Derivatives and proximal maps do not
    depend on the offset.
    
    Parameters
    ----------
    func: Functional
        Functional to evaluate or transform.
    offset: float
        Offset.
    """

    def __init__(self, func: Functional, offset: float) -> None:
        super().__init__(func, offset=offset)


class HorizontalShiftDilation(Functional):
    r"""Implements a horizontal shift and/or a horizontal translation of the graph of a functional :math:`F`, i.e. replaces 
    :math:`F(x)` by :math:`F(dilation(x-shift-data))`.
    
    Nested HorizontalShiftDilation instances are combined into one wrapper.
    An inner transformation b*(x-t) and an outer transformation a*(x-s)
    combine to (b*a)*(x-s-t/a). Wrapper-owned data translations follow the
    same rule; data of the underlying functional remain in its coordinates.
    Supplied shift and data vectors are copied and are not modified.
    
    Parameters
    ----------
    func: Functional
        The functional to be shifted and dilated.
    dilation: float [default: 1.0]
        Dilation factor.
    shift: Vector | Scalar | None [default: None]
        Shift vector in self.domain. The default case (None) yields the same results as shift=0, but no zero-additions are performed.
    data: ReadOnlyVector | None [default: None]
        If not None, HorizontalShiftDilation is used as a data functional
    """
    def __init__(
        self,
        func: Functional,
        dilation: float = 1.0,
        shift: Vector | Scalar | None = None,
        data: ReadOnlyVector | None = None,
    ) -> None:
        if not isinstance(func, Functional) or not isinstance(dilation,(int,float)) or (shift is not None and not np.isscalar(shift) and shift not in func.domain):
            raise ValueError(util.Errors.value_error(f""" 
            The HorizontalShiftDilation only takes three arguments Functional, 
            dilation a scalar and shift that is either a scalar or a element in the domain of the functionals. 
            However, you gave:
                func = {func},
                dilation = {dilation}
                shift = {shift}."""))
        if dilation==0.:
            raise ValueError(util.Errors.value_error("dilation must not vanish."))
        if np.isscalar(shift):
            if isinstance(func.domain, vecsps.NumPyVectorSpace):
                shift = np.broadcast_to(shift,func.domain.shape)
            else:
                shift = shift * func.domain.ones()
        elif shift is not None and shift not in func.domain:
            raise ValueError(util.Errors.not_in_vecsp(shift,func.domain,vec_name="shift vector",space_name="domain of functional"))
        if data is not None and data not in func.domain:
            raise ValueError(util.Errors.not_in_vecsp(data,func.domain,vec_name="data vector",space_name="domain of functional"))
        # F(b*(a*(x-s-d)-t-e)) = F((b*a)*(x-s-d-t/a-e/a)).
        # Keep wrapper-owned data separate from intrinsic shifts.
        if shift is not None:
            shift = func._copy_vector(shift)
        if data is not None:
            data = func._copy_vector(data)
        while isinstance(func, HorizontalShiftDilation):
            if func._shift_val is not None:
                translated = func._shift_val / dilation
                shift = translated if shift is None else shift + translated
            if func.is_data_func and not func._shifted_data_fid:
                translated_data = func.data / dilation
                data = translated_data if data is None else data + translated_data
            dilation *= func.dilation_factor
            func = func.func
        # Data owned by the underlying functional stay in its own coordinates.
        # Additional outer data are a translation, and must not be discarded.
        if func.is_data_func and data is not None:
            shift = data if shift is None else shift + data
            data = None
        self.func = func
        self._dilation_factor = dilation
        self._shift_val = shift
        if self.func.is_data_func:
            if data is not None:
                self.log.warning("Both the underlying functional for the HorizontalShiftDilation is already a data functional. The provided data argument will be ignored.")
            self._shifted_data_fid = True
            self.is_data_func = True
        elif data is None:
            self._shifted_data_fid = False
            self.is_data_func = False
        else:
            self._data = data
            self._shifted_data_fid = False
            self.is_data_func = True

        super().__init__(func.domain, h_domain = func.h_domain, 
                         is_separable = func.is_separable,                         
                         methods = func.methods, conj_methods=func._conj_methods,
                         is_data_func = self.is_data_func#(shift is not None)
                         )
        
    @property
    def dilation_factor(self) -> float:
        """Scalar argument multiplier in ``x -> F(factor * (x - shift))``.

        This intentionally has a different name from the inherited
        ``Functional.dilation(a)`` constructor method. In particular,
        ``f.dilation(a)`` must remain callable for already-shifted functionals
        such as ``KL(offset=sigma)`` and for repeated rescaling.
        """
        return self._dilation_factor

    @util.memoized_property
    def shift_val(self) -> Vector | None:
        r"""Shift val.
        
        
        Returns
        -------
        Vector | None
            Result described above.
        """
        if self.is_data_func and not self._shifted_data_fid:
            if self._shift_val is None:
                return self.data
            else:
                return self._shift_val + self.data
        else:
            if self._shift_val is None:
                return None
            else:
                return self._shift_val
        
    @property
    def data(self) -> ReadOnlyVector:
        r"""Data.
        
        
        Returns
        -------
        ReadOnlyVector
            Read-only vector view.
        """
        if self._shifted_data_fid:
            return self.domain.read_only(self.func.data)
        else:
            return self.domain.read_only(self._data)
    
    @data.setter
    def data(self, new_data: ReadOnlyVector | None) -> None:
        r"""Data.
        
        Parameters
        ----------
        new_data: ReadOnlyVector | None
            Input vector; its coefficients are only read.
        """
        if new_data is None:
            if self._shifted_data_fid:
                del self.func.data
            del self.data
        elif new_data in self.func.domain:
            if self._shifted_data_fid:
                self.func.data = new_data
            else:
                self._data = new_data
            self.is_data_func = True
        else:
            raise ValueError(util.Errors.not_in_vecsp(new_data,self.domain,vec_name="new data vector",space_name="domain of functional"))
        if not self._shifted_data_fid:
            del self.shift_val
        
    @data.deleter
    def data(self) -> None:
        r"""Data.
        """
        if self._shifted_data_fid:
            del self.func.data
            self._shifted_data_fid = False
            self.is_data_func = False
        elif self.is_data_func:
            del self._data
            del self.shift_val
            self.is_data_func = False

    def _shifted_argument(
        self, x: ReadOnlyVector, out: Vector | None = None
    ) -> tuple[Vector, Vector | None]:
        r"""Build ``dilation * (x - shift)`` in one work vector.
        
        Parameters
        ----------
        x: ReadOnlyVector
            Input vector; its coefficients are only read.
        out: Vector | None [default: None]
            Writable output buffer. If None, allocate a new result.
        
        
        Returns
        -------
        tuple[Vector, Vector | None]
            Result described above.
        """
        shift = self.shift_val
        if (
            shift is not None and out is not None
            and self._shares_memory(out, shift)
        ):
            shift = self._copy_vector(shift)
        argument = self._copy_vector(x, out=out)
        if shift is not None:
            argument -= shift
        argument *= self.dilation_factor
        return argument, shift

    def _dual_argument(
        self, xstar: ReadOnlyVector, out: Vector | None = None
    ) -> Vector:
        r"""Build ``xstar / dilation`` in one work vector.
        
        Parameters
        ----------
        xstar: ReadOnlyVector
            Input vector; its coefficients are only read.
        out: Vector | None [default: None]
            Writable output buffer. If None, allocate a new result.
        
        
        Returns
        -------
        Vector
            Writable result vector.
        """
        argument = self._copy_vector(xstar, out=out)
        argument *= 1. / self.dilation_factor
        return argument

    def _eval(self, x: ReadOnlyVector, **kwargs: Any) -> float:
        argument, _ = self._shifted_argument(x)
        return self.func(argument, **kwargs)
         
    def _subgradient(
        self, x: ReadOnlyVector, out: Vector | None = None, **subgradient_par: Any
    ) -> Vector:
        argument, _ = self._shifted_argument(x, out=out)
        result = self.func.subgradient(
            argument, out=argument, **subgradient_par
        )
        result *= self.dilation_factor
        return result

    def dist_subdiff(
        self, vstar: ReadOnlyVector, x: ReadOnlyVector, best_approx: Vector | None = None,
        **kwargs: Any,
    ) -> float:
        r"""Transform the distance and optional projection back by dilation.
        
        Parameters
        ----------
        vstar: ReadOnlyVector
            Input vector; its coefficients are only read.
        x: ReadOnlyVector
            Input vector; its coefficients are only read.
        best_approx: Vector | None [default: None]
            Writable vector used by this operation.
        **kwargs: Any
            Additional keyword arguments forwarded to the underlying implementation.
        
        
        Returns
        -------
        float
            Computed scalar value.
        """
        self._validate_out(best_approx)
        dual_argument = self._dual_argument(vstar)
        argument, _ = self._shifted_argument(x)
        distance = self.func.dist_subdiff(
            dual_argument, argument, best_approx=best_approx, **kwargs
        )
        if best_approx is not None:
            best_approx *= self.dilation_factor
        return abs(self.dilation_factor) * distance

    def _hessian(self, x: ReadOnlyVector, **kwargs: Any) -> operators.Operator:
        argument, _ = self._shifted_argument(x)
        return self.dilation_factor**2 * self.func._hessian(argument, **kwargs)

    def _proximal(
        self,
        x: ReadOnlyVector,
        tau: float,
        out: Vector | None = None,
        **proximal_par: Any,
    ) -> Vector:
        argument, shift = self._shifted_argument(x, out=out)
        result = self.func.proximal(
            argument, tau * self.dilation_factor ** 2,
            out=argument, **proximal_par
        )
        result *= 1. / self.dilation_factor
        if shift is not None:
            result += shift
        return result
    
    def _conj(self, x_star: ReadOnlyVector, **kwargs: Any) -> float:
        argument = self._dual_argument(x_star)
        if self.shift_val is None:
            return self.func._conj(argument, **kwargs)
        return (
            self.func._conj(argument, **kwargs)
            + self.domain.vdot(x_star, self.shift_val).real
        )

    def _conj_subgradient(
        self,
        x_star: ReadOnlyVector,
        out: Vector | None = None,
        **subgradient_par: Any,
    ) -> Vector:
        shift = self.shift_val
        if (
            shift is not None and out is not None
            and self._shares_memory(out, shift)
        ):
            shift = self._copy_vector(shift)
        argument = self._dual_argument(x_star, out=out)
        result = self.func.conj_subgradient(
            argument, out=argument, **subgradient_par
        )
        result *= 1. / self.dilation_factor
        if shift is not None:
            result += shift
        return result
 
    def _conj_dist_subdiff(
        self, v: ReadOnlyVector, x_star: ReadOnlyVector, best_approx: Vector | None = None,
        **kwargs: Any,
    ) -> float:
        r"""Undo dilation and add the shift to the conjugate projection.
        
        Parameters
        ----------
        v: ReadOnlyVector
            Input vector; its coefficients are only read.
        x_star: ReadOnlyVector
            Input vector; its coefficients are only read.
        best_approx: Vector | None [default: None]
            Writable vector used by this operation.
        **kwargs: Any
            Additional keyword arguments forwarded to the underlying implementation.
        
        
        Returns
        -------
        float
            Computed scalar value.
        """
        self._validate_out(best_approx)
        dual_argument = self._dual_argument(x_star)
        primal_argument = self._copy_vector(v)
        shift = self.shift_val
        if shift is not None:
            primal_argument -= shift
            if best_approx is not None and self._shares_memory(best_approx, shift):
                shift = self._copy_vector(shift)
        primal_argument *= self.dilation_factor
        distance = self.func._conj_dist_subdiff(
            primal_argument, dual_argument, best_approx=best_approx, **kwargs
        )
        if best_approx is not None:
            best_approx *= 1. / self.dilation_factor
            if shift is not None:
                best_approx += shift
        return distance / abs(self.dilation_factor)

    def _conj_hessian(
        self, x_star: ReadOnlyVector, **kwargs: Any
    ) -> operators.Operator:
        argument = self._dual_argument(x_star)
        return self.dilation_factor**(-2)*self.func._conj_hessian(argument,**kwargs)

    def _conj_proximal(
        self,
        xstar: ReadOnlyVector,
        tau: float,
        out: Vector | None = None,
        **proximal_par: Any,
    ) -> Vector:
        shift = self.shift_val
        gram_shift = None
        if shift is not None:
            gram_shift = self.domain.empty()
            self.h_domain.gram(shift, out=gram_shift)
            gram_shift *= tau / self.dilation_factor
        argument = self._dual_argument(xstar, out=out)
        if gram_shift is not None:
            argument -= gram_shift
        result = self.func.conj_proximal(
            argument, tau / self.dilation_factor ** 2,
            out=argument, **proximal_par
        )
        result *= self.dilation_factor
        return result
    @property
    def Lipschitz(self) -> float:
        r"""Lipschitz.
        
        
        Returns
        -------
        float
            Computed scalar value.
        """
        return self.func.Lipschitz * self.dilation_factor**2
    
    @property
    def convexity_param(self) -> float:
        r"""Convexity param.
        
        
        Returns
        -------
        float
            Computed scalar value.
        """
        return self.func.convexity_param  * self.dilation_factor**2
    
    @property
    def is_strictly_convex(self) -> bool:
        r"""Whether the functional is strictly convex.
        
        
        Returns
        -------
        bool
            Whether the stated condition is satisfied.
        """
        return bool(self.func.is_strictly_convex)

    @property
    def conj_is_strictly_convex(self) -> bool:
        r"""Conj is strictly convex.
        
        
        Returns
        -------
        bool
            Whether the stated condition is satisfied.
        """
        return bool(self.func.conj_is_strictly_convex)

    @property
    def is_convex(self) -> bool:
        r"""Whether the functional is convex.
        
        
        Returns
        -------
        bool
            Whether the stated condition is satisfied.
        """
        return self.func.is_convex

    @property
    def is_linear(self) -> bool:
        r"""Is linear.
        
        
        Returns
        -------
        bool
            Whether the stated condition is satisfied.
        """
        return self.func.is_linear and self.shift_val is None
    
    @property
    def is_quadratic(self) -> bool:
        r"""Is quadratic.
        
        
        Returns
        -------
        bool
            Whether the stated condition is satisfied.
        """
        return self.func.is_quadratic

    @property
    def is_norm_squared_type(self) -> bool:
        r"""Whether the functional is norm squared type.
        
        
        Returns
        -------
        bool
            Whether the stated condition is satisfied.
        """
        return super().is_norm_squared_type
    
    @property
    def dom_l(self) -> Vector | None:
        r"""Dom l.
        
        
        Returns
        -------
        Vector | None
            Result described above.
        """
        if self.func.is_separable:
            if self.dilation_factor>0:
                return self.func.dom_l/self.dilation_factor if self.shift_val is None else self.func.dom_l/self.dilation_factor + self.shift_val
            else:
                return self.func.dom_u/self.dilation_factor if self.shift_val is None else self.func.dom_u/self.dilation_factor + self.shift_val
        else:
            return None

    @property
    def dom_u(self) -> Vector | None:
        r"""Dom u.
        
        
        Returns
        -------
        Vector | None
            Result described above.
        """
        if self.func.is_separable:
            if self.dilation_factor>0:
                return self.func.dom_u/self.dilation_factor if self.shift_val is None else self.func.dom_u/self.dilation_factor + self.shift_val
            else:
                return self.func.dom_l/self.dilation_factor if self.shift_val is None else self.func.dom_l/self.dilation_factor + self.shift_val
        else:
            return None

    @property
    def conj_dom_l(self) -> Vector | None:
        r"""Conj dom l.
        
        
        Returns
        -------
        Vector | None
            Result described above.
        """
        if self.func.is_separable:
            return  self.func.conj_dom_l*self.dilation_factor if self.dilation_factor>0. else self.func.conj_dom_u*self.dilation_factor
        else:
            return None

    @property
    def conj_dom_u(self) -> Vector | None:
        r"""Conj dom u.
        
        
        Returns
        -------
        Vector | None
            Result described above.
        """
        if self.func.is_separable:
            return  self.func.conj_dom_u*self.dilation_factor if self.dilation_factor>0. else self.func.conj_dom_l*self.dilation_factor
        else:
            return None
  


class Composed(Functional):
    r"""Composition of an operator with a functional :math:`F\circ O`. This should not be called
    directly but rather used by multiplying the `Functional` object with an `regpy.operators.Operator`.
    
    Parameters
    ----------
    func: Functional
        Functional to be composed with.
    op: operators.Operator
        Operator to be composed with.
    op_norm: float [default: inf]
        Norm of the operator. Used only to define self.Lipschitz
    op_lower_bound: float [default: 0]
        Lower bound of operator: \|op(f)\|\geq op_lower_bound * \|f\|
        Used only to define self.convexity_param
    hessian: Callable[[Vector], Operator] | None [default: None]
        If not None, this allows to specify a user-defined hessian (e.g.,  with allowing an inversion or more efficient inversion)
    compute_op_norm: bool [default: False]
        If true the op norm will be computed using the norm method of the operator. Will only be computed if op_norm 
        is default value inf.
    norm_kwargs: dict[str, Any] [default: {}]
        possible arguments passed to the operator norm computation.
    h_domain: HilbertSpace | None [default: None]
        Underlying Hilbert space.
    methods: set[str] | None [default: None]
        Names of the implemented functional operations.
    conj_methods: set[str] | None [default: None]
        Names of the implemented functional operations.
    """
    def __init__(self, func:Functional,
                 op:operators.Operator, 
                 op_norm :float= inf, 
                 op_lower_bound:float = 0, 
                 hessian: Callable[[Vector], Operator] | None = None,
                 compute_op_norm:bool = False,
                 norm_kwargs: dict[str, Any] = {},
                 h_domain: HilbertSpace | None = None,
                 methods:set[str]|None = None,
                 conj_methods:set[str]|None=None) -> None:
        if not isinstance(func, Functional):
            raise TypeError(util.Errors.not_instance(func,Functional))
        if not isinstance(op,operators.Operator):
            raise TypeError(util.Errors.not_instance(op,operators.Operator))
        if func.domain != op.codomain:
            raise ValueError(util.Errors.not_equal(func.domain,op.codomain, add_info="Codomain of operator and domain of fucntional have to match to be composed."))
        self.op_norm = op.norm(h_codomain = func.h_domain, **norm_kwargs)  if (op_norm == inf and compute_op_norm) else op_norm
        self.op_lower_bound = op_lower_bound
        if conj_methods is None:
            if op.invertible:
                conj_methods = {'eval','subgradient','hessian'}
            else:
                conj_methods = set()

        super().__init__(op.domain,
                         h_domain = h_domain, 
                         methods = {'eval','subgradient','hessian'} if methods is None else methods,
                         conj_methods = conj_methods
                         )
        self.user_defined_hessian = hessian
        if isinstance(func, type(self)):
            op = func.op * op
            func = func.func
        self.func = func
        """Functional that is composed with an Operator. 
        """
        self.op = op
        """Operator composed that is composed with a functional. 
        """

    def _eval(self, x: ReadOnlyVector) -> float:
        argument = self.func.domain.empty()
        self.op(x, out=argument)
        return self.func(argument)

    def _linearize(self, x: ReadOnlyVector) -> tuple[float, Vector]:
        y = self.func.domain.empty()
        y, deriv = self.op.linearize(x, out=y)
        z, grad = self.func.linearize(y)
        return z, deriv.adjoint(grad)

    def _subgradient(
        self, x: ReadOnlyVector, out: Vector | None = None, **subgradient_par: Any
    ) -> Vector:
        y = self.func.domain.empty()
        y, deriv = self.op.linearize(x, out=y)
        inner_grad = self.func.subgradient(
            y, out=y, **subgradient_par
        )
        return deriv.adjoint(inner_grad, out=out)

    def _hessian(self, x: ReadOnlyVector) -> operators.Operator:
        if self.user_defined_hessian is None:
            if self.op.linear:
                argument = self.func.domain.empty()
                self.op(x, out=argument)
                return self.op.adjoint * self.func.hessian(argument) * self.op
            else:
                return super()._hessian(x)
        else:
            return self.user_defined_hessian(x)

    def _conj(self, x_star: ReadOnlyVector) -> float:
        if self.op.linear:
            argument = self.func.domain.empty()
            self.op.inverse.adjoint(x_star, out=argument)
            return self.func._conj(argument)

    def _conj_subgradient(
        self,
        x_star: ReadOnlyVector,
        out: Vector | None = None,
        **subgradient_par: Any,
    ) -> Vector:
        if not self.op.linear:
            raise NotImplementedError
        dual_argument = self.func.domain.empty()
        self.op.inverse.adjoint(x_star, out=dual_argument)
        inner_grad = self.func.conj_subgradient(
            dual_argument, out=dual_argument, **subgradient_par
        )
        return self.op.inverse(inner_grad, out=out)

    def _conj_hessian(self, x_star: ReadOnlyVector) -> operators.Operator:
        if self.op.linear:
            argument = self.func.domain.empty()
            self.op.inverse.adjoint(x_star, out=argument)
            return (
                self.op.inverse * self.func._conj_hessian(argument)
                * self.op.inverse.adjoint
            )

    def _proximal(
        self,
        x: ReadOnlyVector,
        tau: float,
        out: Vector | None = None,
        cg_params: dict[str, Any] | None = None,
        **proximal_par: Any,
    ) -> Vector:
        # TODO: Remove this from the general class! All derived classes should implement their own prox!
        # In case it is a functional 1/2||Tx-g^delta||^2 can approximated by a Tikhonov solver
        if isinstance(self.func, SquaredNorm) and self.func.a == 1 and (self.func.b == 0).all() and self.func.c == 0 and isinstance(self.op, operators.OuterShift) and self.op.op.linear:
            from regpy.solvers.linear.tikhonov import TikhonovCG
            from regpy.solvers import Setting
            params = {} if cg_params is None else cg_params
            params = {**params, **proximal_par}
            result, _ = TikhonovCG(
                setting=Setting(self.op.op, hilbert.L2, self.func.h_domain),
                data=-self.op.offset,
                xref=x,
                regpar=tau,
                **params
            ).run()
            if out is None:
                return result
            self.domain.copyto(out, result)
            return out
        raise NotImplementedError
        
    @property
    def is_linear(self) -> bool:
        r"""Is linear.
        
        
        Returns
        -------
        bool
            Whether the stated condition is satisfied.
        """
        return self.func.is_linear and self.op.linear
    
    @property
    def is_quadratic(self) -> bool:
        r"""Is quadratic.
        
        
        Returns
        -------
        bool
            Whether the stated condition is satisfied.
        """
        return self.func.is_quadratic and self.op.linear
    
    @property
    def conj_is_quadratic(self) -> bool:
        r"""Conj is quadratic.
        
        
        Returns
        -------
        bool
            Whether the stated condition is satisfied.
        """
        return self. func.is_quadratic and self.op.linear  and  self.func.convexity_param * self.op_lower_bound**2>0

    @property 
    def convexity_param(self) -> float:
        r"""Convexity param.
        
        
        Returns
        -------
        float
            Computed scalar value.
        """
        return self.func.convexity_param * self.op_lower_bound**2
    
    @property
    def Lipschitz(self) -> float:
        r"""Lipschitz.
        
        
        Returns
        -------
        float
            Computed scalar value.
        """
        return self.func.Lipschitz * self. op_norm**2
    
    @property
    def is_convex(self) -> bool:
        r"""Whether the functional is convex.
        
        
        Returns
        -------
        bool
            Whether the stated condition is satisfied.
        """
        return self.func.is_convex and self.op.linear         

class FunctionalOnDirectSum(Functional):
    r"""Helper to define Functionals with respective prox-operators on sum spaces (vecsps.DirectSum objects).
    The functionals are given as a list of the functionals on the summands of the sum space.
    
    .. math::
        F(x_1,... x_n) = \sum_{j=1}^n F_j(x_j)
    
    
    Parameters
    ----------
    funcs: list[Functional] | tuple[Functional, ...]
        List of functionals each defined on one summand of the direct sum of vector spaces.
    domain: vecsps.DirectSum | None [default: None]
        Domain on which the combined functional is defined.
    
    Notes
    -----
    The Hilbert geometry is the direct sum of the component functionals'
    ``h_domain`` spaces. In particular, components defined on dual Hilbert
    spaces retain their dual geometry, as required by the componentwise
    proximal operators and Moreau's identity.
    """
    def __init__(
        self,
        funcs: list[Functional] | tuple[Functional, ...],
        domain: vecsps.DirectSum | None = None,
    ) -> None:
        if not isinstance(funcs,(list,tuple)) or any([not isinstance(f_i, Functional) for f_i in funcs]):
            raise TypeError(util.Errors.generic_message(f"To setup a FunctionalOnDirectSum the functionals have to be provided as a list or tuple of functionals."))
        if domain is not None:
            if not isinstance(domain, vecsps.DirectSum) or len(funcs)!=len(domain.summands) or any([f_i.domain != domain_i for f_i,domain_i in zip(funcs,domain.summands)]) :
                raise TypeError(util.Errors.not_instance(domain,vecsps.DirectSum,add_info="The given domain for a FunctionalOnDirectSum has to be a DirectSum with the same length of arguments and matching domains!"))
        else:
            domain = vecsps.DirectSum(*[f_i.domain for f_i in funcs])
        self.length = len(domain.summands)
        """Number of the summands in the direct sum domain. 
        """
        self.funcs = list(funcs)
        """List of the functionals on each summand of the direct sum domain.
        """
        is_separable = all([func.is_separable for func in funcs])
        methods = set.intersection(*[func.methods for func in funcs])
        conj_methods = set.intersection(*[func.conj.methods for func in funcs])
        super().__init__(domain,
                        h_domain=hilbert.DirectSum(
                            *[func.h_domain for func in funcs], vecsp=domain),
                        is_separable = is_separable,                        
                        methods=methods,conj_methods=conj_methods 
                        )

    def bregman_func(self, y: ReadOnlyVector, ystar: ReadOnlyVector | None = None) -> Functional:
        r"""Construct componentwise Bregman functionals on the same direct sum.
        
        Parameters
        ----------
        y: ReadOnlyVector
            Input vector; its coefficients are only read.
        ystar: ReadOnlyVector | None [default: None]
            Input vector; its coefficients are only read.
        
        
        Returns
        -------
        Functional
            Constructed functional.
        """
        if not self.is_convex:
            return super().bregman_func(y, ystar)
        if y not in self.domain or (ystar is not None and ystar not in self.domain):
            raise ValueError('Bregman vectors must belong to the direct-sum domain.')
        return FunctionalOnDirectSum([
            f.bregman_func(y[i], None if ystar is None else ystar[i])
            for i, f in enumerate(self.funcs)
        ], self.domain)


    def conj_bregman_func(self, y: ReadOnlyVector, ystar: ReadOnlyVector | None = None) -> Functional:
        r"""Keep the componentwise structure for the conjugate as well.
        
        Parameters
        ----------
        y: ReadOnlyVector
            Input vector; its coefficients are only read.
        ystar: ReadOnlyVector | None [default: None]
            Input vector; its coefficients are only read.
        
        
        Returns
        -------
        Functional
            Constructed functional.
        """
        if y not in self.domain or (ystar is not None and ystar not in self.domain):
            raise ValueError('Bregman vectors must belong to the direct-sum domain.')
        return FunctionalOnDirectSum([
            f.conj.bregman_func(y[i], None if ystar is None else ystar[i])
            for i, f in enumerate(self.funcs)
        ], self.domain)


    def _preserve_input_if_needed(
        self, x: vecsps.TupleVector, out: vecsps.TupleVector
    ) -> vecsps.TupleVector:
        r"""Protect later components from writes through aliased outputs.
        
        Parameters
        ----------
        x: vecsps.TupleVector
            X.
        out: vecsps.TupleVector
            Writable output buffer. If None, allocate a new result.
        
        
        Returns
        -------
        vecsps.TupleVector
            Result described above.
        """
        for i, out_i in enumerate(out[:-1]):
            if any(
                self._shares_memory(out_i, x_j) for x_j in x[i + 1:]
            ):
                return self._copy_vector(x)
        return x

    def _eval(self, x: vecsps.TupleVector) -> float:
        return sum(f_i(x_i) for f_i, x_i in zip(self.funcs, x))

    def _subgradient(
        self,
        x: vecsps.TupleVector,
        out: vecsps.TupleVector | None = None,
        **subgradient_par: Any,
    ) -> vecsps.TupleVector:
        if out is None:
            out = self.domain.zeros()
        else:
            x = self._preserve_input_if_needed(x, out)
        for func, x_i, out_i in zip(self.funcs, x, out):
            func.subgradient(x_i, out=out_i, **subgradient_par)
        return out

    def dist_subdiff(
        self, vstar: vecsps.TupleVector, x: vecsps.TupleVector,
        best_approx: vecsps.TupleVector | None = None,
    ) -> float:
        r"""Combine component projections and distances in the dual sum norm.
        
        Parameters
        ----------
        vstar: vecsps.TupleVector
            Vstar.
        x: vecsps.TupleVector
            X.
        best_approx: vecsps.TupleVector | None [default: None]
            Best approx.
        
        
        Returns
        -------
        float
            Computed scalar value.
        """
        self._validate_out(best_approx)
        if best_approx is not None:
            vstar = self._preserve_input_if_needed(vstar, best_approx)
            x = self._preserve_input_if_needed(x, best_approx)
        return sum(
            f_i.dist_subdiff(vstar_i, x_i,
                            best_approx=None if best_approx is None else best_approx[i]) ** 2
            for i, (f_i, vstar_i, x_i) in enumerate(zip(self.funcs, vstar, x))
        ) ** 0.5

    def _hessian(self, x: vecsps.TupleVector) -> operators.DirectSum:
        return operators.DirectSum(*tuple(f_i.hessian(x_i) for f_i,x_i in zip(self.funcs,x)))

    def _proximal(
        self,
        x: vecsps.TupleVector,
        tau: float,
        out: vecsps.TupleVector | None = None,
        proximal_par_list: list[dict[str, Any] | None] | None = None,
        **proximal_par: Any,
    ) -> vecsps.TupleVector:
        if proximal_par_list is not None and len(proximal_par_list) != self.length:
            raise ValueError(util.Errors.generic_message(
                "The proximal parameters in FunctionalOnDirectSum have to be either a list of dictionaries of same length or None!"
            ))
        if out is None:
            out = self.domain.zeros()
        else:
            x = self._preserve_input_if_needed(x, out)
        for i, (func, x_i, out_i) in enumerate(zip(self.funcs, x, out)):
            params = dict(proximal_par)
            if proximal_par_list is not None and proximal_par_list[i] is not None:
                params.update(proximal_par_list[i])
            func.proximal(x_i, tau, out=out_i, **params)
        return out

    def _conj(self, xstar: vecsps.TupleVector) -> float:
        return sum(
            f_i.conj(xstar_i)
            for f_i, xstar_i in zip(self.funcs, xstar)
        )

    def _conj_subgradient(
        self,
        xstar: vecsps.TupleVector,
        out: vecsps.TupleVector | None = None,
        **subgradient_par: Any,
    ) -> vecsps.TupleVector:
        if out is None:
            out = self.domain.zeros()
        else:
            xstar = self._preserve_input_if_needed(xstar, out)
        for func, xstar_i, out_i in zip(self.funcs, xstar, out):
            func.conj_subgradient(
                xstar_i, out=out_i, **subgradient_par
            )
        return out

    def _conj_dist_subdiff(
        self, v: vecsps.TupleVector, xstar: vecsps.TupleVector,
        best_approx: vecsps.TupleVector | None = None,
    ) -> float:
        r"""Combine conjugate projections and distances in the primal sum norm.
        
        Parameters
        ----------
        v: vecsps.TupleVector
            V.
        xstar: vecsps.TupleVector
            Xstar.
        best_approx: vecsps.TupleVector | None [default: None]
            Best approx.
        
        
        Returns
        -------
        float
            Computed scalar value.
        """
        self._validate_out(best_approx)
        if best_approx is not None:
            v = self._preserve_input_if_needed(v, best_approx)
            xstar = self._preserve_input_if_needed(xstar, best_approx)
        return sum(
            f_i.conj.dist_subdiff(v_i, xstar_i,
                                 best_approx=None if best_approx is None else best_approx[i]) ** 2
            for i, (f_i, v_i, xstar_i) in enumerate(zip(self.funcs, v, xstar))
        ) ** 0.5

    def _conj_hessian(self, xstar: vecsps.TupleVector) -> operators.DirectSum:
        return operators.DirectSum(*tuple(f_i.conj.hessian(xstar_i) for f_i,xstar_i in zip(self.funcs,xstar)))

    def _conj_proximal(
        self,
        xstar: vecsps.TupleVector,
        tau: float,
        out: vecsps.TupleVector | None = None,
        proximal_par_list: list[dict[str, Any] | None] | None = None,
        **proximal_par: Any,
    ) -> vecsps.TupleVector:
        if proximal_par_list is not None and len(proximal_par_list) != self.length:
            raise ValueError(util.Errors.generic_message(
                "The proximal parameters in conjugate of FunctionalOnDirectSum have to be either a list of dictionaries of same length or None!"
            ))
        if out is None:
            out = self.domain.zeros()
        else:
            xstar = self._preserve_input_if_needed(xstar, out)
        for i, (func, xstar_i, out_i) in enumerate(zip(self.funcs, xstar, out)):
            params = dict(proximal_par)
            if proximal_par_list is not None and proximal_par_list[i] is not None:
                params.update(proximal_par_list[i])
            func.conj_proximal(xstar_i, tau, out=out_i, **params)
        return out

    @property
    def is_linear(self) -> bool:
        r"""Is linear.
        
        
        Returns
        -------
        bool
            Whether the stated condition is satisfied.
        """
        return all([func.is_linear for func in self.funcs])
    
    @property
    def is_quadratic(self) -> bool:
        r"""Is quadratic.
        
        
        Returns
        -------
        bool
            Whether the stated condition is satisfied.
        """
        return all([func.is_quadratic for func in self.funcs])
    
    @property
    def conj_is_quadratic(self) -> bool:
        r"""Conj is quadratic.
        
        
        Returns
        -------
        bool
            Whether the stated condition is satisfied.
        """
        return all([func.conj_is_quadratic for func in self.funcs])

    @property
    def is_norm_squared_type(self) -> bool:
        r"""Whether the functional is norm squared type.
        
        
        Returns
        -------
        bool
            Whether the stated condition is satisfied.
        """
        conv_params = [func.convexity_param for func in self.funcs]
        return all([func.is_norm_squared_type for func in self.funcs]) and np.allclose(conv_params,conv_params[0])
    
    @property
    def convexity_param(self) -> float:
        r"""Convexity param.
        
        
        Returns
        -------
        float
            Computed scalar value.
        """
        return min([func.convexity_param for func in self.funcs])
    
    @property
    def Lipschitz(self) -> float:
        r"""Lipschitz.
        
        
        Returns
        -------
        float
            Computed scalar value.
        """
        return max([func.Lipschitz for func in self.funcs])
    
    @property
    def is_strictly_convex(self) -> bool:
        r"""Whether the functional is strictly convex.
        
        
        Returns
        -------
        bool
            Whether the stated condition is satisfied.
        """
        return all(func.is_strictly_convex for func in self.funcs)

    @property
    def conj_is_strictly_convex(self) -> bool:
        r"""Conj is strictly convex.
        
        
        Returns
        -------
        bool
            Whether the stated condition is satisfied.
        """
        return all(func.conj_is_strictly_convex for func in self.funcs)

    @property
    def is_convex(self) -> bool:
        r"""Whether the functional is convex.
        
        
        Returns
        -------
        bool
            Whether the stated condition is satisfied.
        """
        return all([func.is_convex for func in self.funcs])

    def _join_domain_bounds(self, name: str) -> vecsps.TupleVector | None:
        r"""Assemble component bounds, broadcasting scalars in their spaces.
        
        Unknown bounds remain unknown. Scalar bounds (including infinities)
        are expanded with fill rather than multiplication, which would
        produce NaNs from zero coordinates.
        
        Parameters
        ----------
        name: str
            Name.
        
        
        Returns
        -------
        vecsps.TupleVector | None
            Result described above.
        """
        if not self.is_separable:
            return None
        bounds = []
        for func, space in zip(self.funcs, self.domain.summands, strict=True):
            bound = getattr(func, name)
            if bound is None:
                return None
            if np.isscalar(bound):
                value = space.empty()
                space.fill(value, bound)
                bound = value
            bounds.append(bound)
        return self.domain.join(*bounds)

    @property
    def dom_l(self) -> vecsps.TupleVector | None:
        r"""Dom l.
        
        
        Returns
        -------
        vecsps.TupleVector | None
            Result described above.
        """
        return self._join_domain_bounds("dom_l")

    @property
    def dom_u(self) -> vecsps.TupleVector | None:
        r"""Dom u.
        
        
        Returns
        -------
        vecsps.TupleVector | None
            Result described above.
        """
        return self._join_domain_bounds("dom_u")

    @property
    def conj_dom_l(self) -> vecsps.TupleVector | None:
        r"""Conj dom l.
        
        
        Returns
        -------
        vecsps.TupleVector | None
            Result described above.
        """
        return self._join_domain_bounds("conj_dom_l")

    @property
    def conj_dom_u(self) -> vecsps.TupleVector | None:
        r"""Conj dom u.
        
        
        Returns
        -------
        vecsps.TupleVector | None
            Result described above.
        """
        return self._join_domain_bounds("conj_dom_u")

    def __add__(self, other: object) -> Functional | NotImplementedType:
        if isinstance(other,FunctionalOnDirectSum):
            return FunctionalOnDirectSum([F+G for F,G in zip(self.funcs,other.funcs)],self.domain)
        elif isinstance(other,(int,float)):
            return FunctionalOnDirectSum([F+other for F in self.funcs])
        else: 
            return super().__add__(self,other)
        
    def __rmul__(self, other: object) -> FunctionalOnDirectSum:
        if (isinstance(other,int) or isinstance(other,float)):
            return FunctionalOnDirectSum([other*F for F in self.funcs],self.domain)
        else:
            raise NotImplementedError(f"Recursive multiplication of other={other} with self={self} is not defined.")


def as_functional(
    func: HilbertSpace
    | Operator
    | Functional
    | Callable[[VectorSpaceBase], HilbertSpace | Functional],
    vecsp: VectorSpaceBase,
) -> Functional:
    r"""Convert `func` to Functional instance on vecsp.
    
    - If func is a `HilbertSpace` then it generated the `SquaredNorm`.
    - If func is an Operator, it's wrapped in a `GramHilbertSpace` and then `SquaredNorm` functional.
    - If func is callable, e.g. an `hilbert.AbstractSpace` or `AbstractFunctional`, it is called on `vecsp` to construct the concrete functional or Hilbert space. In the later case the functional will be the `SquaredNorm`
    
    Parameters
    ----------
    func: HilbertSpace | Operator | Functional | Callable[[VectorSpaceBase], HilbertSpace | Functional]
        Functional or object from which to construct the Functional.
    vecsp: VectorSpaceBase
        Underlying vector space for the functional.
    
    Returns
    -------
    Functional
        Constructed Functional on the underlying vectorspace.
    """
    if not isinstance(func,Functional):
        if isinstance(func, operators.Operator):
            func = SquaredNorm(hilbert.GramHilbertSpace(func))
        elif callable(func):
            func = func(vecsp)
        if isinstance(func, hilbert.HilbertSpace):
            func = SquaredNorm(func)
    if func.domain != vecsp:
        raise ValueError(f"Given Vector space {vecsp} and the domain of the functional {func.domain} do not match.")
    elif isinstance(func,Composed) and func.op.domain != vecsp:
        raise ValueError(f"Given Vector space {vecsp} and the domain of the composed functional {func.func.domain} do not match.")
    return func
