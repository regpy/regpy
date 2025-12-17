from collections import defaultdict
from copy import copy
from math import inf
import logging

import numpy as np
from numpy import isscalar

from regpy import operators, util, vecsps, hilbert

__all__ = ["as_functional","AbstractFunctional","Functional","LinearFunctional","LinearCombination","Composed","SquaredNorm","VerticalShift","HorizontalShiftDilation","FunctionalOnDirectSum"]

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

    def __mul__(self, other):
        if isscalar(other) and other == 1:
            return self
        elif isinstance(other, operators.Operator):
            return AbstractComposed(self, other)
        return NotImplemented

    def __rmul__(self, other):
        if isscalar(other):
            if other == 1:
                return self
            elif util.is_real_dtype(other):
                return AbstractLinearCombination((other, self))
        return NotImplemented

    def __truediv__(self, other):
        return (1 / other) * self

    def __add__(self, other):
        if isinstance(other, AbstractFunctional):
            return AbstractLinearCombination(self, other)
        elif isscalar(other):
            return AbstractVerticalShift(self, other)
        return NotImplemented

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def __neg__(self):
        return (-1) * self

    def __pos__(self):
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
    name : str
        A name for this abstract functional. Currently, this is only used in error messages, when no
        implementation was found for some vector space.
    """

    def __init__(self, name):
        self._registry = {}
        self.name = name
        self.args = {}

    def register(self, vecsp_type, impl=None):
        r"""Either registers a new implementation on a specific `regpy.vecsps.VectorSpaceBase` 
        for a given Abstract functional or returns as decorator that can output any implementation
        option for a given vector space.

        Parameters
        ----------
        vecsp_type : `regpy.vecsps.VectorSpaceBase`
            Vector Space on which the functional should be registered. 
        impl : regpy.functionals.Functional, optional
            The explicit implementation to be used for that Vector Space, by default None

        Returns
        -------
        None or decorator : None or map
            Either nothing or map that can output any of the registered implementations for 
            a specific vector space. 
        """
        if impl is not None:
            self._registry.setdefault(vecsp_type, []).append(impl)
            self.__doc__ += "-"*125+ "\n" + f"--- Implementation for {vecsp_type.__name__} is given by {impl.__name__} with the following documentation ---" +"\n" + f"{impl.__doc__}" +"\n" + "-"*125
        else:
            def decorator(i):
                self.register(vecsp_type, i)
                return i
            return decorator

    def __call__(self, vecsp=None, **kwargs):
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
    *args : (scalar, regpy.functionals.AbstractFunctional) or regpy.functionals.AbstractFunctional
        List of coefficients and functionals to be taken as linear combinations.
    """
    def __init__(self,*args):
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

    def __call__(self,vecsp, **kwargs):
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

    def __getitem__(self,item):
        return self.coeffs[item], self.funcs[item]

    def __iter__(self):
        return iter(zip(self.coeffs,self.funcs))

class AbstractVerticalShift(AbstractFunctional):
    r"""Abstract analogue to `VerticalShift` class. Shifting a functional by some offset. Should not be used directly but rather by adding some scalar to the functional. 

    Parameters
    ----------
    func : regpy.functionals.AbstractFunctional
        Functional to be offset.
    offset : scalar
        Offset added to the evaluation of the functional.
    """
    def __init__(self, func, offset):
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

    def __call__(self,vecsp = None, **kwargs):
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
    func : `regpy.functionals.AbstractFunctional`
        Functional to be composed with. 
    op : `regpy.operators.Operator`
        Operator to be composed with. 
    """
    def __init__(self, func, op):
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

    def __call__(self, vecsp = None, **kwargs):
        if vecsp is None:
            self.func = self.func(**kwargs)
        elif not isinstance(vecsp,vecsps.VectorSpaceBase):
            raise ValueError(util.Errors.not_instance(vecsp,vecsps.VectorSpaceBase))
        else:
            return Composed(func=self.func(vecsp),op=self.op)


class Functional:
    r"""
    Base class for implementation of functionals. Subclasses should at least implement the 
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
    domain : regpy.vecsps.VectorSpaceBase
        The underlying vector space for the function space on which it is defined.
    h_domain : regpy.hilbert.HilbertSpace (default: None)
        The underlying Hilbert space. The proximal mapping, the parameter of strong convexity, 
        and the Lipschitz constant are defined with respect to this Hilbert space.
        In the default case `L2(domain)` is used.
    convex: bool [default: True]
        If true, the functional should be convex.   
    linear: bool [default: False]
        If true, the functional should be linear. 
    separable: bool [default: False]
        If true, the functional should be the sum of functionals acting on only one component of the input vector.
        In this case, the parameters  
    dom_u, dom_l, conj_dom_u, conj_dom_l: self.domain [default:None]
        should not be None, and they should specify the essential domain of the functional by 
        :math:`\{x in domain: dom_l<=x<=dom_u}`, 
        and the essential domain of the conjugate functional (which is then also separable) by 
        :math:`\{xstar in domain: conj_dom_l<=xstar <= conj_dom_u\}`.
        In case of open domains, the boundaries should be shifted in the order of machine precision. 
    convexity_param: float [default: 0]
        parameter of strong convexity of the functional. 
        0 if the functional is not strongly convex.
    Lipschitz: float [default: math.inf]
        Lipschitz continuity constant of the gradient.  
        math.inf the gradient is not Lipschitz continuous.
    methods: set [default: set()]
        names of the methods implemented by a given Functional instance.
        Subset of {'eval', 'subgradient', 'hessian', 'proximal', 'dist_subdiff'}
    conj_methods set [default: set()]
        names of the methods implemented by the conjugate of a given Functional instance.
        Subset of {'eval', 'subgradient', 'hessian', 'proximal', 'dist_subdiff'}        
    """

    log = util.ClassLogger()

    def __init__(self, domain, h_domain=None, 
                 linear = False,
                 convexity_param=0.,
                 Lipschitz = inf,
                 separable = False,
                 convex = True,
                 dom_l=None, dom_u=None,conj_dom_l=None,conj_dom_u=None,
                 methods = set(), conj_methods = set(),
                 is_data_func = False
                 ):
        if not isinstance(domain, vecsps.VectorSpaceBase):
            raise TypeError(f'domain must be an instance of VectorSpaceBase. Got {domain}')
        self.domain = domain
        """The underlying vector space."""
        self.h_domain = hilbert.as_hilbert_space(h_domain,domain) or hilbert.L2(domain)
        """The underlying Hilbert space."""

        if not isinstance(convexity_param, (float, np.floating,int,np.integer)) and convexity_param>=0:
            raise ValueError(f'convexity_param must be a scalar, nonnegative float. Got {convexity_param}.')
        self.convexity_param = np.float64(convexity_param)
        """parameter of strong convexity of the functional."""
        if not isinstance(Lipschitz, (float, np.floating,int,np.integer)) and Lipschitz>=0:
            raise ValueError(f'Lipschitz must be a scalar, nonnegative float. Got {Lipschitz}.')
        self.Lipschitz = np.float64(Lipschitz)
        """Lipschitz continuity constant of the gradient."""

        if not isinstance(linear,(bool, np.bool_)):
            raise TypeError(f'linear must be boolean. Got {linear}.')
        self.linear = linear
        """boolean indicating if the functional is linear"""
        if not isinstance(separable,(bool, np.bool_)):
            raise TypeError(f'separable must be boolean. Got {separable}.')
        self.separable = separable
        """boolean indicating if the functional is separable."""
        self.convex = convex
        """boolean indicating if the functional is convex."""

        if self.separable:
            if isinstance(dom_l,np.ndarray) and isinstance(dom_u,np.ndarray) and np.any(dom_l>dom_u):
                raise ValueError('dom_l must be smaller or equal to dom_u.')
            if conj_dom_l is not None  and isinstance(conj_dom_l,np.ndarray) \
                and conj_dom_u is not None and isinstance(conj_dom_u, np.ndarray) \
                and np.any(conj_dom_l>conj_dom_u):
                raise ValueError(util.Errors.value_error(f'conj_dom_l must be smaller or equal conj_dom_u. Was given conj_dom_l = {conj_dom_l} and conj_dom_u = {conj_dom_u}',self,"__init__"))
        self.dom_l, self.dom_u, self.conj_dom_l, self.conj_dom_u = dom_l, dom_u, conj_dom_l, conj_dom_u
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

    def __call__(self, x):
        if x not in self.domain:
            raise ValueError(util.Errors.not_in_vecsp(x,self.domain,space_name="domain", add_info=f"Not able to evaluate {self} on the given vector."))
        try:
            y = self._eval(x)
        except NotImplementedError:
            y, _ = self._linearize(x)
        if not isinstance(y, (int,float)):
            raise RuntimeError(util.Errors.not_instance(x,float, add_info=f"The evaluation of the functional {self} did not return a float or int."+"\n\t"+f"x = {x}"))
        return y

    def linearize(self, x):
        r"""
        Bounds the functional from below by a linear functional at `x` given by the value at that point and a subgradient v such that

        .. math::
            F(x+ h) \geq  F(x) + vdot(v,h) for all h

        Requires the implementation of either `_subgradient` or `_linearize`.

        Parameters
        ----------
        x : in self.domain
            Element at which will be linearized

        Return
        ------
        y 
            Value of :math:`F(x)`.
        grad : in self.domain
            Subgradient of :math:`F` at :math:`x`.        
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

    def subgradient(self, x):
        r"""Returns a subgradient :math:`\xi` of the functional at `x` characterized by

        .. math::
            F(y) \geq  F(x) + vdot(\xi,y-x) for all y  

        Requires the implementation of either `_subgradient` or `_linearize`.

        Parameters
        ----------
        x : in self.domain
            Element at which will be linearized

        Returns
        -------
        grad : in self.domain
            subgradient of :math:`F` at :math:`x`.        
        """
        if x not in self.domain:
            raise ValueError(util.Errors.not_in_vecsp(x,self.domain,space_name="domain", add_info=f"Not able to compute subgradient of {self} on the given vector."))
        try:
            grad = self._subgradient(x)
        except NotImplementedError:
            _, grad = self._linearize(x)
        if grad not in self.domain:
            raise RuntimeError(util.Errors.not_in_vecsp(grad,self.domain,vec_name="gradient",space_name="domain", add_info=f"The computation of the gradient of functional {self} did not return an element in the domain." ))
        return grad
    
    def dist_subdiff(self,vstar,x):
        r"""Returns the distance of a vector :math:`v^*` to the subdifferential :math:`\partial F(x)` at x with respect 
        to the dual norm.
        Needs to be re-implemented for functionals which are not Gateaux differentiable.

        Parameters
        ----------
        vstar: in `self.domain`
            Vector of which the distance is to be determined. 
        x: in `self.domain`
            Point at which the subdifferential is evaluated
        """
        xi = self.subgradient(x)
        return self.h_domain.dual_space().norm(vstar-xi)

    def hessian(self, x,recursion_safeguard=False):
        r"""The hessian of the functional at `x` as an `regpy.operators.Operator` mapping form the 
        functionals `domain` to it self. It is defined by 

        .. math::
            F(x+h) = F(x) + (\nabla F)(x)^T h + \frac{1}{2} h^T Hess F(x) h + \mathcal{o}(\|h\|^2)

        Require either the implementation of _hessian or of _hessian_conj and _subgradient

        Parameters
        ----------
        `x` : `self.domain`
            Point in `domain` at which to compute the hessian. 

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
            raise RuntimeError(util.Errors.not_instance(h,operators.Operator,add_info=f"Computing the Hessian of {self} return a possibly non-linear operator with non matching domain and codomain that is not identical to the domain of the functional."))

    def conj_subgradient(self, xstar):
        r"""Gradient of the conjugate functional. Should not be called directly, but via self.conj.subgradient.  
        Requires the implementation of `_conj_subgradient`.       
        """
        if xstar not in self.domain:
            raise ValueError(util.Errors.not_in_vecsp(xstar,self.domain,space_name="domain", add_info=f"Not able to compute subgradient of conjugate of {self} on the given vector."))
        try:
            grad = self._conj_subgradient(xstar)
        except NotImplementedError:
            try:
                _, grad = self._conj_linearize(xstar)
            except (NotInEssentialDomainError, NotImplementedError) as e:
                raise e
        if grad not in self.domain:
            raise RuntimeError(util.Errors.not_in_vecsp(grad,self.domain,vec_name="gradient",space_name="domain", add_info=f"The computation of the gradient of conjugate functional {self} did not return an element in the domain." ))
        return grad

    def _conj_dist_subdiff(self,v,xstar):
        r"""Returns the distance of a vector :math:`v` to the subdifferential :math:`\partial F^*(x^*)` at :math:`x^*`.
        Needs to be re-implemented for functionals whose conjugate is not Gateaux differentiable.

        Parameters
        ----------
        v: in `self.domain`
            Vector of which the distance is to be determined. 
        x_star: in `self.domain`
            Point at which the subdifferential is evaluated
        """
        xi = self.conj_subgradient(xstar)
        return self.domain.norm(v-xi)        

    def conj_hessian(self,xstar, recursion_safeguard=False):
        r"""The hessian of the functional. Should not be called directly, but via self.conj.hessian.
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
            raise RuntimeError(util.Errors.not_instance(h,operators.Operator,add_info=f"Computing the Hessian of conjugate of {self} return a possibly non-linear operator with non matching domain and codomain that is not identical to the domain of the functional."))

    def conj_linearize(self, xstar):
        r"""
        Linearizes the conjugate functional :math:`F^*`. Should not be called directly, but via self.conj.linearize
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

    def proximal(self, x, tau, recursion_safeguard = False,**proximal_par):
        r"""Proximal operator 

        .. math::
            \mathrm{prox}_{\tau F}(x)=\arg \min _{v\in {\mathcal {X}}}(F(v)+{\frac{1}{2\tau}}\Vert v-x\Vert_{\mathcal {X}}^{2}).

        Requires either an implementation of `_proximal` or of `_subgradient` and `_conj_proximal`.

        Parameters
        ----------
        x : array-like
            Vector in the respective domain. Point at which to compute proximal.
        tau : scalar
            Regularization parameter for the proximal. 

        Returns
        -------
        proximal : `self.domain`
            the computed proximal at :math:`x` with parameter :math:`\tau`.
        """
        if x not in self.domain:
            raise ValueError(util.Errors.not_in_vecsp(x,self.domain,space_name="domain", add_info=f"Not able to compute subgradient of {self} on the given vector."))
        try: 
            proximal = self._proximal(x, tau,**proximal_par)
        except NotImplementedError:
            # evaluation by Moreau's identity
            if recursion_safeguard: 
                raise NotImplementedError("Neither proximal nor conj_proximal are implemented.")
            else:
                gram = self.h_domain.gram
                gram_inv = self.h_domain.gram_inv
                proximal = x - tau *gram_inv(self.conj_proximal(gram(x)/tau,1/tau,recursion_safeguard=True,**proximal_par))
        if proximal not in self.domain:
            raise ValueError(util.Errors.not_in_vecsp(proximal,self.domain,space_name="domain", add_info=f"The proximal of the functional {self} did not return somthing the domain."+"\n\t"+ f"x = {x}"))
        return proximal

    def conj_proximal(self, xstar, tau, recursion_safeguard = False,**proximal_par):
        r"""Proximal operator of conjugate functional. Should not be called directly, but via self.conj.proximal
        """
        if xstar not in self.domain:
            raise ValueError(util.Errors.not_in_vecsp(xstar,self.domain,space_name="domain", add_info=f"Not able to compute linearize of conjugate of {self} on the given vector."))
        try:
            proximal = self._conj_proximal(xstar, tau,**proximal_par)
        except NotImplementedError:
            if recursion_safeguard: 
                raise NotImplementedError("neither proximal nor conj_proximal are implemented")
            else:
                gram = self.h_domain.gram
                gram_inv = self.h_domain.gram_inv
                proximal = xstar - tau * gram(self.proximal(gram_inv(xstar/tau),1/tau,recursion_safeguard=True,**proximal_par))
        if proximal not in self.domain:
            raise ValueError(util.Errors.not_in_vecsp(proximal,self.domain,space_name="domain", add_info=f"The proximal of the conjugate of functional {self} did not return somthing the domain."+"\n\t"+f"xstar = {xstar}"))
        return proximal 
    
    def as_data_func(self,data):
        return HorizontalShiftDilation(self, data = data)

    def shift(self,v=None,data_shift=None):
        r"""Returns the functional :math:`x\mapsto F(x-v-data)` """
        return HorizontalShiftDilation(self,shift=v,data=data_shift)
    
    def dilation(self,a):
        r"""Returns the functional :math:`x\mapsto F(ax)` """
        return HorizontalShiftDilation(self,dilation=a)

    def _eval(self, x):
        raise NotImplementedError

    def _linearize(self, x):
        raise NotImplementedError

    def _subgradient(self, x):
        raise NotImplementedError

    def _hessian(self, x):
        raise NotImplementedError
    
    def _conj(self, xstar):
        raise NotImplementedError

    def _conj_linearize(self, xstar):
        raise NotImplementedError
    
    def _conj_subgradient(self, xstar):
        raise NotImplementedError

    def _conj_hessian(self, xstar):
        raise NotImplementedError

    def _proximal(self, x, tau,**proximal_par):
        raise NotImplementedError

    def _conj_proximal(self, xstar, tau,**proximal_par):
        raise NotImplementedError

    def __mul__(self, other):
        if isscalar(other) and other == 1:
            return self
        elif isinstance(other, operators.Operator):
            return Composed(self, other)
        elif other in self.domain:
            return self * operators.PtwMultiplication(self.domain, other)
        return NotImplemented

    def __rmul__(self, other):
        if isscalar(other):
            if other == 1:
                return self
            elif util.is_real_dtype(other):
                return LinearCombination((other, self))
        return NotImplemented

    def __truediv__(self, other):
        return (1 / other) * self

    def __add__(self, other):
        if isinstance(other, Functional):
            return LinearCombination(self, other)
        elif isscalar(other):
            return self if other==0 else VerticalShift(self, other)
        return NotImplemented

    def __radd__(self, other):
        return self+other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def __neg__(self):
        return (-1) * self

    def __pos__(self):
        return self

    @property
    def methods(self):
        r"""Set of strings of the names of the methods are available for a given Functional instance. 
        """
        return self._methods

    @util.memoized_property
    def conj(self):
        r"""For linear operators, this is the adjoint as a linear `regpy.operators.Operator`
        instance. Will only be computed on demand and saved for subsequent invocations.

        Returns
        -------
        Adjoint
            The adjoint as an `regpy.operators.Operator` instance.
        """
        return Conj(self)


class Conj(Functional):
    r"""An proxy class wrapping a functional. Calling it will evaluate the functional's
    conj method. This class should not be instantiated directly, but rather through the
    `Functional.conj` property of a functional.
    """

    def __init__(self, func):
        self.func = func
        if not func.convex:
            self.log.warning("Taking conjugate of a non-convex functional. The biconjugate will not coincide with the primal functional.")
        """The underlying functional."""
        super().__init__(func.domain, h_domain = func.h_domain.dual_space(),
                         Lipschitz = 1/func.convexity_param if func.convexity_param>0 else inf,
                         convexity_param = 1/func.Lipschitz if func.Lipschitz>0 else inf,
                         separable = func.separable,
                         convex = True,
                         dom_u = func.conj_dom_u if func.separable else None, 
                         dom_l = func.conj_dom_l if func.separable else None, 
                         conj_dom_u = func.dom_u if func.separable else None,  
                         conj_dom_l = func.dom_l if func.separable else None 
                         )         

    def _eval(self,x):
        return self.func._conj(x)
    # def __call__(self, x):
    #     return self.func._conj(x)

    def _conj(self, x):
        return self.func._eval(x)
    
    def _subgradient(self, x):
        return self.func.conj_subgradient(x)

    def dist_subdiff(self, v,x):
        return self.func._conj_dist_subdiff(v,x,)

    def _conj_subgradient(self, x):
        return self.func.subgradient(x)

    def _conj_dist_subdiff(self, v,x):
        return self.func.dist_subdiff(v,x)
    
    def _hessian(self, x):
        return self.func.conj_hessian(x)
    
    def _conj_hessian(self, x):
        return self.func.hessian(x)
    
    def _proximal(self, x,tau,**proximal_par):
        return self.func.conj_proximal(x,tau,**proximal_par)
    
    def _conj_proximal(self, x,tau,**proximal_par):
        return self.func.proximal(x,tau,**proximal_par)    

    @property
    def conj(self):
        return self.func
    
    @property
    def methods(self):
        return self.func._conj_methods

    def __repr__(self):
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
    gradient: domain
        The gradient of the linear functional. :math:`a=gradient` if gradient_in_dual_space == True
    domain: regpy.vecsps.VectorSpaceBase, optional
        The `regpy.vecsps.VectorSpaceBase` on which the functional is defined
    h_domain: regpy.hilbert.HilbertSpace (default: `L2(domain)`)
        Hilbert space for proximity operator
    gradient_in_dual_space: bool (default: False)
        If false, the argument gradient is considered as an element of the primal space, 
        and :math:`a = h_domain.gram(gradient)`.
    """
    def __init__(self,gradient,domain=None,h_domain = None,gradient_in_dual_space = False):
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
            self._gradient = h_domain.gram(gradient)
        super().__init__(domain=domain,h_domain=h_domain,linear=True,Lipschitz = 0,
                         separable=True,
                         convex = True,
                         dom_l=np.broadcast_to(-inf,domain.shape), dom_u = np.broadcast_to(inf,domain.shape),
                         conj_dom_l = self._gradient, conj_dom_u = self._gradient,
                         methods = {'eval','subgradient','hessian','proximal','dist_subdiff'},
                         conj_methods= {'eval','subgradient','proximal','dist_subdiff'}
                         ) 

    def _eval(self,x):
        return self.domain.vdot(self._gradient,x).real

    @property
    def gradient(self):
        return self._gradient.copy()

    def _subgradient(self,x):
        return self._gradient.copy()

    def _hessian(self, x):
        return operators.Zero(self.domain)

    def _conj(self,x_star):
        return 0 if self.domain.norm(x_star- self._gradient)==0 else inf

    def _conj_subgradient(self, xstar):
        if xstar == self._gradient:
            return self.domain.zeros()
        else:
            raise NotInEssentialDomainError('LinearFunctional.conj')     

    def _conj_dist_subdiff(self,v,xstar):
        if xstar == self.gradient:
            return 0.
        else:
            raise NotInEssentialDomainError('LinearFunctional.conj')          

    def _proximal(self, x, tau,**proximal_par):
        return x-tau*self._gradient

    def _conj_proximal(self, xstar, tau,**proximal_par):
        return self._gradient.copy()

    def dilation(self, a):
        return LinearFunctional(a*self.gradient,domain=self.domain,h_domain=self.h_domain,gradient_in_dual_space=True)
    
    def shift(self,v=None,data_shift=None):
        if(data_shift is None):
            return self - self.domain.vdot(self._gradient,v).real
        #Linear functional cannot be data functional at the moment
        return HorizontalShiftDilation(self-self.domain.vdot(self._gradient,v).real,data=data_shift)

    def __add__(self, other):
        if isinstance(other,LinearFunctional):
            return LinearFunctional(self.gradient+other.gradient,domain=self.domain, h_domain=self.h_domain,gradient_in_dual_space=True)
        elif other in self.domain:
            return LinearFunctional(self.gradient+other,domain=self.domain, h_domain=self.h_domain,gradient_in_dual_space=True)
        elif isinstance(other,SquaredNorm):
            return other+self
        else:
            return super().__add__(other)

    def __iadd__(self, other):
        if isinstance(other,LinearCombination):
            self.gradient += other.gradient
            return self
        else:
            return NotImplemented
        
    def __rmul__(self, other):
        if isscalar(other):
            return LinearFunctional(other*self.gradient,domain=self.domain, h_domain=self.h_domain,gradient_in_dual_space=True)
        else:
            return NotImplemented

    def __imul__(self, other):
        if isscalar(other):
            self.gradient *=other
            return self
        else:
            return NotImplemented

class SquaredNorm(Functional):
    r"""Functionals of the form 

    .. math::
        \mathcal{F}(x) = \frac{a}{2}\|x\|_X^2 +\langle b,x\rangle_X + c

    Here the linear term represents an inner product in the Hilbert space, not a pairing with the dual space.

    The operators `__add__` , `__iadd__` , `__mul__` , `__imul__` with `SquaredNorm`\s, `LinearFunctional`\s and scalars 
    as other arguments are overwritten to yield the expected `SquaredNorm`\s. 

    Parameters
    --------
    domain : regpy.vecsps.VectorSpaceBase
        The uncerlying vector space for the function space on which it is defined.
    h_sapce : regpy.hilbert.HilbertSpace (default: None)
        The underlying Hilbert space.
    a: float [default:1]
       coefficient of quadratic term
    b: h_space.domain [default:None]
        coefficient of linear term. In the default case it is 0.
    c: float [default: 0]
        constant term
    shift: h_space.domain [default:None]
        If not None, then we must have b is None and c==0. 
        In this case the functional is initialized as :math:`\mathcal{F}(x) = \frac{a}{2}\|x-shift-data\|^2`.
    data: h_space.domain [default:None]
        If not None, then we must have b is None and c==0. 
        In this case the functional is initialized as :math:`\mathcal{F}(x) = \frac{a}{2}\|x-shift-data\|^2`.
    """

    def __init__(self, h_space, a=1., b=None,c=0.,shift=None, data = None):
        super().__init__(h_space.vecsp,h_domain=h_space, 
                        linear = (a==0 and shift is None and c==0),
                        convex = (a>=0),
                        convexity_param = a,
                        Lipschitz = a, 
                        methods = {'eval','subgradient','hessian','proximal','dist_subdiff'},
                        conj_methods= {'eval','subgradient','hessian','proximal','dist_subdiff'}
                        )
        if not isinstance(a,(float,int)): raise ValueError(util.Errors.not_instance(a,float,add_info="for SquaredNorm `a` has to be a scalar!"))
        self.gram = self.h_domain.gram
        try:
            self.gram_inv = self.h_domain.gram_inv
        except NotImplementedError:
            self.gram_inv = None
            self.log.warning("The inverse of the gram operator is not implemented. This will lead to errors in the conjugate functionals.")
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
        
    @util.memoized_property
    def a(self):
        return self._a

    @util.memoized_property
    def b(self):
        if self.is_data_func:
            return self._b-self._a*self.data
        else:
            return self._b

    @util.memoized_property
    def c(self):
        if self.is_data_func:
            return self._c - self.h_domain.inner(self._b,self.data) + (self._a/2)* self.h_domain.norm(self.data)**2
        else:
            return self._c

    @property
    def data(self):
        return self._data
    
    @data.setter
    def data(self, new_data):
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
    def data(self):
        if self.is_data_func:
            del self._data
            del self.a; del self.b; del self.c
            self.is_data_func = False

    def as_data_func(self,data):
        self.data=data
        return self

    def _eval(self, x):
        return (self.a/2.) * self.h_domain.inner(x,x)  + self.h_domain.inner(self.b,x) + self.c
    
    def _subgradient(self, x):
        return self.gram(self.a*x+self.b)
    
    def _hessian(self,x):
        return self.a * self.gram
    
    def _proximal(self,z, tau, **proximal_par):
        if self.a<=0:
            raise NotImplementedError(util.Errors.generic_message(f"The prox operator for the SquaredNorm functional {self} is not implemented for a = {self.a}. a<=0"))
        return (1./(tau*self.a+1)) * (z-tau*self.b)
    
    def _conj(self, xstar):
        bstar = self.gram(self.b)
        if self.a>0:
            if self.gram_inv is None:
                raise RuntimeError("The inverse of the gram operator is not implemented. Thus not allowing an application of the conjugate functional.")
            return (self.h_domain.vecsp.vdot(xstar-bstar, self.gram_inv(xstar-bstar))).real / (2.*self.a) - self.c
        elif self.a==0:
            eps = 1e-10
            return -self.c if self.domain.norm(xstar-bstar)<=eps*(self.domain.norm(xstar)+eps) else inf
        else:
            return -inf

    def _conj_subgradient(self, xstar):
        bstar = self.gram(self.b)
        if self.a>0:
            if self.gram_inv is None:
                raise RuntimeError("The inverse of the gram operator is not implemented. Thus not allowing an application of the conjugate subgradient functional.")
            return (1./self.a) * self.gram_inv(xstar-bstar)
        elif self.a==0:
            return self.domain.zeros()
        else:
            return NotInEssentialDomainError

    def _conj_dist_subdiff(self,v,xstar):
        if self.a==0:
            if xstar==self.gram(self.b):
                return True 
            else:
                return NotInEssentialDomainError
        elif self.a <0:
            return RuntimeError("Can't evaluate conjugate and its subdifferential if a<0.")
        else: 
            return super()._conj_dist_subdiff(v,xstar)

    def _conj_hessian(self, xstar):
        if self.a>0:
            if self.gram_inv is None:
                raise RuntimeError("The inverse of the gram operator is not implemented. Thus not allowing an application of the conjugate hessian functional.")
            return (1./self.a) * self.gram_inv
        else:
            return NotTwiceDifferentiableError
    
    def _conj_proximal(self, zstar, tau, **proximal_par):
        if self.a<=0:
            raise NotImplementedError(util.Errors.generic_message(f"The prox operator for the conjugate of the SquaredNorm functional {self} is not implemented for a = {self.a}. it has to satisfy a>0"))
        bstar = self.gram(self.b)
        return (1./(1.+tau/self.a)) * (zstar-bstar) + bstar

    def dilation(self, dil):
        return SquaredNorm(self.h_domain,
                               a = dil**2 *self.a,
                               b = dil*self.b,
                               c = self.c 
                               )

    def shift(self, v=None,data_shift=None):
        if(data_shift is not None):
            if(data_shift not in self.domain):
                raise ValueError(util.Errors.not_in_vecsp(data_shift,self.domain,vec_name="shift data vector",space_name="domain of functional"))
            if(self.is_data_func):
                data_shift+=self.data
        #Parameters have to be used without dependence on data so that data and parameters are both updated correctly
        if(v is None):
            return SquaredNorm(self.h_domain,a = self._a,b = self._b,c = self._c,data=data_shift)
        return SquaredNorm(self.h_domain,
                               a = self._a,
                               b = self._b-self._a*v,
                               c = self._c - self.h_domain.inner(self._b,v) + (self._a/2)* self.h_domain.norm(v)**2,
                               data=data_shift
                               )

    def __add__(self, other):
        if isinstance(other, SquaredNorm) and other.h_domain==self.h_domain:
            return SquaredNorm(self.h_domain,
                               a = self.a+other.a,
                               b = self.b+other.b,
                               c = self.c+other.c 
                               )
        elif isinstance(other,LinearFunctional) and other.h_domain==self.h_domain:
            if self.gram_inv is None:
                raise RuntimeError("The inverse of the gram operator is not implemented. Thus not allowing an addition with a LinearFunctional.")
            return SquaredNorm(self.h_domain,
                               a = self.a,
                               b = self.b+self.gram_inv(other.gradient),
                               c = self.c 
                               )
        elif isscalar(other):
            return SquaredNorm(self.h_domain,
                               a = self.a,
                               b = self.b,
                               c = self.c+other 
                               )
        return super().__add__(other)

    def __iadd__(self, other):
        if isinstance(other, SquaredNorm) and other.h_domain==self.h_domain:
            self._a += other.a,
            self._b += other.b,
            self._c += other.c
            del self.a; del self.b; del self.c
            return self
        elif isinstance(other,LinearFunctional) and other.h_domain==self.h_domain:
            if self.gram_inv is None:
                raise RuntimeError("The inverse of the gram operator is not implemented. Thus not allowing an addition with a LinearFunctional.")
            self._b += self.gram_inv(other.gradient),
            del self.b
            return self
        elif isscalar(other):
            self._c += other
            del self.c
            return self
        return NotImplemented

    def __rmul__(self,other):
        if isscalar(other):
            return SquaredNorm(self.h_domain,
                               a = other*self.a,
                               b = other*self.b,
                               c = other*self.c 
                               )
        else:
            return NotImplemented

    def __imul__(self, other):
        if isscalar(other):
            self._a *=other
            self._b *=other
            self._c *=other
            del self.a; del self.b; del self.c
            return self
        return NotImplemented


class LinearCombination(Functional):
    r"""Linear combination of functionals. 

    Parameters
    ----------
    *args : (scalar, regpy.functionals.Functional) or regpy.functionals.Functional
        List of coefficients and functionals to be taken as linear combinations.
    """
    def __init__(self, *args):
        coeff_for_func = defaultdict(lambda: 0)
        for arg in args:
            if isinstance(arg, tuple):
                coeff, func = arg
            else:
                coeff, func = 1, arg
            if not isinstance(func, Functional) or not isinstance(coeff,(int,float)):
                raise ValueError(util.Errors.value_error(f"""
        The LinearCombination only takes a list of arbitrary items provided either tuples 
        (coeff,func) which are a real non-negative number and functional or a only a functional. However, you gave:
            [{"; ".join(f"({arg})" for arg in args)}]"""))
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
        self.linear_table = []
        for func, coeff in coeff_for_func.items():
            self.coeffs.append(coeff)
            self.funcs.append(func)
            self.linear_table.append(func.linear or coeff==0)        

        domains = [func.domain for func in self.funcs if func.domain]
        domain = domains[0]
        if  any(d != domain for d in domains): raise ValueError(util.Errors.generic_message(f"The domains of the functionals to be combined in a linear combination have to match. The functioanls domains are" +"\n\t" +f"domains = {domains}"))

        if self.linear_table.count(False)<=1 and self.linear_table.count(True)>=1:
            self.grad_sum = self.funcs[0].domain.zeros()
            for coeff,func,linear in zip(self.coeffs,self.funcs,self.linear_table):
                if linear:
                    self.grad_sum += coeff * func.gradient

        separable = np.all([F.separable for F in self.funcs])
        conj_dom_l, conj_dom_u = None, None
        if separable:
            if len(self.funcs) == 1:
                if self.coeffs[0]>0:
                    conj_dom_l = self.funcs[0].conj_dom_l * self.coeffs[0]
                    conj_dom_u = self.funcs[0].conj_dom_u * self.coeffs[0]
                else:
                    conj_dom_u = self.funcs[0].conj_dom_l * self.coeffs[0]
                    conj_dom_l = self.funcs[0].conj_dom_u * self.coeffs[0]
            elif self.linear_table.count(False)==0:
                conj_dom_l = self.grad_sum
                conj_dom_u = self.grad_sum 
            elif self.linear_table.count(False)==1:
                j = self.linear_table.index(False)
                conj_dom_l = self.funcs[0].domain.zeros()
                conj_dom_u = self.funcs[0].domain.zeros()
                conj_dom_l += self.funcs[j].conj_dom_l*self.coeffs[j] if self.coeffs[j] >0 else self.funcs[j].conj_dom_u*self.coeffs[j]
                conj_dom_u += self.funcs[j].conj_dom_u*self.coeffs[j] if self.coeffs[j] >0 else self.funcs[j].conj_dom_l*self.coeffs[j]
                conj_dom_l += self.grad_sum
                conj_dom_u += self.grad_sum

        methods = set.intersection(*[func.methods for func in self.funcs])
        conj_computable = (len(self.funcs) == 1) or (self.linear_table.count(False)==0) or (self.linear_table.count(False)==1)
        if not conj_computable:
            methods -= {'proximal'}
        if  conj_computable: 
            conj_methods = set.intersection(*[func.conj.methods for func in self.funcs])
        else: 
            conj_methods = set()
        all_convex = all([func.convex for func in self.funcs])
        Lipschitz = sum(coeff*fun.Lipschitz for coeff,fun in zip(self.coeffs,self.funcs) if coeff>=0.)
        Lipschitz -= sum(coeff*fun.convexity_param for coeff,fun in zip(self.coeffs,self.funcs) if coeff<0.)
        convexity_param = sum(coeff*fun.convexity_param for coeff,fun in zip(self.coeffs,self.funcs) if coeff>=0.)
        convexity_param += sum(coeff*fun.Lipschitz for coeff,fun in zip(self.coeffs,self.funcs) if coeff<0.)
        super().__init__(domain, linear = all(self.linear_table),
                         Lipschitz = Lipschitz if all_convex else np.inf,
                         convexity_param = convexity_param if (convexity_param>=0 and all_convex) else 0.,
                         convex =  all_convex and convexity_param>=0,
                         separable=separable,
                         dom_l = np.max([F.dom_l for F in self.funcs]) if separable else None,
                         dom_u = np.min([F.dom_u for F in self.funcs]) if separable else None,
                         conj_dom_l = conj_dom_l, conj_dom_u = conj_dom_u,
                         methods = methods, conj_methods = conj_methods
                         )

    def _eval(self, x,**kwargs):
        y = 0
        for coeff, func in zip(self.coeffs, self.funcs):
            y += coeff * func(x,**kwargs)
        return y

    def _linearize(self, x,**kwargs):
        y = 0
        grad = self.domain.zeros()
        for coeff, func in zip(self.coeffs, self.funcs):
            f, g = func.linearize(x,**kwargs)
            y += coeff * f
            grad += coeff * g
        return y, grad

    def _subgradient(self, x,**kwargs):
        grad = self.domain.zeros()
        for coeff, func in zip(self.coeffs, self.funcs):
            grad += coeff * func.subgradient(x,**kwargs)
        return grad

    def dist_subdiff(self, vstar, x,**kwargs):
        if len(self.funcs) == 1 or self.linear_table.count(False)==0:
            return super().dist_subdiff(vstar, x,**kwargs)
        elif self.linear_table.count(False)==1:
            j = self.linear_table.index(False)
            return self.funcs[j].dist_subdiff((vstar-self.grad_sum)/self.coeffs[j],x,**kwargs)
        else:
            return NotImplementedError

    def _hessian(self, x,**kwargs):
        if self.linear_table.count(False)==1: 
            # separate implementation of this case to be able to use inverse of hessian
            j = self.linear_table.index(False)
            return self.coeffs[j] * self.funcs[j].hessian(x,**kwargs)
        else:
            return operators.LinearCombination(
                *((coeff, func.hessian(x,**kwargs)) for coeff, func in zip(self.coeffs, self.funcs))
            )

    def _proximal(self, x, tau,**proximal_par):
        if len(self.funcs) == 1:
            return self.funcs[0].proximal(x,self.coeffs[0]*tau,**proximal_par)
        elif self.linear_table.count(False)==0:
            return x-tau*self.h_domain.gram_inv(self.grad_sum)
        elif self.linear_table.count(False)==1:
            j = self.linear_table.index(False)
            return self.funcs[j].proximal(x-tau*self.h_domain.gram_inv(self.grad_sum),self.coeffs[j]*tau,**proximal_par)
        else:
            return NotImplementedError
    
    def _conj(self, xstar,**kwargs):
        if not self.convex:
            raise RuntimeError('conj of non-convex LinearCombination not implemented.')
        if len(self.funcs) == 1:
            return self.coeffs[0]*self.funcs[0]._conj(xstar/self.coeffs[0],**kwargs)
        elif self.linear_table.count(False)==0:
            return 0 if xstar == self.grad_sum else inf
        elif self.linear_table.count(False)==1:
            j = self.linear_table.index(False)
            return self.coeffs[j]*self.funcs[j]._conj((xstar-self.grad_sum)/self.coeffs[j],**kwargs)
        else:
            return NotImplementedError

    def _conj_subgradient(self, xstar,**kwargs):
        if not self.convex:
            raise RuntimeError('conj.subgradient of non-convex linear combination not implemented.')        
        if len(self.funcs) == 1:
            return self.funcs[0]._conj_subgradient(xstar/self.coeffs[0],**kwargs)
        elif self.linear_table.count(False)==0:
            if xstar == self.grad_sum:
                return self.domain.zeros() 
            else: 
                raise NotInEssentialDomainError('Linear combination of linear functionals')
        elif self.linear_table.count(False)==1:
            j = self.linear_table.index(False)
            return self.funcs[j]._conj_subgradient((xstar-self.grad_sum)/self.coeffs[j],**kwargs)
        else:
            return NotImplementedError

    def _conj_dist_subdiff(self, v, xstar,**kwargs):
        if len(self.funcs) == 1:
            return self.funcs[0]._conj_dist_subdiff(v,xstar/self.coeffs[0],**kwargs)
        elif self.linear_table.count(False)==0:
            return self.domain.norm(xstar-self.grad_sum)
        elif self.linear_table.count(False)==1:
            j = self.linear_table.index(False)
            return self.funcs[j]._conj_dist_subdiff(v,(xstar-self.grad_sum)/self.coeffs[j],**kwargs)
        else:
            return NotImplementedError

    def _conj_hessian(self, xstar,**kwargs):
        if not self.convex:
            raise RuntimeError('conj.hessian of non-convex linear combination not implemented.') 
        if len(self.funcs) == 1:
            return (1./self.coeffs[0])*self.funcs[0]._conj_hessian(xstar/self.coeffs[0],**kwargs)
        elif self.linear_table.count(False)==0:
            raise NotTwiceDifferentiableError('Conjugate of linear combination of linear functionals')
        elif self.linear_table.count(False)==1:
            j = self.linear_table.index(False)
            return (1./self.coeffs[j])*self.funcs[j]._conj_hessian((xstar-self.grad_sum)/self.coeffs[j],**kwargs)
        else:
            return NotImplementedError

    def _conj_proximal(self, xstar,tau,**kwargs):
        if not self.convex:
            raise RuntimeError('conj.proximal of non-convex linear combination not implemented.')
        if len(self.funcs) == 1:
            return self.coeffs[0]*self.funcs[0]._conj_proximal((1./self.coeffs[0])*xstar,tau/self.coeffs[0],**kwargs)
        elif self.linear_table.count(False)==0:
            return self.grad_sum
        elif self.linear_table.count(False)==1:
            j = self.linear_table.index(False)            
            return self.coeffs[j]*self.funcs[j]._conj_proximal((1./self.coeffs[j])*(xstar-self.grad_sum),tau/self.coeffs[j],**kwargs) + self.grad_sum
        else:
            return NotImplementedError

class VerticalShift(Functional):
    r"""Shifting a functional by some offset. Should not be used directly but rather by adding some scalar to the functional.

    Parameters
    ----------
    func : regpy.functionals.Functional
        Functional to be offset.
    offset : scalar
        Reals offset added to the evaluation of the functional.
    """
    def __init__(self, func, offset):
        if not isinstance(func, Functional) or not isinstance(offset,(int,float)):
            raise ValueError(util.Errors.value_error(f""" 
            The VerticalShift only takes two arguments one AbstractFunctional 
            and one offset that is a scalar. However, you gave:
                func = {func},
                offset = {offset}."""))
        super().__init__(func.domain, linear = False, 
                         convexity_param= func. convexity_param,
                         Lipschitz = func.Lipschitz,
                         separable = func.separable,
                         convex = func.convex,
                         dom_l = func.dom_l, 
                         dom_u = func.dom_u, 
                         conj_dom_l = func.conj_dom_l, 
                         conj_dom_u = func.conj_dom_u,
                         methods = func._methods, conj_methods = func._conj_methods 
                         )
        self.func = func
        """Functional to be offset.
        """
        self.offset = offset
        """Offset added to the evaluation of the functional.
        """

    def _eval(self, x,**kwargs):
        return self.func(x,**kwargs) + self.offset

    def _linearize(self, x,**kwargs):
        return self.func._linearize(x,**kwargs)

    def _subgradient(self, x,**kwargs):
        return self.func._subgradient(x,**kwargs)

    def dist_subdiff(self, vstar,x,**kwargs):
        return self.func.dist_subdiff(vstar,x,**kwargs)    

    def _hessian(self, x,**kwargs):
        return self.func.hessian(x,**kwargs)
    
    def _proximal(self, x, tau,**proximal_par):
        return self.func.proximal(x, tau,**proximal_par)

    def _conj(self,x,**kwargs):
        return self.func.conj(x,**kwargs) - self.offset
    
    def _conj_subgradient(self, xstar,**kwargs):
        return self.func.conj.subgradient(xstar,**kwargs)

    def _conj_dist_subdiff(self, v,xstar,**kwargs):
        return self.func.conj.dist_subdiff(v,xstar,**kwargs)

    def _conj_hessian(self, xstar,**kwargs):
        return self.func.conj.hessian(xstar,**kwargs)

    def _conj_proximal(self, x, tau,**proximal_par):
        return self.func.conj.proximal(x, tau,**proximal_par)

class HorizontalShiftDilation(Functional):
    r"""Implements a horizontal shift and/or a horizontal translation of the graph of a functional :math:`F`, i.e. replaces 
    :math:`F(x)` by :math:`F(dilation(x-shift))`
    
    Parameters
    ----------
    func: Functional
        The functional to be shifted and dilated.
    dilation: float [default: 1]
        Dilation factor.
    shift: self.domain or scalar or None [default: None]
        Shift vector. The default case (None) yields the same results as shift=0, but no zero-additions are performed.
    """
    def __init__(self, func, dilation =1., shift = None, data = None):
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
            raise ValueError(util.Errors.not_in_vecsp(shift,func.domain,vec_name="data vector",space_name="domain of functional"))
        if isinstance(func,HorizontalShiftDilation):
            #prevents nested shifts
            if(func._shift_val is not None):
                if(shift is None):
                    shift=(1/func.dilation)*func._shift_val
                else:
                    shift+=(1/func.dilation)*func._shift_val
            if(func.is_data_func):
                if(data is None):
                    data=(1/func.dilation)*func.data
                else:
                    self.log.warning("The underlying functional for the HorizontalShiftDilation is already a HorizontalShiftDialtion functional with data. The provided data argument will used and the original ignored.")
            dilation*=func.dilation
            func=func.func
        self.func = func
        self.dilation = dilation
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
        if func.separable:
            dom_u = func.dom_u/dilation if self.shift_val is None else func.dom_u/dilation + self.shift_val
            dom_l = func.dom_l/dilation if self.shift_val is None else func.dom_l/dilation + self.shift_val
            conj_dom_u = func.conj_dom_u*dilation
            conj_dom_l = func.conj_dom_l*dilation
            if dilation<0:
                dom_u, dom_l = dom_l, dom_u
                conj_dom_u, conj_dom_l = conj_dom_l, conj_dom_u
        else:
            dom_u, dom_l, conj_dom_u, conj_dom_l = None, None, None, None
        super().__init__(func.domain, h_domain = func.h_domain, 
                         linear = func.linear and self.shift_val is None,
                         Lipschitz = func.Lipschitz * dilation**2,
                         convexity_param= func.convexity_param  * dilation**2,
                         separable = func.separable,
                         convex = func.convex,
                         dom_l=dom_l, dom_u=dom_u, conj_dom_l=conj_dom_l, conj_dom_u= conj_dom_u,
                         methods = func.methods, conj_methods=func._conj_methods,
                         is_data_func = True
                         )
        
    @util.memoized_property
    def shift_val(self):
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
        
    def recompute_cutoff(self):
        if self.func.separable:
            if self.dilation > 0:
                self.dom_u = self.func.dom_u/self.dilation if self.shift_val is None else self.func.dom_u/self.dilation + self.shift_val
                self.dom_l = self.func.dom_l/self.dilation if self.shift_val is None else self.func.dom_l/self.dilation + self.shift_val
            else:
                self.dom_l = self.func.dom_u/self.dilation if self.shift_val is None else self.func.dom_u/self.dilation + self.shift_val
                self.dom_u = self.func.dom_l/self.dilation if self.shift_val is None else self.func.dom_l/self.dilation + self.shift_val

    @property
    def data(self):
        if self._shifted_data_fid:
            return self.func.data
        else:
            return self._data
    
    @data.setter
    def data(self, new_data):
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
            self.recompute_cutoff()
        
    @data.deleter
    def data(self):
        if self._shifted_data_fid:
            del self.func.data
            self._shifted_data_fid = False
            self.is_data_func = False
        elif self.is_data_func:
            del self._data
            del self.shift_val
            self.is_data_func = False
            self.recompute_cutoff()

    def _eval(self, x,**kwargs):
        return self.func(self.dilation * (x if self.shift_val is None else x-self.shift_val),**kwargs)
         
    def _subgradient(self, x,**kwargs):
        return self.dilation * self.func._subgradient(self.dilation * (x if self.shift_val is None else x-self.shift_val),**kwargs)

    def dist_subdiff(self, vstar, x, **kwargs):
        return self.func.dist_subdiff(vstar/self.dilation, self.dilation * (x if self.shift_val is None else x-self.shift_val),**kwargs)

    def _hessian(self, x,**kwargs):
        return self.dilation**2 * self.func._hessian(self.dilation * (x if self.shift_val is None else x-self.shift_val),**kwargs)

    def _proximal(self, x, tau,**proximal_par):
        if self.shift_val is None:
            return              (1./self.dilation) * self.func.proximal(self.dilation*x,tau*self.dilation**2,**proximal_par)
        else:
            return self.shift_val + (1./self.dilation) * self.func.proximal(self.dilation*(x-self.shift_val),tau*self.dilation**2,**proximal_par)
    
    def _conj(self,x_star,**kwargs):
        if self.shift_val is None:
            return self.func._conj(x_star/self.dilation,**kwargs)             
        else:
            return self.func._conj(x_star/self.dilation,**kwargs) + self.domain.vdot(x_star,self.shift_val).real

    def _conj_subgradient(self,x_star,**kwargs):
        if self.shift_val is None:
            return self.func._conj_subgradient(x_star/self.dilation,**kwargs)/self.dilation             
        else:
            return self.func._conj_subgradient(x_star/self.dilation,**kwargs)/self.dilation + self.shift_val
 
    def _conj_dist_subdiff(self,v,x_star,**kwargs):
        if self.shift_val is None:
            return self.func._conj_dist_subdiff(self.dilation *v, x_star/self.dilation, **kwargs) 
        else:
            return self.func._conj_dist_subdiff(self.dilation *(v - self.shift_val), x_star/self.dilation, **kwargs)

    def _conj_hessian(self,x_star,**kwargs):
        return self.dilation**(-2)*self.func._conj_hessian(x_star/self.dilation,**kwargs)

    def _conj_proximal(self, xstar, tau,**proximal_par):
        gram = self.h_domain.gram
        if self.shift_val is None:
            return self.dilation*self.func.conj_proximal(xstar/self.dilation,
                                                  tau/self.dilation**2,
                                                  **proximal_par
                                                  )
        else:
            return self.dilation*self.func.conj_proximal(xstar/self.dilation-(tau/self.dilation)*gram(self.shift_val),
                                                  tau/self.dilation**2,
                                                  **proximal_par
                                                  )

class Composed(Functional):
    r"""Composition of an operator with a functional :math:`F\circ O`. This should not be called
    directly but rather used by multiplying the `Functional` object with an `regpy.operators.Operator`.

    Parameters
    ----------
    func : `regpy.functionals.Functional`
        Functional to be composed with. 
    op : `regpy.operators.Operator`
        Operator to be composed with. 
    op_norm : float [default: inf]
        Norm of the operator. Used only to define self.Lipschitz
    op_lower_bound : float
        Lower bound of operator: \|op(f)\|\geq op_lower_bound * \|f\|
        Used only to define self.convexity_param
    compute_op_norm : boolean
        If true the op norm will be computed using the norm method of the operator. Will only be computed if op_norm 
        is default value inf.
    norm_kwargs : dict
        possible arguments passed to the operator norm computation.
    """
    def __init__(self, func, op, op_norm = inf, op_lower_bound = 0, compute_op_norm = False, norm_kwargs = {},
                 methods = None,conj_methods=None):
        if not isinstance(func, Functional):
            raise TypeError(util.Errors.not_instance(func,Functional))
        if not isinstance(op,operators.Operator):
            raise TypeError(util.Errors.not_instance(op,operators.Operator))
        if func.domain != op.codomain:
            raise ValueError(util.Errors.not_equal(func.domain,op.codomain, add_info="Codomain of operator and domain of fucntional have to match to be composed."))
        if op_norm == inf and compute_op_norm:
            op_norm = op.norm(h_codomain = func.h_domain, **norm_kwargs)
        if conj_methods is None:
            if op.invertible:
                conj_methods = {'eval','subgradient','hessian'}
            else:
                conj_methods = set()

        super().__init__(op.domain,
                         linear = func.linear and  op.linear,
                         convexity_param= func.convexity_param * op_lower_bound**2,
                         Lipschitz= func.Lipschitz * op_norm**2,
                         convex = func.convex and op.linear, 
                         methods = {'eval','subgradient','hessian'} if methods is None else methods,
                         conj_methods = conj_methods
                         )
        if isinstance(func, type(self)):
            op = func.op * op
            func = func.func
        self.func = func
        """Functional that is composed with an Operator. 
        """
        self.op = op
        """Operator composed that is composed with a functional. 
        """

    def _eval(self, x):
        return self.func(self.op(x))

    def _linearize(self, x):
        y, deriv = self.op.linearize(x)
        z, grad = self.func.linearize(y)
        return z, deriv.adjoint(grad)

    def _subgradient(self, x):
        y, deriv = self.op.linearize(x)
        return deriv.adjoint(self.func.subgradient(y))

    def _hessian(self, x):
        if self.op.linear:
            return self.op.adjoint * self.func.hessian(self.op(x)) * self.op
        else:
            return super()._hessian(x)

    def _conj(self,x_star):
        if self.op.linear:
            return self.func._conj(self.op.inverse.adjoint(x_star))

    def _conj_subgradient(self, x_star):
        if self.op.linear:
            return self.op.inverse(self.func._conj_subgradient(self.op.inverse.adjoint(x_star)))

    def _conj_hessian(self, x_star):
        if self.op.linear:
            return self.op.inverse * self.func._conj_hessian (self.op.inverse.adjoint(x_star)) * self.op.inverse.adjoint

    def _proximal(self, x, tau, cg_params={}):
        # TODO: Remove this from the general class! All derived classes should implement their own prox!
        # In case it is a functional 1/2||Tx-g^delta||^2 can approximated by a Tikhonov solver
        if isinstance(self.func,SquaredNorm) and self.func.a == 1 and (self.func.b == 0).all() and self.func.c == 0 and isinstance(self.op,operators.OuterShift) and self.op.op.linear:
            from regpy.solvers.linear.tikhonov import TikhonovCG
            from regpy.solvers import Setting
            f, _ = TikhonovCG(
                setting=Setting(self.op.op, hilbert.L2, self.func.h_domain),
                data=-self.op.offset,
                xref=x,
                regpar=tau,
                **cg_params
            ).run()
            return f
        else:
            return NotImplementedError

class FunctionalOnDirectSum(Functional):
    r"""Helper to define Functionals with respective prox-operators on sum spaces (vecsps.DirectSum objects).
    The functionals are given as a list of the functionals on the summands of the sum space.

    .. math::
        F(x_1,... x_n) = \sum_{j=1}^n F_j(x_j)


    Parameters
    ----------
    funcs : [regpy.functionals.Functional, ...]
        List of functionals each defined on one summand of the direct sum of vector spaces.
    domain : regpy.vecsps.DirectSum
        Domain on which the combined functional is defined. 
    """
    def __init__(self, funcs,domain=None):
        if not isinstance(funcs,(list,tuple)) or any([not isinstance(f_i, Functional) for f_i in funcs]):
            raise TypeError(util.Errors.generic_message(f"To setup a FunctionalOnDirectSum the functionals have to be provided as a list or tuple of functionals."))
        if domain is not None:
            if not isinstance(domain, vecsps.DirectSum) or len(funcs)!=len(domain.summands) or any([f_i.domain != domain_i for f_i,domain_i in zip(funcs,domain.summands)]) :
                raise TypeError(util.Errors.not_instance(domain,vecsps.DirectSum,add_info="The given domain for a FunctionalOnDirectSum has to be a DirectSum with the same length of arguemnts and matching domains!"))
        else:
            domain = vecsps.DirectSum(*[f_i.domain for f_i in funcs])
        self.length = len(domain.summands)
        """Number of the summands in the direct sum domain. 
        """
        self.funcs = list(funcs)
        """List of the functionals on each summand of the direct sum domain.
        """
        separable = all([func.separable for func in funcs])
        convex = all([func.convex for func in funcs])
        dom_l = domain.join(*[func.dom_l for func in funcs]) if separable else None
        dom_u = domain.join(*[func.dom_u for func in funcs]) if separable else None
        conj_dom_l = domain.join(*[func.conj_dom_l for func in funcs]) if separable else None
        conj_dom_u = domain.join(*[func.conj_dom_u for func in funcs]) if separable else None
        methods = set.intersection(*[func.methods for func in funcs])
        conj_methods = set.intersection(*[func.conj.methods for func in funcs])
        super().__init__(domain, linear = all([func.linear for func in funcs]),
                        convexity_param = min([func.convexity_param for func in funcs]),
                        Lipschitz = max([func.Lipschitz for func in funcs]),
                        separable = separable,
                        convex = convex,
                        dom_l = dom_l, dom_u = dom_u, conj_dom_l = conj_dom_l, conj_dom_u = conj_dom_u,
                        methods=methods,conj_methods=conj_methods 
                        )

    def _eval(self, x): 
        return np.sum([f_i(x_i) for f_i,x_i in zip(self.funcs,x)])

    def _subgradient(self, x):
        return self.domain.join(*[f_i.subgradient(x_i) for f_i,x_i in zip(self.funcs,x)])

    def dist_subdiff(self,vstar, x):
        return sum([f_i.dist_subdiff(vstar_i,x_i) for f_i,vstar_i,x_i in zip(self.funcs,vstar,x)])

    def _hessian(self, x):
        return operators.DirectSum(*tuple(f_i.hessian(x_i) for f_i,x_i in zip(self.funcs,x)))

    def _proximal(self, x, tau,proximal_par_list = None):
        if proximal_par_list is None:
            return self.domain.join(*[f_i.proximal(x_i,tau) for f_i,x_i in zip(self.funcs,x)])
        elif len(proximal_par_list) != self.length:
            raise ValueError(util.Errors.generic_message("The proximal parameters in FuncitonalOnDirectSum have to be either a list of dictionaries of same length or None!"))
        return self.domain.join(*[f_i.proximal(x_i,tau, proximal_par_i) for f_i,x_i,proximal_par_i in zip(self.funcs,x,proximal_par_list)])

    def _conj(self, xstar):
        return sum([f_i.conj(xstar_i) for f_i,xstar_i in zip(self.funcs,xstar)])

    def _conj_subgradient(self, xstar):
        return self.domain.join(*[f_i.conj.subgradient(xstar_i) for f_i,xstar_i in zip(self.funcs,xstar)])

    def _conj_dist_subdiff(self,v, xstar):
        return sum([f_i.conj.dist_subdiff(v_i,xstar_i) for f_i,v_i,xstar_i in zip(self.funcs,v,xstar)])

    def _conj_hessian(self, xstar):
        return operators.DirectSum(*tuple(f_i.conj.hessian(xstar_i) for f_i,xstar_i in zip(self.funcs,xstar)))

    def _conj_proximal(self, xstar, tau,proximal_par_list = None):
        if proximal_par_list is None:
            return self.domain.join(*[f_i.conj.proximal(x_i,tau) for f_i,x_i in zip(self.funcs,xstar)])
        elif len(proximal_par_list) != self.length:
            raise ValueError(util.Errors.generic_message("The proximal parameters in conjugate of FuncitonalOnDirectSum have to be either a list of dictionaries of same length or None!"))
        return self.domain.join(*[f_i.conj.proximal(xstar_i,tau, proximal_par_i) for f_i,xstar_i,proximal_par_i in zip(self.funcs,xstar,proximal_par_list)])
    
    def __add__(self,other):
        if isinstance(other,FunctionalOnDirectSum):
            return FunctionalOnDirectSum([F+G for F,G in zip(self.funcs,other.funcs)],self.domain)
        elif isinstance(other,(int,float)):
            return FunctionalOnDirectSum([F+other for F in self.funcs])
        else: 
            return super().__add__(self,other)
        
    def __rmul__(self,other):
        if (isinstance(other,int) or isinstance(other,float)):
            return FunctionalOnDirectSum([other*F for F in self.funcs],self.domain)
        else:
            raise NotImplementedError(f"Recursive multiplication of other={other} with self={self} is not defined.")


def as_functional(func, vecsp):
    r"""Convert `func` to Functional instance on vecsp.

    - If func is a `HilbertSpace` then it generated the `SquaredNorm`.
    - If func is an Operator, it's wrapped in a `GramHilbertSpace` and then `SquaredNorm` functional.
    - If func is callable, e.g. an `hilbert.AbstractSpace` or `AbstractFunctional`, it is called on `vecsp` to construct the concrete functional or Hilbert space. In the later case the functional will be the `SquaredNorm`

    Parameters
    ----------
    func : Functional or HilbertSapce or regpy.operators.Operator or callable
        Functional or object from which to construct the Functional.
    vecsp : regpy.vecsps.VectorSpaceBase
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
