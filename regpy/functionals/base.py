from collections import defaultdict
from copy import copy
from math import inf
import logging

import numpy as np
from numpy import isscalar

from regpy import operators, util, vecsps
from regpy import hilbert

__all__ = ["as_functional","AbstractFunctional","Functional","LinearFunctional","LinearCombination","Composed","SquaredNorm","VerticalShift","HorizontalShiftDilation","FunctionalOnDirectSum"]

logging.basicConfig(
    format='%(asctime)s %(levelname)s %(name)-20s :: %(message)s'
)

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
        if isinstance(other, Functional):
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
            self.__doc__ += "-"*125 + f"\n--- Implementation for {vecsp_type.__name__} is given by {impl.__name__} with the following documentation ---\n {impl.__doc__}\n" + "-"*125
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
                assert isinstance(result, Functional)
                return result
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
            assert isinstance(func, AbstractFunctional)
            assert (isinstance(coeff,int) or isinstance(coeff,float))
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

    def __call__(self,vecsp):
        assert isinstance(vecsp, vecsps.VectorSpaceBase), "vecsp is not a VectorSpaceBase instance"
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
        assert isinstance(func, AbstractFunctional), "func not an AbstractFunctional"
        assert (isinstance(offset,int) or isinstance(offset,float)), "offset not a scalar"
        super().__init__(func.domain)
        self.func = func
        """Functional to be offset.
        """
        self.offset = offset
        """Offset added to the evaluation of the functional.
        """

    def __call__(self,vecsp):
        assert isinstance(vecsp, vecsps.VectorSpaceBase), "vecsp is not a VectorSpaceBase instance"
        return VerticalShift(func=self.func(vecsp),offset=self.offset)
    
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
        assert isinstance(func, AbstractFunctional), "func not a AbstractFunctional"
        assert isinstance(op, operators.Operator), "op not a regpy.operators.Operator"
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

    def __call__(self,vecsp):
        assert isinstance(vecsp, vecsps.VectorSpaceBase), "vecsp is not a VectorSpaceBase instance"
        assert vecsp == self.op.codomain, "domain of functional must match codomain of operator"
        return Composed(func=self.func(vecsp),op=self.op)


class Functional:
    r"""
    Base class for implementation of convex functionals. Subsclasses should at least implement the 
        `_eval` :  evaluating the funcitonal
    and 
        `_subgradient` or `_linearize` : returning a subgradient at `x`.
    
    The evaluation of a specific functional on some element of the `domain` can be done by
    simply calling the functional on that element. 
        
    Functionals can be added by taking `LinearCombination` of them. The `domain` has to be the
    same for each functional. 

    They can also be multiplied by scalars or vector of their respective `domain`or multiplied by 
    `regpy.operators.Operator`. This leads to a functional that is composed with the operator
    :math:`F\circ O` where :math:`F` is the functional and \(O)\ some operator. Multiplying by a scalar
    results in a composition with the `PtwMultiplication` operator.

    Parameters
    ----------
    domain : regpy.vecsps.VectorSpaceBase
        The uncerlying vector space for the function space on which it is defined.
    h_domain : regpy.hilbert.HilbertSpace (default: None)
        The underlying Hilbert space. The proximal mapping, the parameter of strong convexity, 
        and the Lipschitz constant are defined with respect to this Hilbert space.
        In the default case `L2(domain)` is used.   
    linear: bool [default: False]
        If true, the functional should be linear. 
    separable: bool [default: False]
        If true, the functional should be the sum of functionals acting on only one component of the input vector.
        In this case, the parameters  
    dom_u, dom_l, conj_dom_u, conj_dom_l: self.domain [default:None]
        should not be None, and they shoulspecify the essential domain of the functional by 
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
    """

    log = util.ClassLogger()

    def __init__(self, domain, h_domain=None, 
                 linear = False,
                 convexity_param=0.,
                 Lipschitz = inf,
                 separable = False,
                 dom_l=None, dom_u=None,conj_dom_l=None,conj_dom_u=None):
        assert isinstance(domain, vecsps.VectorSpaceBase)
        self.domain = domain
        """The underlying vector space."""
        self.h_domain = hilbert.as_hilbert_space(h_domain,domain) or hilbert.L2(domain)
        """The underlying Hilbert space."""
        self.linear = linear
        """boolean indicating if the functional is linear"""
        self.convexity_param = convexity_param
        """parameter of strong convexity of the functional."""
        self.Lipschitz = Lipschitz
        """Lipschitz continuity constant of the gradient."""
        self.separable = separable
        """boolean indicating if the functional is separable."""
        self.dom_l, self.dom_u, self.conj_dom_l, self.conj_dom_u = dom_l, dom_u, conj_dom_l, conj_dom_u
        """vectors indicating the essential domain of the functional and its conjugate"""

    def __call__(self, x):
        assert x in self.domain
        try:
            y = self._eval(x)
        except NotImplementedError:
            y, _ = self._linearize(x)
        assert isinstance(y, float)
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
        assert x in self.domain
        try:
            y, grad = self._linearize(x)
        except NotImplementedError:
            y = self._eval(x)
            grad = self._subgradient(x)
        assert isinstance(y, float)
        assert grad in self.domain
        return y, grad

    def subgradient(self, x):
        r"""Returns a subgradient \(\xi)\ of the functional at `x` characterized by

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
            subgradient of \(F)\ at \(x)\.        
        """
        assert x in self.domain
        try:
            grad = self._subgradient(x)
        except NotImplementedError:
            _, grad = self._linearize(x)
        assert grad in self.domain, f"The vector {grad} is not in the domain {self.domain}"
        return grad

    def is_subgradient(self,vstar,x,eps = 1e-10):
        r"""Returns `True` if \(v)\ is a subgradient of \(F)\ at \(x)\, otherwise `False`.
        Needs to be re-implemented for functionals which are not Gateaux differentiable.

        Parameters
        ----------
        eps: float (default: 1e-10)
            relative accuracy for the test
        """
        xi = self.subgradient(x)
        return self.domain.norm(vstar-xi)<=eps*(self.domain.norm(xi)+eps)

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
        assert x in self.domain
        try:
            h = self._hessian(x)
        except NotImplementedError:
            if recursion_safeguard:
                raise NotImplementedError("Neither hessian nor conj_hessian are implemented.")
            else:
                h = self.conj_hessian(self.subgradient(x),recursion_safeguard=True).inverse
        assert isinstance(h, operators.Operator)
        assert h.linear
        assert h.domain == h.codomain == self.domain
        return h

    def conj_subgradient(self, xstar):
        r"""Gradient of the conjugate functional. Should not be called directly, but via self.conj.subgradient.  
        Requires the implementation of `_conj_subgradient`.       
        """
        assert xstar in self.domain
        try:
            grad = self._conj_subgradient(xstar)
        except NotImplementedError:
            try:
                _, grad = self._conj_linearize(xstar)
            except (NotInEssentialDomainError, NotImplementedError) as e:
                raise e
        assert grad in self.domain
        return grad

    def _conj_is_subgradient(self,v,xstar,eps = 1e-10):
        r"""Returns `True` if \(v)\ is a subgradient of \(F.conj)\ at \(x)\, otherwise `False`.
        """
        xi = self.conj_subgradient(xstar)
        return self.domain.norm(v-xi)<=eps*(self.domain.norm(xi)+eps)

    def conj_hessian(self,xstar, recursion_safeguard=False):
        r"""The hessian of the functional. Should not be called directly, but via self.conj.hessian.
        """
        assert xstar in self.domain
        try:
            h = self._conj_hessian(xstar)
        except NotImplementedError:
            if recursion_safeguard:
                raise NotImplementedError("Neither hessian nor conj_hessian are implemented.")
            else:
                h = self.hessian(self.conj_subgradient(xstar),recursion_safeguard=True).inverse
        assert isinstance(h, operators.Operator)
        assert h.linear
        assert h.domain == h.codomain == self.domain
        return h

    def conj_linearize(self, xstar):
        r"""
        Linearizes the conjugate functional :math:`F^*`. Should not be called directly, but via self.conj.linearize
        """
        assert xstar in self.domain
        try:
            y, grad = self._conj_linearize(xstar)
        except NotImplementedError:
            y = self._conj(xstar)
            grad = self._conj_subgradient(xstar)
        assert isinstance(y, float)
        assert grad in self.domain
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
        assert x in self.domain
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
        assert proximal in self.domain
        return proximal

    def conj_proximal(self, xstar, tau, recursion_safeguard = False,**proximal_par):
        r"""Proximal operator of conjugate functional. Should not be called directly, but via self.conj.proximal
        """
        assert xstar in self.domain
        try:
            proximal = self._conj_proximal(xstar, tau,**proximal_par)
        except NotImplementedError:
            if recursion_safeguard: 
                raise NotImplementedError("neither proximal nor conj_proximal are implemented")
            else:
                gram = self.h_domain.gram
                gram_inv = self.h_domain.gram_inv
                proximal = xstar - tau * gram(self.proximal(gram_inv(xstar/tau),1/tau,recursion_safeguard=True,**proximal_par))
        assert proximal in self.domain
        return proximal 

    def shift(self,v):
        r"""Returns the functional \(x\mapsto F(x-v) )\ """
        return HorizontalShiftDilation(self,shift=v)
    
    def dilation(self,a):
        r"""Returns the functional \(x\mapsto F(ax) )\ """
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
        """The underlying functional."""
        super().__init__(func.domain, h_domain = func.h_domain.dual_space(),
                         Lipschitz = 1/func.convexity_param if func.convexity_param>0 else inf,
                         convexity_param = 1/func.Lipschitz if func.Lipschitz>0 else inf,
                         separable = func.separable,
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

    def is_subgradient(self, v,x,eps = 1e-10):
        return self.func._conj_is_subgradient(v,x,eps)

    def _conj_subgradient(self, x):
        return self.func.subgradient(x)

    def _conj_is_subgradient(self, v,x,eps = 1e-10):
        return self.func.is_subgradient(v,x,eps)

    def _hessian(self, x):
        return self.func.conj_hessian(x)
    
    def _conj_hessian(self, x):
        return self.func.hessian(x)
    
    def _proximal(self, x,tau,**proximal_par):
        return self.func.conj_proximal(x,tau,**proximal_par)
    
    def _conj_proximal(self, x,tau,**proximal_par):
        return self.func.proximal(x,tau,**proximal_par)    

    @property
    def conj_functional(self):
        return self.func

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
        The VectorSpaceBase on which the functional is defined
    h_domain: regpy.hilbert.HilbertSpace (default: `L2(domain)`)
        Hilbert space for proximity operator
    gradient_in_dual_space: bool (default: False)
        If false, the argument gradient is considered as an element of the primal space, 
        and :math:`a = h_domain.gram(gradient).`.
    """
    def __init__(self,gradient,domain=None,h_domain = None,gradient_in_dual_space = False):
        if domain is None:
            domain = vecsps.NumPyVectorSpace(shape=gradient.shape,dtype=float)
        assert gradient in domain
        if h_domain is None:
            h_domain = hilbert.as_hilbert_space(h_domain,domain) or hilbert.L2(domain)
        if gradient_in_dual_space:
            self._gradient = gradient
        else:
            self._gradient = h_domain.gram(gradient)
        super().__init__(domain=domain,h_domain=h_domain,linear=True,Lipschitz = 0,
                         separable=True,
                         dom_l=np.broadcast_to(-inf,domain.shape), dom_u = np.broadcast_to(inf,domain.shape),
                         conj_dom_l = self._gradient, conj_dom_u = self._gradient
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

    def _conj_is_subgradient(self,v,xstar,eps = 1e-10):
        return self.domain.norm(xstar-self.gradient)<=eps*(self.domain.norm(self.gradient)+eps)

    def _proximal(self, x, tau,**proximal_par):
        return x-tau*self._gradient

    def _conj_proximal(self, xstar, tau,**proximal_par):
        return self._gradient.copy()

    def dilation(self, a):
        return LinearFunctional(a*self.gradient,domain=self.domain,h_domain=self.h_domain,gradient_in_dual_space=True)
    
    def shift(self,v):
        return self - self.domain.vdot(self._gradient,v).real

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
        In this case the functional is initialized as \(\mathcal{F}(x) = \frac{a}{2}\|x-shift\|^2)\.
    """

    def __init__(self, h_space, a=1., b=None,c=0.,shift=None):
        super().__init__(h_space.vecsp,h_domain=h_space, 
                        linear = (a==0 and shift is None and c==0),
                        convexity_param = a,
                        Lipschitz = a
                        )
        assert isinstance(a,(float,int))
        self.gram = self.h_domain.gram
        try:
            self.gram_inv = self.h_domain.gram_inv
        except NotImplementedError:
            self.gram_inv = None
            self.log.warning("The inverse of the gram operator is not implemented. This will lead to errors in the conjugate functionals.")
        self.a=float(a)
        if shift is None:
            assert b is None or b in self.domain
            if isinstance(self.domain,vecsps.NumPyVectorSpace):
                self.b = np.broadcast_to(np.zeros(()),self.domain.shape) if b is None else b
            else:
                self.b = self.domain.zeros() if b is None else b
            assert isinstance(c,(float,int))
            self.c = float(c)
        else:
            assert shift in self.domain
            self.b = -self.a*shift
            self.c = (self.a/2.) * self.h_domain.norm(shift)**2

    def _eval(self, x):
        return (self.a/2.) * self.h_domain.norm(x)**2  + self.h_domain.inner(self.b,x) + self.c
    
    def _subgradient(self, x):
        return self.gram(self.a*x+self.b)
    
    def _hessian(self,x):
        return self.a * self.gram
    
    def _proximal(self,z, tau, **proximal_par):
        assert self.a>=0
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
        
    def _conj_is_subgradient(self,v,xstar,eps = 1e-10):
        if self.a==0:
            xi=self.gram(self.b)
            return self.domain.norm(xstar-xi)<=eps*(self.domain.norm(xi)+eps)
        elif self.a <0:
            return False
        else: 
            return super()._conj_is_subgradient(v,xstar,eps)
    
    def _conj_hessian(self, xstar):
        if self.a>0:
            if self.gram_inv is None:
                raise RuntimeError("The inverse of the gram operator is not implemented. Thus not allowing an application of the conjugate hessian functional.")
            return (1./self.a) * self.gram_inv
        else:
            return NotTwiceDifferentiableError
    
    def _conj_proximal(self, zstar, tau, **proximal_par):
        assert self.a>0
        bstar = self.gram(self.b)
        return (1./(1.+tau/self.a)) * (zstar-bstar) + bstar

    def dilation(self, dil):
        return SquaredNorm(self.h_domain,
                               a = dil**2 *self.a,
                               b = dil*self.b,
                               c = self.c 
                               )

    def shift(self, v):
        return SquaredNorm(self.h_domain,
                               a = self.a,
                               b = self.b-self.a*v,
                               c = self.c - self.h_domain.inner(self.b,v) + (self.a/2)* self.h_domain.norm(v)**2
                               )

    def __add__(self, other):
        if isinstance(other, SquaredNorm):
            return SquaredNorm(self.h_domain,
                               a = self.a+other.a,
                               b = self.b+other.b,
                               c = self.c+other.c 
                               )
        elif isinstance(other,LinearFunctional):
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
        if isinstance(other, SquaredNorm):
            self.a += other.a,
            self.b += other.b,
            self.c += other.c
            return self
        elif isinstance(other,LinearFunctional):
            if self.gram_inv is None:
                raise RuntimeError("The inverse of the gram operator is not implemented. Thus not allowing an addition with a LinearFunctional.")
            self.b += self.gram_inv(other.gradient),
            return self
        elif isscalar(other):
            self.c += other 
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
            self.a *=other
            self.b *=other
            self.c *=other
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
            assert isinstance(func, Functional)
            assert (isinstance(coeff,int) or isinstance(coeff,float)) and coeff>=0
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
        if domains:
            domain = domains[0]
            assert all(d == domain for d in domains)
        else:
            domain = None

        if self.linear_table.count(False)<=1 and self.linear_table.count(True)>=1:
            self.grad_sum = self.funcs[0].domain.zeros()
            for coeff,func,linear in zip(self.coeffs,self.funcs,self.linear_table):
                if linear:
                    self.grad_sum += coeff * func.gradient

        separable = np.all([F.separable for F in self.funcs])
        conj_dom_l, conj_dom_u = None, None
        if separable:
            if len(self.funcs) == 1:
                conj_dom_l = self.funcs[0].conj_dom_l * self.coeffs[0]
                conj_dom_u = self.funcs[0].conj_dom_u * self.coeffs[0]
            elif self.linear_table.count(False)==0:
                conj_dom_l = self.grad_sum
                conj_dom_u = self.grad_sum 
            elif self.linear_table.count(False)==1:
                j = self.linear_table.index(False)
                conj_dom_l = self.funcs[j].conj_dom_l*self.coeffs[j] + self.grad_sum
                conj_dom_u = self.funcs[j].conj_dom_u*self.coeffs[j] + self.grad_sum

        super().__init__(domain, linear = all(self.linear_table),
                         convexity_param= sum(coeff*fun.convexity_param for coeff,fun in zip(self.coeffs,self.funcs)),
                         Lipschitz = sum(coeff*fun.Lipschitz for coeff,fun in zip(self.coeffs,self.funcs)),
                         separable=separable,
                         dom_l = np.max([F.dom_l for F in self.funcs]) if separable else None,
                         dom_u = np.min([F.dom_u for F in self.funcs]) if separable else None,
                         conj_dom_l = conj_dom_l, conj_dom_u = conj_dom_u
                         )

    def _eval(self, x):
        y = 0
        for coeff, func in zip(self.coeffs, self.funcs):
            y += coeff * func(x)
        return y

    def _linearize(self, x):
        y = 0
        grad = self.domain.zeros()
        for coeff, func in zip(self.coeffs, self.funcs):
            f, g = func.linearize(x)
            y += coeff * f
            grad += coeff * g
        return y, grad

    def _subgradient(self, x):
        grad = self.domain.zeros()
        for coeff, func in zip(self.coeffs, self.funcs):
            grad += coeff * func.subgradient(x)
        return grad

    def is_subgradient(self, vstar, x, eps=1e-10):
        if len(self.funcs) == 1 or self.linear_table.count(False)==0:
            return super().is_subgradient(vstar, x, eps)
        elif self.linear_table.count(False)==1:
            j = self.linear_table.index(False)
            return self.funcs[j].is_subgradient((vstar-self.grad_sum)/self.coeffs[j],x,eps)
        else:
            return NotImplementedError

    def _hessian(self, x):
        if self.linear_table.count(False)==1: 
            # separate implementation of this case to be able to use inverse of hessian
            j = self.linear_table.index(False)
            return self.coeffs[j] * self.funcs[j].hessian(x)
        else:
            return operators.LinearCombination(
                *((coeff, func.hessian(x)) for coeff, func in zip(self.coeffs, self.funcs))
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
    
    def _conj(self, xstar):
        if len(self.funcs) == 1:
            return self.coeffs[0]*self.funcs[0]._conj(xstar/self.coeffs[0])
        elif self.linear_table.count(False)==0:
            return 0 if xstar == self.grad_sum else inf
        elif self.linear_table.count(False)==1:
            j = self.linear_table.index(False)
            return self.coeffs[j]*self.funcs[j]._conj((xstar-self.grad_sum)/self.coeffs[j])
        else:
            return NotImplementedError

    def _conj_subgradient(self, xstar):
        if len(self.funcs) == 1:
            return self.funcs[0]._conj_subgradient(xstar/self.coeffs[0])
        elif self.linear_table.count(False)==0:
            if xstar == self.grad_sum:
                return self.domain.zeros() 
            else: 
                raise NotInEssentialDomainError('Linear combination of linear functionals')
        elif self.linear_table.count(False)==1:
            j = self.linear_table.index(False)
            return self.funcs[j]._conj_subgradient((xstar-self.grad_sum)/self.coeffs[j])
        else:
            return NotImplementedError

    def _conj_is_subgradient(self, v, xstar,eps = 1e-10):
        if len(self.funcs) == 1:
            return self.funcs[0]._conj_is_subgradient(v,xstar/self.coeffs[0],eps)
        elif self.linear_table.count(False)==0:
            return self.domain.norm(xstar-self.grad_sum)<=eps*(self.domain.norm(self.grad_sum)+eps)
        elif self.linear_table.count(False)==1:
            j = self.linear_table.index(False)
            return self.funcs[j]._conj_is_subgradient(v,(xstar-self.grad_sum)/self.coeffs[j],eps)
        else:
            return NotImplementedError
    
    def _conj_hessian(self, xstar):
        if len(self.funcs) == 1:
            return (1./self.coeffs[0])*self.funcs[0]._conj_hessian(xstar/self.coeffs[0])
        elif self.linear_table.count(False)==0:
            raise NotTwiceDifferentiableError('Conjugate of linear combination of linear functionals')
        elif self.linear_table.count(False)==1:
            j = self.linear_table.index(False)
            return (1./self.coeffs[j])*self.funcs[j]._conj_hessian((xstar-self.grad_sum)/self.coeffs[j])
        else:
            return NotImplementedError

    def _conj_proximal(self, xstar,tau):
        if len(self.funcs) == 1:
            return self.coeffs[0]*self.funcs[0]._conj_proximal((1./self.coeffs[0])*xstar,tau/self.coeffs[0])
        elif self.linear_table.count(False)==0:
            return self.grad_sum
        elif self.linear_table.count(False)==1:
            j = self.linear_table.index(False)            
            return self.coeffs[j]*self.funcs[j]._conj_proximal((1./self.coeffs[j])*(xstar-self.grad_sum),tau/self.coeffs[j]) + self.grad_sum
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
        assert isinstance(func, Functional)
        assert isinstance(offset,int) or isinstance(offset,float)
        super().__init__(func.domain, linear = False, 
                         convexity_param= func. convexity_param,
                         Lipschitz = func.Lipschitz,
                         separable = func.separable,
                         dom_l = func.dom_l, 
                         dom_u = func.dom_u, 
                         conj_dom_l = func.conj_dom_l, 
                         conj_dom_u = func.conj_dom_u
                         )
        self.func = func
        """Functional to be offset.
        """
        self.offset = offset
        """Offset added to the evaluation of the functional.
        """

    def _eval(self, x):
        return self.func(x) + self.offset

    def _linearize(self, x):
        return self.func._linearize(x)

    def _subgradient(self, x):
        return self.func._subgradient(x)

    def is_subgradient(self, vstar,x,eps = 1e-10):
        return self.func.is_subgradient(vstar,x,eps)
    
    def _hessian(self, x):
        return self.func.hessian(x)
    
    def _proximal(self, x, tau,**proximal_par):
        return self.func.proximal(x, tau,**proximal_par)

    def _conj(self,x):
        return self.func.conj(x) - self.offset
    
    def _conj_subgradient(self, xstar):
        return self.func.conj.subgradient(xstar)

    def _conj_is_subgradient(self, v,xstar,eps = 1e-10):
        return self.func.conj.is_subgradient(v,xstar,eps)

    def _conj_hessian(self, xstar):
        return self.func.conj.hessian(xstar)

    def _conj_proximal(self, x, tau,**proximal_par):
        return self.func.conj.proximal(x, tau,**proximal_par)

class HorizontalShiftDilation(Functional):
    r"""Implements a horizontal shift and/or a horizontal translation of the graph of a functional :math:`F`, i.e. replaces 
    :math:`F(x)` by \(F(dilation(x-shift)))
    
    Parameters
    ----------
    F: Functional
        The functional to be shifted and dilated.
    dilation: float [default: 1]
        Dilation factor.
    shift: self.domain or scalar or None [default: None]
        Shift vector. The default case (None) yields the same results as shift=0, but no zero-additions are performed.
    """
    def __init__(self, F, dilation =1., shift = None):
        if np.isscalar(shift):
            shift = np.broadcast_to(shift,F.domain.shape)
        assert shift is None or shift in F.domain
        assert isinstance(dilation,int) or isinstance(dilation,float)        
        if F.separable:
            dom_u = F.dom_u/dilation if shift is None else F.dom_u/dilation + shift
            dom_l = F.dom_l/dilation if shift is None else F.dom_l/dilation + shift
            conj_dom_u = F.conj_dom_u*dilation
            conj_dom_l = F.conj_dom_l*dilation
        else:
            dom_u, dom_l, conj_dom_u, conj_dom_l = None, None, None, None
        super().__init__(F.domain, h_domain = F.h_domain, 
                         linear = F.linear and shift is None,
                         Lipschitz = F.Lipschitz * dilation**2,
                         convexity_param= F.convexity_param  * dilation**2,
                         separable = F.separable,
                         dom_l=dom_l, dom_u=dom_u, conj_dom_l=conj_dom_l, conj_dom_u= conj_dom_u
                         )
        self.F = F
        self.dilation = dilation
        self.shift = shift

    def _eval(self, x):
        return self.F(self.dilation * (x if self.shift is None else x-self.shift))
         
    def _subgradient(self, x):
        return self.dilation * self.F._subgradient(self.dilation * (x if self.shift is None else x-self.shift))

    def is_subgradient(self, vstar, x, eps= 1e-10):
        return self.F.is_subgradient(vstar/self.dilation, self.dilation * (x if self.shift is None else x-self.shift),eps)

    def _hessian(self, x):
        return self.dilation**2 * self.F._hessian(self.dilation * (x if self.shift is None else x-self.shift))

    def _proximal(self, x, tau,**proximal_par):
        if self.shift is None:
            return              (1./self.dilation) * self.F.proximal(self.dilation*x,tau*self.dilation**2,**proximal_par)
        else:
            return self.shift + (1./self.dilation) * self.F.proximal(self.dilation*(x-self.shift),tau*self.dilation**2,**proximal_par)
    
    def _conj(self,x_star):
        if self.shift is None:
            return self.F._conj(x_star/self.dilation)             
        else:
            return self.F._conj(x_star/self.dilation) + self.domain.vdot(x_star,self.shift).real

    def _conj_subgradient(self,x_star):
        if self.shift is None:
            return self.F._conj_subgradient(x_star/self.dilation)/self.dilation             
        else:
            return self.F._conj_subgradient(x_star/self.dilation)/self.dilation + self.shift

    def _conj_is_subgradient(self,v,x_star, eps= 1e-10):
        if self.shift is None:
            return self.F._conj_is_subgradient(self.dilation *v, x_star/self.dilation, eps) 
        else:
            return self.F._conj_is_subgradient(self.dilation *(v - self.shift), x_star/self.dilation, eps)

    def _conj_hessian(self,x_star):
        return self.dilation**(-2)*self.F._conj_hessian(x_star/self.dilation)

    def _conj_proximal(self, xstar, tau,**proximal_par):
        gram = self.h_domain.gram
        return self.dilation*self.F.conj_proximal(xstar/self.dilation-(tau/self.dilation)*gram(self.shift),
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
    """
    def __init__(self, func, op,op_norm = inf, op_lower_bound = 0):
        assert isinstance(func, Functional)
        assert isinstance(op, operators.Operator)
        assert func.domain == op.codomain
        super().__init__(op.domain,
                         linear = func.linear,
                         convexity_param= func.convexity_param * op_lower_bound**2,
                         Lipschitz= func.Lipschitz * op_norm**2   
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
            return self.op.adjoint * self.func.hessian(x) * self.op
        else:
            # TODO this can be done slightly more efficiently
            return super()._hessian(x)

    def _conj(self,x):
        if self.op.linear:
            return self.func._conj(self.op.adjoint.inverse(x))

    def _proximal(self, x, tau, cg_params={}):
        # In case it is a functional 1/2||Tx-g^delta||^2 can approximated by a Tikhonov solver
        if isinstance(self.func,SquaredNorm) and self.func.a == 1 and (self.func.b == 0).all() and self.func.c == 0 and isinstance(self.op,operators.OuterShift) and self.op.op.linear:
            from regpy.solvers.linear.tikhonov import TikhonovCG
            from regpy.solvers import RegularizationSetting
            f, _ = TikhonovCG(
                setting=RegularizationSetting(self.op.op, hilbert.L2, self.func.h_domain),
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
        assert isinstance(funcs,list) and all([isinstance(f_i, Functional) for f_i in funcs])
        if domain is not None:
            assert isinstance(domain, vecsps.DirectSum)
            assert len(funcs)==len(domain.summands)
            assert all([f_i.domain == domain_i for f_i,domain_i in zip(funcs,domain.summands)]) 
        else:
            domain = vecsps.DirectSum(*[f_i.domain for f_i in funcs])
        self.length = len(domain.summands)
        """Number of the summands in the direct sum domain. 
        """
        self.funcs = funcs
        """List of the functionals on each summand of the direct sum domain.
        """
        super().__init__(domain, linear = all([func.linear for func in funcs]),
                        convexity_param = min([func.convexity_param for func in funcs]),
                        Lipschitz = max([func.Lipschitz for func in funcs]),
                        separable = all([func.separable for func in funcs]),
                        dom_l = domain.join(*[func.dom_l for func in funcs]),
                        dom_u = domain.join(*[func.dom_u for func in funcs]),
                        conj_dom_l = domain.join(*[func.conj_dom_l for func in funcs]),
                        conj_dom_u = domain.join(*[func.conj_dom_u for func in funcs]),                        
                        )

    def _eval(self, x):
        toret = 0 
        for f_i,x_i in zip(self.funcs,x):
            toret += f_i(x_i)
        return toret

    def _subgradient(self, x):
        return self.domain.join([f_i.subgradient(x_i) for f_i,x_i in zip(self.funcs,x)])

    def _is_subgradient(self,vstar, x, eps= 1e-10):
        assert vstar in self.domain and x in self.domain
        return all([f_i.is_subgradient(vstar_i,x_i,eps) for f_i,vstar_i,x_i in zip(self.funcs,vstar,x)])

    def _hessian(self, x):
        return operators.DirectSum(*tuple(f_i.hessian(x_i) for f_i,x_i in zip(self.funcs,x)))

    def _proximal(self, x, tau,proximal_par_list = None):
        if proximal_par_list is None:
            proximal_par_list = [{}] *self.length
        else:
            assert len(proximal_par_list) == self.length
        return self.domain.join(*[f_i.proximal(x_i,tau, proximal_par_i) for f_i,x_i,proximal_par_i in zip(self.funcs,x,proximal_par_list)])

    def _conj(self, xstar):
        return sum([f_i.conj(xstar_i) for f_i,xstar_i in (self.funcs,xstar)])

    def _conj_subgradient(self, xstar):
        return self.domain.join(*[f_i.conj.subgradient(xstar_i) for f_i,xstar_i in zip(self.funcs,xstar)])

    def _conj_is_subgradient(self,v, xstar, eps= 1e-10):
        assert v in self.domain and xstar in self.domain
        return all([f_i.conj.is_subgradient(v_i,xstar_i,eps) for f_i,v_i,xstar_i in zip(self.funcs,v,xstar)])

    def _conj_hessian(self, xstar):
        return operators.DirectSum(*tuple(f_i.conj.hessian(xstar_i) for f_i,xstar_i in zip(self.funcs,xstar)))

    def _conj_proximal(self, xstar, tau,proximal_par_list = None):
        if proximal_par_list is None:
            proximal_par_list = [{}] *self.length
        else:
            assert len(proximal_par_list) == self.length
        return self.domain.join(*[f_i.conj.proximal(xstar_i,tau, proximal_par_i) for f_i,xstar_i,proximal_par_i in zip(self.funcs,xstar,proximal_par_list)])
    
    def __add__(self,other):
        if isinstance(other,FunctionalOnDirectSum):
            return FunctionalOnDirectSum([F+G for F,G in zip(self.funcs,other.funcs)],self.domain)
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
    vecsp : VectorSpaceBase
        Underlying vector space for the functional. 

    Returns
    -------
    Functional
        Constructed Functional on the underlying vectorspace. 
    """
    from regpy.operators import Operator  # imported here to avoid circular dependency
    if not isinstance(func,Functional):
        if isinstance(func, operators.Operator):
            func = SquaredNorm(hilbert.GramHilbertSpace(func))
        elif callable(func):
            func = func(vecsp)
        if isinstance(func, hilbert.HilbertSpace):
            func = SquaredNorm(func)
    assert isinstance(func,Functional)
    if func.domain != vecsp:
        raise ValueError(f"Given Vector space {vecsp} and the domain of the functional {func.domain} do not match.")
    elif isinstance(func,Composed) and func.func.domain != vecsp:
        raise ValueError(f"Given Vector space {vecsp} and the domain of the composed functional {func.func.domain} do not match.")
    return func



