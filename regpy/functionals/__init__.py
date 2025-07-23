from collections import defaultdict

from copy import copy

import numpy as np

from regpy import operators, util, vecsps
from regpy import hilbert

class NotInEssentialDomainError(Exception):
    r"""
    Raised if value of the functional is np.infty at given argument. In this case the subdifferential is empty. 
    """
    pass

class NotTwiceDifferentiableError(Exception):
    r"""
    Raised if hessian is called at an argument where a functional is not twice differentiable. 
    """
    pass

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

    They can also be multiplied by scalars or `np.ndarrays`of `domain.shape`or multiplied by 
    `regpy.operators.Operator`. This leads to a functional that is composed with the operator
    :math:`F\circ O` where :math:`F` is the functional and \(O)\ some operator. Multiplying by a scalar
    results in a composition with the `PtwMultiplication` operator.

    Parameters
    ----------
    domain : regpy.vecsps.VectorSpace
        The uncerlying vector space for the function space on which it is defined.
    h_domain : regpy.hilbert.HilbertSpace (default: None)
        The underlying Hilbert space. The proximal mapping, the parameter of strong convexity, 
        and the Lipschitz constant are defined with respect to this Hilbert space.
        In the default case `L2(domain)` is used.   
    linear: bool [default: False]
        If true, the functional should be linear. 
    convexity_param: float [default: 0]
        parameter of strong convexity of the functional. 
        0 if the functional is not strongly convex.
    Lipschitz: float [default: np.inf]
        Lipschitz continuity constant of the gradient.  
        np.inf the gradient is not Lipschitz continuous.
    """
    def __init__(self, domain, h_domain=None, 
                 linear = False,
                 convexity_param=0.,
                 Lipschitz = np.inf):
        assert isinstance(domain, vecsps.VectorSpace)
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
            F(x+ h) \geq  F(x) + \np.vdot(v,h) for all h

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
            F(y) \geq  F(x) + np.vdot(\xi,y-x) for all y  

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
        assert grad in self.domain
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
        return np.linalg.norm(vstar-xi)<=eps*(np.linalg.norm(xi)+eps)

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
            _, grad = self._conj_linearize(xstar)
        assert grad in self.domain
        return grad

    def _conj_is_subgradient(self,v,xstar,eps = 1e-10):
        r"""Returns `True` if \(v)\ is a subgradient of \(F.conj)\ at \(x)\, otherwise `False`.
        """
        xi = self.conj_subgradient(xstar)
        return np.linalg.norm(v-xi)<=eps*(np.linalg.norm(xi)+eps)

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
        x : `self.domain`
            Point at which to compute proximal.
        tau : `np.number`
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
        if np.isscalar(other) and other == 1:
            return self
        elif isinstance(other, operators.Operator):
            return Composed(self, other)
        elif np.isscalar(other) or isinstance(other, np.ndarray):
            return self * operators.PtwMultiplication(self.domain, other)
        return NotImplemented

    def __rmul__(self, other):
        if np.isscalar(other):
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
        elif np.isscalar(other):
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
                         Lipschitz = 1/func.convexity_param if func.convexity_param>0 else np.inf,
                         convexity_param = 1/func.Lipschitz if func.Lipschitz>0 else np.inf
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
    domain: regpy.vecsps.VectorSpace, optional
        The VectorSpace on which the functional is defined
    h_domain: regpy.hilbert.HilbertSpace (default: `L2(domain)`)
        Hilbert space for proximity operator
    gradient_in_dual_space: bool (default: False)
        If false, the argument gradient is considered as an element of the primal space, 
        and :math:`a = h_domain.gram(gradient).`.
    """
    def __init__(self,gradient,domain=None,h_domain = None,gradient_in_dual_space = False):
        if domain is None:
            domain = vecsps.VectorSpace(shape=gradient.shape,dtype=float)
        super().__init__(domain=domain,h_domain=h_domain,linear=True,Lipschitz = 0)
        assert gradient in self.domain
        if gradient_in_dual_space:
            self._gradient = gradient
        else:
            self._gradient = self.h_domain.gram(gradient)

    def _eval(self,x):
        return np.vdot(self._gradient,x).real

    @property
    def gradient(self):
        return self._gradient.copy()

    def _subgradient(self,x):
        return self._gradient.copy()

    def _hessian(self, x):
        return operators.Zero(self.domain)

    def _conj(self,x_star):
        return 0 if np.linalg.norm(x_star- self._gradient)==0 else np.inf

    def _conj_subgradient(self, xstar):
        if xstar == self._gradient:
            return self.domain.zeros()
        else:
            raise NotInEssentialDomainError('LinearFunctional.conj')

    def _conj_is_subgradient(self,v,xstar,eps = 1e-10):
        return np.linalg.norm(xstar-self.gradient)<=eps*(np.linalg.norm(self.gradient)+eps)

    def _proximal(self, x, tau,**proximal_par):
        return x-tau*self._gradient

    def _conj_proximal(self, xstar, tau,**proximal_par):
        return self._gradient.copy()

    def dilation(self, a):
        return LinearFunctional(a*self.gradient,domain=self.domain,h_domain=self.h_domain,gradient_in_dual_space=True)
    
    def shift(self,v):
        return self - np.vdot(self._gradient,v).real

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
        if np.isscalar(other):
            return LinearFunctional(other*self.gradient,domain=self.domain, h_domain=self.h_domain,gradient_in_dual_space=True)
        else:
            return NotImplemented

    def __imul__(self, other):
        if np.isscalar(other):
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
    domain : regpy.vecsps.VectorSpace
        The uncerlying vector space for the function space on which it is defined.
    h_domain : regpy.hilbert.HilbertSpace (default: None)
        The underlying Hilbert space.
    a: float [default:1]
       coefficient of quadratic term
    b: h_domain.domain [default:None]
        coefficient of linear term. In the default case it is 0.
    c: float [default: 0]
        constant term
    shift: h_domain.domain [default:None]
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
        self.gram_inv = self.h_domain.gram_inv
        self.a=float(a)
        if shift is None:
            assert b is None or b in self.domain
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
            return np.real(np.vdot(xstar-bstar, self.gram_inv(xstar-bstar))) / (2.*self.a) - self.c
        elif self.a==0:
            eps = 1e-10
            return -self.c if np.linalg.norm(xstar-bstar)<=eps*(np.linalg.norm(xstar)+eps) else np.inf
        else:
            return -np.inf

    def _conj_subgradient(self, xstar):
        bstar = self.gram(self.b)
        if self.a>0:
            return (1./self.a) * self.gram_inv(xstar-bstar)
        elif self.a==0:
            return self.domain.zeros()
        else:
            return NotInEssentialDomainError
        
    def _conj_is_subgradient(self,v,xstar,eps = 1e-10):
        if self.a==0:
            xi=self.gram(self.b)
            return np.linalg.norm(xstar-xi)<=eps*(np.linalg.norm(xi)+eps)
        elif self.a <0:
            return False
        else: 
            return super()._conj_is_subgradient(v,xstar,eps)
    
    def _conj_hessian(self, xstar):
        if self.a>0:
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
            return SquaredNorm(self.h_domain,
                               a = self.a,
                               b = self.b+self.gram_inv(other.gradient),
                               c = self.c 
                               )
        elif np.isscalar(other):
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
            self.b += self.gram_inv(other.gradient),
            return self
        elif np.isscalar(other):
            self.c += other 
            return self
        return NotImplemented

    def __rmul__(self,other):
        if np.isscalar(other):
            return SquaredNorm(self.h_domain,
                               a = other*self.a,
                               b = other*self.b,
                               c = other*self.c 
                               )
        else:
            return NotImplemented

    def __imul__(self, other):
        if np.isscalar(other):
            self.a *=other
            self.b *=other
            self.c *=other
            return self
        return NotImplemented



class LinearCombination(Functional):
    r"""Linear combination of functionals. 

    Parameters
    ----------
    *args : (np.number, regpy.functionals.Functional) or regpy.functionals.Functional
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
            assert np.isscalar(coeff) and util.is_real_dtype(coeff) and coeff>=0
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

        super().__init__(domain, linear = all(self.linear_table),
                         convexity_param= sum(coeff*fun.convexity_param for coeff,fun in zip(self.coeffs,self.funcs)),
                         Lipschitz = sum(coeff*fun.Lipschitz for coeff,fun in zip(self.coeffs,self.funcs))
                         )

        if self.linear_table.count(False)<=1 and self.linear_table.count(True)>=1:
            self.grad_sum = self.domain.zeros()
            for coeff,func,linear in zip(self.coeffs,self.funcs,self.linear_table):
                if linear:
                    self.grad_sum += coeff * func.gradient

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
            return 0 if xstar == self.grad_sum else np.inf
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
            return np.linalg.norm(xstar-self.grad_sum)<=eps*(np.linalg.norm(self.grad_sum)+eps)
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
    offset : np.number
        Offset added to the evaluation of the functional.
    """
    def __init__(self, func, offset):
        assert isinstance(func, Functional)
        assert np.isscalar(offset) and util.is_real_dtype(offset)
        super().__init__(func.domain, linear = False, 
                         convexity_param= func. convexity_param,
                         Lipschitz = func.Lipschitz
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
    shift: self.domain [default: None]
        Shift vector. 0 in the default case.
    """
    def __init__(self, F, dilation =1., shift = None):
        super().__init__(F.domain, h_domain = F.h_domain, 
                         linear = F.linear and shift is None,
                         Lipschitz = F.Lipschitz * dilation**2,
                         convexity_param= F.convexity_param  * dilation**2
                         )
        assert shift is None or shift in self.domain
        assert np.isscalar(dilation) and util.is_real_dtype(dilation)
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
            return self.F._conj(x_star/self.dilation) + np.vdot(x_star,self.shift).real

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
    op_norm : float [default: np.inf]
        Norm of the operator. Used only to define self.Lipschitz
    op_lower_bound : float
        Lower bound of operator: \|op(f)\|\geq op_lower_bound * \|f\|
        Used only to define self.convexity_param
    """
    def __init__(self, func, op,op_norm = np.inf, op_lower_bound = 0):
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
        if isinstance(self.func,HilbertNormGeneric) and isinstance(self.op,operators.OuterShift) and self.op.op.linear:
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

class AbstractFunctionalBase:
    r"""Class representing abstract functionals without reference to a concrete implementation.

    Abstract functionals do not have elements, properties or any other structure, their sole purpose is
    to pick the proper concrete implementation for a given vector space.
    """

    def __mul__(self, other):
        if np.isscalar(other) and other == 1:
            return self
        elif isinstance(other, operators.Operator):
            return AbstractComposed(self, other)
        elif np.isscalar(other) or isinstance(other, np.ndarray):
            return self * operators.PtwMultiplication(self.domain, other)
        return NotImplemented

    def __rmul__(self, other):
        if np.isscalar(other):
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
        elif np.isscalar(other):
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
        r"""Either registers a new implementation on a specific `regpy.vecsps.VectorSpace` 
        for a given Abstract functional or returns as decorator that can output any implementation
        option for a given vector space.

        Parameters
        ----------
        vecsp_type : `regpy.vecsps.VectorSpace`
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

L1 = AbstractFunctional('L1')
TV = AbstractFunctional('TV')
HilbertNorm = AbstractFunctional('HilbertNorm')

class AbstractLinearCombination(AbstractFunctional):
    r"""Linear combination of abstract functionals. 

    Parameters
    ----------
    *args : (np.number, regpy.functionals.AbstractFunctional) or regpy.functionals.AbstractFunctional
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
            assert np.isscalar(coeff) and util.is_real_dtype(coeff)
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
        assert isinstance(vecsp, vecsps.VectorSpace), "vecsp is not a VectorSpace instance"
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
    offset : np.number
        Offset added to the evaluation of the functional.
    """
    def __init__(self, func, offset):
        assert isinstance(func, AbstractFunctional), "func not an AbstractFunctional"
        assert np.isscalar(offset) and util.is_real_dtype(offset), "offset not a scalar"
        super().__init__(func.domain)
        self.func = func
        """Functional to be offset.
        """
        self.offset = offset
        """Offset added to the evaluation of the functional.
        """

    def __call__(self,vecsp):
        assert isinstance(vecsp, vecsps.VectorSpace), "vecsp is not a VectorSpace instance"
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
        assert isinstance(vecsp, vecsps.VectorSpace), "vecsp is not a VectorSpace instance"
        assert vecsp == self.op.codomain, "domain of functional must match codomain of operator"
        return Composed(func=self.func(vecsp),op=self.op)
    

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
        if domain is not None:
            assert isinstance(domain, vecsps.DirectSum())
        else:
            domain = vecsps.DirectSum(*tuple())
        self.length = len(domain.summands)
        """Number of the summands in the direct sum domain. 
        """
        for i in range(self.length):
            assert isinstance(funcs[i], Functional)
            assert funcs[i].domain == domain.summands[i] 
        self.funcs = funcs
        """List of the functionals on each summand of the direct sum domain.
        """
        super().__init__(domain, linear = np.all([func.linear for func in funcs]),
                        convexity_param = np.min([func.convexity_param for func in funcs]),
                        Lipschitz = np.max([func.Lipschitz for func in funcs])
                        )

    def _eval(self, x):
        splitted = self.domain.split(x)
        toret = 0 
        for i in range(self.length):
            toret += self.funcs[i](splitted[i])
        return toret

    def _subgradient(self, x):
        splitted = self.domain.split(x)
        subgradients = []
        for i in range(self.length):
            subgradients.append( self.funcs[i].subgradient(splitted[i]) )
        return np.asarray(subgradients).flatten()

    def is_subgradient(self,v_star, x, eps= 1e-10):
        x_splitted = self.domain.split(x)
        vstar_splitted = self.domain.split(v_star)
        res = True
        for i in range(self.length):
            if not self.funcs[i].is_subgradient(vstar_splitted[i],x_splitted[i],eps):
                res = False
        return res

    def _hessian(self, x):
        splitted = self.domain.split(x)
        return operators.DirectSum(*tuple(self.funcs[i].hessian(splitted[i]) for i in range(self.length)))

    def _proximal(self, x, tau,proximal_par_list = None):
        splitted = self.domain.split(x)
        if proximal_par_list is None:
            proximal_par_list = [{}] *self.length
        proximals = []
        for i in range(self.length):
            proximals.append( self.funcs[i].proximal(splitted[i], tau,proximal_par_list[i]) )
        return np.asarray(proximals).flatten()

    def _conj(self, xstar):
        splitted = self.domain.split(xstar)
        toret = 0 
        for i in range(self.length):
            toret += self.funcs[i].conj(splitted[i])
        return toret

    def _conj_subgradient(self, xstar):
        splitted = self.domain.split(xstar)
        subgradients = []
        for i in range(self.length):
            subgradients.append( self.funcs[i].conj.subgradient(splitted[i]) )
        return np.asarray(subgradients).flatten()

    def _conj_is_subgradient(self,v, xstar, eps= 1e-10):
        xstar_splitted = self.domain.split(xstar)
        v_splitted = self.domain.split(v)
        res = True
        for i in range(self.length):
            if not self.funcs[i].conj.is_subgradient(v_splitted[i],xstar_splitted[i], eps):
                res = False
        return res

    def _conj_hessian(self, xstar):
        splitted = self.domain.split(xstar)
        return operators.DirectSum(*tuple(self.funcs[i].conj.hessian(splitted[i]) for i in range(self.length)))

    def _conj_proximal(self, xstar, tau,proximal_par_list = None):
        splitted = self.domain.split(xstar)
        if proximal_par_list is None:
            proximal_par_list = [{}] *self.length
        proximals = []
        for i in range(self.length):
            proximals.append( self.funcs[i].conj.proximal(splitted[i], tau,proximal_par_list[i]) )
        return np.asarray(proximals).flatten()
    
    def __add__(self,other):
        if isinstance(other,FunctionalOnDirectSum):
            return FunctionalOnDirectSum([F+G for F,G in zip(self.funcs,other.funcs)],self.domain)
        else: 
            return super().__add__(self,other)
        
    def __rmul__(self,other):
        if np.isscalar(other):
            return FunctionalOnDirectSum([other*F for F in self.funcs],self.domain)

class HilbertNormGeneric(Functional):
    r"""Generic implementation of the HilbertNorm :math:`1/2*\Vert x\Vert^2`. Proximal operator defined on `h_space`.

    Parameters
    ----------
    h_space : regpy.hilbert.HilbertSpace
        Hilbert space used for norm. 
    h_domain : regpy.hilbert.HilbertSpace
        Hilbert Space wrt the proximal operator gets computed. (Defaults : h_space)
    """
    def __init__(self, h_space, h_domain=None):
        assert isinstance(h_space, hilbert.HilbertSpace)
        super().__init__(h_space.vecsp, h_domain= h_domain or h_space,
                         convexity_param=1., Lipschitz=1.)
        self.h_space = h_space
        """ Hilbert space used for norm.
        """

    def _eval(self, x):
        return np.real(np.vdot(x, self.h_space.gram(x))) / 2

    def _linearize(self, x):
        gx = self.h_space.gram(x)
        y = np.real(np.vdot(x, gx)) / 2
        return y, gx

    def _subgradient(self, x):
        return self.h_space.gram(x)

    def _hessian(self, x):
        return self.h_space.gram

    def _proximal(self, x, tau):
        if self.h_domain == self.h_space:
            return 1/(1+tau)*x
        else:
            op = self.h_domain.gram+tau*self.h_space.gram
            inverse = operators.CholeskyInverse(op)
            return inverse(self.h_domain.gram(x))
        
    def _conj(self, xstar):
        return np.real(np.vdot(xstar, self.h_space.gram_inv(xstar))) / 2

    def _conj_linearize(self, xstar):
        gx = self.h_space.gram_inv(xstar)
        y = np.real(np.vdot(xstar, gx)) / 2
        return y, gx

    def _conj_subgradient(self, xstar):
        return self.h_space.gram_inv(xstar)

    def _conj_hessian(self, xstar):
        return self.h_space.gram_inv

    def _conj_proximal(self, xstar, tau):
        if self.h_domain == self.h_space:
            return 1/(1+tau)*xstar
        else:
            raise NotImplementedError


class IntegralFunctionalBase(Functional):
    r"""
    This class provides a general framework for Integral functionals of the type
    
    .. math::
        F\colon X \to \mathbb{R}
        v\mapsto \Int_\Omega f(v(x),x)\mathrm{d}x

    with \(f\colon \mathbb{R}^2\to \mathbb{R})\. 

    Subclasses defining explicit functionals of this type have to implement
     * `_f` evaluation the function \(f)\
     * `_f_deriv` giving the derivative \(\partial_1 f)\
     * `_f_prox` giving the prox of \(v>->f(v,x))\
    
    since 

    .. math::
        F'[g]h = \int_\Omega h(x)(\partial_1 f)(g(x),x)

    is a functional of the same type and

    .. math::
        \mathrm{prox}_F(v)(x) = \mathrm{prox}_{f(\cdot,x)}(v(x)).


    Parameters
    ----------
    domain : `regpy.vecsps.MeasureSpaceFcts`
        Domain on which it is defined. Needs some Measure therefore a MeasureSpaceFcts
    h_domain : `regpy.hilbert.HilbertSpace` [default: None]
        Hilbert space defined on `domain`. Proximal operator is computed  wrt to that. Default: `L2(domain)`
    """

    def __init__(self,domain,h_domain = None,**kwargs):
        assert isinstance(domain,vecsps.MeasureSpaceFcts)
        assert domain == h_domain.vecsp
        self.kwargs = kwargs
        super().__init__(domain,**kwargs)
        self.h_domain = hilbert.L2(domain) if h_domain is None else h_domain
        """ Hilbert space on `domain` wrt to which is the prox computed."""

    def _eval(self, v):
        return np.sum(self._f(v,**self.kwargs)*self.domain.measure)

    def _conj(self,vstar):
        return np.sum(self._f_conj(vstar/self.domain.measure,**self.kwargs)*self.domain.measure)

    def _subgradient(self, v):
        return self._f_deriv(v,**self.kwargs)*self.domain.measure

    def _hessian(self, v):
        return operators.PtwMultiplication(self.domain,self._f_second_deriv(v,**self.kwargs)*self.domain.measure)

    def _proximal(self, v, tau):
        return self._f_prox(v,tau,**self.kwargs)
    
    def _conj_proximal(self, vstar, tau):
        return self._f_conj_prox(vstar/self.domain.measure,tau,**self.kwargs)*self.domain.measure
    
    def _conj_subgradient(self, vstar):
        return self._f_conj_deriv(vstar/self.domain.measure,**self.kwargs)
    
    def _conj_hessian(self, vstar):
        return operators.PtwMultiplication(self.domain,self._f_conj_second_deriv(vstar/self.domain.measure,**self.kwargs))

    def _f(self,v,**kwargs):
        raise NotImplementedError
    
    def _f_deriv(self,v,**kwargs):
        raise NotImplementedError

    def _f_second_deriv(self,v,**kwargs):
        raise NotImplementedError

    def _f_prox(self,v,tau,**kwargs):
        """TODO: write default implementation by Newton's method"""
        raise NotImplementedError
    
    def _f_conj(self,vstar,**kwargs):
        raise NotImplementedError
    
    def _f_conj_deriv(self,vstar,**kwargs):
        raise NotImplementedError

    def _f_conj_second_deriv(self,vstar,**kwargs):
        raise NotImplementedError

    def _f_conj_prox(self,vstar,tau,**kwargs):
        raise NotImplementedError
    
class LppPower(IntegralFunctionalBase):
    r"""
    Implements the \(p)\-power of the \(L^p)\ norm on some domain in `MeasureSpaceFcts`
    as an integral functional.

    Parameters
    ----------
    domain : `regpy.vecsps.MeasureSpaceFcts`
        Domain on which it is defined. Needs some Measure therefore a MeasureSpaceFcts
    p: float >1 [option]
        exponent
    """

    def __init__(self, domain, p=2):
        assert np.isscalar(p) and p >1
        self.p = p
        self.q = p/(p-1)
        super().__init__(domain, hilbert.L2(domain),
                         convexity_param = 2 if p==2 else 0,
                         Lipschitz = 2 if p==2 else np.inf
                         )

    def _f(self,v,**kwargs):
        return np.abs(v)**self.p/self.p
    
    def _f_deriv(self, v,**kwargs):
        return np.abs(v)**(self.p-1)*np.sign(v)
    
    def _f_second_deriv(self, v,**kwargs):
        return (self.p-1)*np.abs(v)**(self.p-2)
    
    def _f_prox(self,v,tau,**kwargs):
        if self.p==2:
            return v/(1+tau)
        else:
            raise NotImplementedError('LppPower')
    
    def _f_conj(self, vstar,**kwargs):
        return np.abs(vstar)**self.q/self.q

    def _f_conj_deriv(self, vstar,**kwargs):
        return np.abs(vstar)**(self.q-1)*np.sign(vstar)
    
    def _f_conj_second_deriv(self, vstar,**kwargs):
        return (self.q-1)*np.abs(vstar)**(self.q-2)
    
    def _f_conj_prox(self,v_star,tau,**kwargs):
        if self.p==2:
            return v_star/(1+tau)
        else:
            raise NotImplementedError('prox of conjugate of LppPower')

class L1MeasureSpace(IntegralFunctionalBase):
    r""":math:`L ^1` Functional on `MeasureSpace`. Proximal implemented for default :math:`L^2` as `h_domain`.

    Parameters
    ----------
    domain : regpy.vecsps.MeasureSpaceFcts
        Domain on which to define the generic L1.
    """
    def __init__(self, domain):
        super().__init__(domain,hilbert.L2(domain))

    def _f(self, v,**kwargs):
        return np.abs(v)

    def _f_deriv(self, v,**kwargs):
        return np.sign(v)

    def _f_second_deriv(self, v,**kwargs):
        if np.any(v==0):
            raise NotTwiceDifferentiableError('L1')
        else:
            return np.zeros_like(v)

    def _f_prox(self, v,tau,**kwargs):
        return np.maximum(0, np.abs(v)-tau)*np.sign(v)

    def _f_conj(self, v_star,**kwargs):
        ind = (np.abs(v_star)>1)
        res = np.zeros_like(v_star)
        res[ind]=np.inf
        return res
    
    def _f_conj_deriv(self, v_star,**kwargs):
        if np.max(np.abs(v_star))>1:
            raise NotInEssentialDomainError()
        else:
            return np.zeros_like(v_star)

    def _f_conj_second_deriv(self, v_star,**kwargs):
        if np.max(np.abs(v_star))>=1:
            raise NotTwiceDifferentiableError('L1')
        else:
            return self.domain.zeros()

    def _f_conj_prox(self,vstar,tau,**kwargs):
        return vstar/np.maximum(np.abs(vstar),1)

    def is_subgradient(self, vstar, x, eps=1e-10):
        zeroind = (x==0)
        if np.any(zeroind) and np.max(np.abs(vstar[zeroind]))>1:
            return False
        else:
            vstar[zeroind]=0
            return super().is_subgradient(vstar, x, eps)

    def _conj_is_subgradient(self, v, xstar, eps=1e-10):
        return np.max(np.abs(xstar)<=1) and v[xstar==1]>=0 and v[xstar==-1]<=0 and v[np.abs(xstar)<1] ==0

class KullbackLeibler(IntegralFunctionalBase):
    r"""Kullback-Leiber divergence define by

    .. math::
        F(u,w) = KL(w,u) = \int (u(x) -w(x) - w(x)\ln \frac{u(x)}{w(x)}) dx


    Parameters
    ----------
    domain : regpy.vecsps.MeasureSpaceFcts
        Domain on which to define the Kullback-Leibler divergence
    w: domain [optional, no default value]
        First argument of Kullback-Leibler divergence.
        Formally optional, but required for proper functioning. 
    """

    def __init__(self, domain,**kwargs):
        w=kwargs['w']
        assert w in domain
        assert np.min(w)>=0
        super().__init__(domain,hilbert.L2(domain))
        self.kwargs = kwargs

    def _f(self, u,**kwargs):
        w= kwargs['w']
        res = u-w - w * np.log(u/w)
        ind_uneg = (u<0)
        res[ind_uneg] = np.inf
        ind_uzero = np.logical_and(u==0,np.logical_not(w==0))
        res[ind_uzero] = np.inf
        return res    
   
    def _f_deriv(self, u,**kwargs):
        w=kwargs['w']
        assert np.min(u)>=0
        assert np.all(np.logical_or(np.logical_not(u==0),w==0))
        res = np.ones_like(u)-w/u
        res[u==0] = 1
        return res

    def _f_second_deriv(self, u, **kwargs):
        w=kwargs['w']
        assert np.min(u)>0
        return w/u**2

    def _f_conj(self, u_star,**kwargs):
        w=kwargs['w']
        if np.any(u_star)>1:
            return np.inf 
        elif np.any(np.logical_and(u_star == 1,np.logical_not(w==0))):
            return np.inf 
        else:
            return -w*np.log(1-u_star)

    def _f_conj_deriv(self, u_star,**kwargs):
        w=kwargs['w']        
        assert np.max(u_star)<=1
        assert np.all(np.logical_or(np.logical_not(u_star==1),w==0))
        return w/(1-u_star)
    
    def _f_conj_second_deriv(self, u_star,**kwargs):
        w=kwargs['w']
        assert np.max(u_star)<=1
        assert np.all(np.logical_or(np.logical_not(u_star==1),w==0))
        return w/(1-u_star)**2

class RelativeEntropy(IntegralFunctionalBase):
    r"""Kullback-Leiber divergence define by

    .. math::
        F(u,w) = \int (u(x)\ln \frac{u(x)}{w(x)}) dx


    Parameters
    ----------
    domain : regpy.vecsps.MeasureSpaceFcts
        Domain on which to define the Kullback-Leibler divergence
    w: domain [optional, no default value]
        reference value.
        Formally optional, but required for proper functioning. 
    """

    def  __init__(self, domain,**kwargs):
        super().__init__(domain,hilbert.L2(domain))
        w = kwargs['w']
        assert w in domain
        assert np.min(w)>0
        self.kwargs = kwargs

    def _f(self, u,**kwargs):
        w=kwargs['w']        
        res =  u * np.log(u/w)
        ind_uneg = (u<0)
        res[ind_uneg] = np.inf
        res[u==0] = 0
        return res    
   
    def _f_deriv(self, u,**kwargs):
        w=kwargs['w']        
        assert np.min(u)>0
        res = np.ones_like(u)+np.log(u/w)
        return res

    def _f_second_deriv(self, u, **kwargs):
        assert np.min(u)>0
        return 1/u

    def _f_conj(self, u_star,**kwargs):
        w=kwargs['w']
        return w*(np.exp(u_star-1))

    def _f_conj_deriv(self, u_star,**kwargs):
        w=kwargs['w']
        return w*np.exp(u_star-1)
    
    def _f_conj_second_deriv(self, u_star,**kwargs):
        w=kwargs['w']
        return w*np.exp(u_star-1)

class Huber(IntegralFunctionalBase):
    r"""Huber functional 

    .. math::
        F(x) = 1/2 |x|^2                if  |x|\leq \sigma
        F(x) = \sigma |x|-\sigma^2/2    if  |x|>\sigma


    Parameters 
    ----------
    domain: regpy.vecsps.MeasureSpaceFcts
        domain on which Huber functional is defined
    sigma: float or domain [default: 1]
        parameter in the Huber functional. 
    as_primal: boolean [default:True]
        If False, then the functional is initiated as conjugate of QuadraticIntv. Then the dual metric is used, 
        and precautions against an infinite recursion of conjugations are taken.
    eps: float [default: 0.]
        Only used for conjugate functional. See description of `QuadraticIntv`
    """

    def  __init__(self, domain,as_primal=True,sigma = 1.,eps=0.):
        if as_primal:
            super().__init__(domain,hilbert.L2(domain),Lipschitz=1)
            self.conjugate = QuadraticIntv(domain,as_primal=False,sigma=sigma,eps=eps)
        else:
            super().__init__(domain,hilbert.L2(domain,weights=1./domain.measure**2), Lipschitz=1)        
            
        assert isinstance(sigma, (float,int)) or sigma in domain 
        assert np.min(sigma)>0
        if isinstance(sigma, (float,int)) :
            self.sigma = np.real(sigma * domain.ones())
        else:
            self.sigma = np.real(sigma) 

    def _f(self, u,**kwargs):
        return np.where(np.abs(u)<=self.sigma,0.5*np.abs(u)**2,self.sigma*np.abs(u)-0.5*self.sigma**2)

           
    def _f_deriv(self, u,**kwargs):
        return np.where(np.abs(u)<=self.sigma,u,self.sigma*u/np.abs(u))


    def _f_second_deriv(self, u, **kwargs):
        return (np.abs(u)<=self.sigma).astype(float)

    def _f_conj(self, ustar,**kwargs):
        return self.conjugate._f(ustar)    
   
    def _f_conj_deriv(self, ustar,**kwargs):
        return self.conjugate._f_deriv(ustar)

    def _f_conj_second_deriv(self, ustar,**kwargs):
        return self.conjugate._f_second_deriv(ustar)

    def _f_conj_prox(self,ustar,tau,**kwargs):
        return self.conjugate._f_prox(ustar,tau)


class QuadraticIntv(IntegralFunctionalBase):
    r"""Functional 

    .. math::
        F(x) = 1/2 |x|^2    if |x|\leq \sigma(x)
        F(x) = \infty    if |x|>\sigma(x)


    Parameters
    ----------
    regpy.vecsps.MeasureSpaceFcts
        domain on which Huber functional is defined
    sigma: float or domain [default: 1]
        interval width. 
    as_primal: boolean [default:True]
        If False, then the functional is initiated as conjugate of Huber. Then the dual metric is used, 
        and precautions against an infinite recursion are taken.
    eps: float [default: 0.]
        sigma is replace by sigma*(1+eps) on all operations except the proximal mapping to avoid np.inf return values 
        or NotInEssentialDomain exceptions in the presence of rounding errors
    """

    def  __init__(self, domain,as_primal=True,sigma=1.,eps=0.):
        if as_primal:
            super().__init__(domain,hilbert.L2(domain),convexity_param=1)
            self.conjugate = Huber(domain,as_primal=False,sigma=sigma)
        else:
            super().__init__(domain,hilbert.L2(domain,weights=1./domain.measure**2), convexity_param=1)
        assert isinstance(sigma, (float,int)) or sigma in domain 
        assert np.min(sigma)>0
        if isinstance(sigma, (float,int)):
            self.sigma = sigma * domain.ones()
        else:
            self.sigma = sigma 
        self.sigmaeps = self.sigma*(1+eps) if eps>0 else self.sigma

    def _f(self, u,**kwargs):
        res =  0.5*np.abs(u)**2
        res[np.abs(u)>self.sigmaeps] = np.inf
        return res    
   
    def _f_deriv(self, u,**kwargs):
        if np.max(np.abs(u)/self.sigmaeps)>1.:
            raise NotInEssentialDomainError('QuadraticIntv')
        return u.copy()

    def _f_prox(self,u,tau,**kwargs):
        res = u/(1+tau)
        return res/np.maximum(np.abs(res)/self.sigma,1)

    def _f_second_deriv(self, u,**kwargs):
        if np.max(np.abs(u)/self.sigmaeps)>=1.:
            raise NotTwiceDifferentiableError('QuadraticIntv')
        else:
            return np.ones_like(u)

    def _f_conj(self, ustar,**kwargs):
        return self.conjugate._f(ustar)    
   
    def _f_conj_deriv(self, ustar,**kwargs):
        return self.conjugate._f_deriv(ustar)

    def _f_conj_second_deriv(self, ustar,**kwargs):
        return self.conjugate._f_second_deriv(ustar)

    def _f_conj_prox(self,ustar,tau,**kwargs):
        return self.conjugate._f_prox(ustar,tau)

    def is_subgradient(self, vstar, x, eps=1e-10):
        grad = self.subgradient(x)
        if(not np.all(np.abs(x)<=self.sigma)):
            return False
        if(not np.all(vstar[self.sigma==x]>=self.sigma)):
            return False
        if(not np.all(vstar[-self.sigma==x]<=-self.sigma)):
            return False
        if(np.linalg.norm(grad[np.abs(x)<self.sigma]-vstar[np.abs(x)<self.sigma]) <= eps*np.linalg.norm(grad[np.abs(x)<self.sigma])):
            return True
        return False


class QuadraticNonneg(IntegralFunctionalBase):

    r"""Functional 

    .. math::
        F(x) = 1/2 |x|^2    if x\geq 0
        F(x) = \infty       if  x<0

    Parameters
    ----------
    domain : regpy.vecsps.MeasureSpaceFcts
        domain on which functional is defined 

    """


    def  __init__(self, domain):
        super().__init__(domain,hilbert.L2(domain),convexity_param = 1.)

    def _f(self, u,**kwargs):
        res =  u*u/2
        res[u<0] = np.inf
        return res    

    def _f_deriv(self, u,**kwargs):
        if np.min(u)<0:
            raise NotInEssentialDomainError('QuadratiNonneg')
        return u.copy()

    def _f_prox(self,u,tau,**kwargs):
        return np.maximum(u/(1+tau),0)

    def _f_second_deriv(self, u,**kwargs):
        if np.min(u)<0:
            raise NotTwiceDifferentiableError('QuadraticNonneg')
        else:
            return np.ones_like(u)

    def _f_conj(self, ustar,**kwargs):
        res = ustar*ustar/2
        res[ustar<0] = 0
        return res

    def _f_conj_deriv(self, ustar,**kwargs):
        res = ustar.copy()
        res[ustar<0] = 0
        return res

    def _f_conj_second_deriv(self, ustar,**kwargs):
        return 1.* (ustar>=0)

    def _f_conj_prox(self,ustar,tau,**kwargs):
        res=ustar.copy()
        res[ustar>0]*=(1/(1+tau))
        return res
    
    def is_subgradient(self, vstar, x, eps=1e-10):
        return np.max(vstar[x<0])<=0 and np.linalg.norm(x[x>=0]-vstar[x>=0]) <= eps*np.linalg.norm(x[x>=0])


class QuadraticBilateralConstraints(LinearCombination):
    r""" Returns `Functional` defined by 

    .. math::
        F(x) = \frac{\alpha}{2}\|x-x0\|^2  if lb\leq x\leq ub
        F(x) = np.inf else


    Parameters
    ----------
    domain: regpy.vecsps.MeasureSpaceFcts
        domain on which functional is defined
    lb: domain
        lower bound
    ub: domain
        upper bound
    x0: domain
        reference value
    alpha: float [default: 1]
        regularization parameter
    eps: real [default: 0]
        Tolerance parameter for violations of the hard constraints (which may occur due to rounding errors).
        If constraints are violated by less then eps times the interval width, the polynomial is evaluated, rather than returning np.inf.
    """

    def __init__(self,domain, lb=None, ub=None, x0=None,alpha=1.,eps=0.):
        assert isinstance(domain,vecsps.MeasureSpaceFcts)
        if isinstance(lb,(float,int)):
            lb = lb*domain.ones()
        elif lb is None:
            lb = domain.zeros()
        assert lb in domain
        if isinstance(ub,(float,int)):
            ub = ub*domain.ones()
        elif ub is None:
            ub = domain.zeros()
        assert ub in domain
        assert np.all(lb<ub)
        if x0 is None:
            x0 =0.5*(lb+ub)
        elif isinstance(x0,(float,int)):
            x0 = x0*domain.ones()
        assert x0 in domain 
        assert isinstance(alpha,(float,int))

        self.lb = lb; self.ub = ub; self.x0 =x0; self.alpha = alpha
        F = QuadraticIntv(domain,sigma=(ub-lb)/2.,eps=eps)
        center = (ub+lb)/2
        lin = LinearFunctional(center-x0,
                            domain=domain,
                            gradient_in_dual_space=False
                            )
        offset = 0.5*(np.sum((x0**2-center**2)*domain.measure))
        # return  alpha*HorizontalShiftDilation(F,shift=center) + alpha*lin + alpha*offset
        super().__init__((alpha,HorizontalShiftDilation(F,shift=center)+offset),
                          (alpha,lin)
                          )

def QuadraticLowerBound(domain, lb, x0,a=1.):
    r""" Returns `Functional` defined by 

    \[F(x) = \frac{a}{2}\|x-x0\|^2  if lb\leq x
     F(x) = np.inf else


    Parameters
    ----------
    domain: `vecsps.MeasureSpaceFcts`
        domain on which the functional is defined
    lb: domain or float [default: None]
        lower bound (zero in the default case)
    x0: domain or float [default: None]
        lower bound (zero in the default case)
    """
    assert isinstance(domain,vecsps.MeasureSpaceFcts)
    if isinstance(lb,(float,int)):
        lb = lb*domain.ones()
    elif lb is None:
        lb = domain.zeros()
    assert lb in domain
    if isinstance(x0,(float,int)):
        x0 = x0*domain.ones()
    elif x0 is None:
        x0 = domain.zeros()
    assert x0 in domain 
    assert isinstance(a,(float,int))

    F = QuadraticNonneg(domain)
    lin = LinearFunctional(lb-x0,domain=domain,gradient_in_dual_space=False)
    offset = 0.5*(np.sum((x0**2-lb**2)*domain.measure))
    return a*HorizontalShiftDilation(F,shift=lb)+ a*lin + a*offset

from scipy.linalg import ishermitian
from numpy.linalg import eigvalsh,eigh

class QuadraticPositiveSemidef(Functional):

    r"""Functional 

    .. math::
        F(x) = 1/2 ||x||_{HS}^2    \text{if } x\geq 0 \text{ and (optional) } tr(x)=c
        F(x) = \infty       \text{else}

    Here x is a quadratic matrix and HS is the Hilbert-Schmidt norm. Conjugate functional
    and prox are only correct for hermitian inputs.

    Parameters
    ---------
    domain: regpy.vecsps.UniformGridFcts
        two dimensional domain on which functional is defined, volume_elements have to be one
    trace_val: float or None, optional
        desired value of trace or None for no trace constraint. Defaults to None.
    tol: float, optional
        tolerance for comparisons determining positive semidefiniteness and correctness of trace 

    """


    def  __init__(self, domain,trace_val=1.0,tol=1e-15):
        assert isinstance(domain,vecsps.UniformGridFcts)
        assert domain.ndim==2
        assert domain.shape[0]==domain.shape[1]
        assert domain.volume_elem==1
        assert tol>=0
        assert trace_val is None or trace_val>0
        self.tol=1e-15
        if(trace_val is not None):
            self.has_trace_constraint=True
            self.trace_val=trace_val
        else:
            self.has_trace_constraint=False
        super().__init__(domain,hilbert.L2(domain),Lipschitz=1,convexity_param=1)

    def is_in_essential_domain(self,rho):
        if(not ishermitian(rho,atol=self.tol)):
            return False
        if(self.has_trace_constraint):
            if(np.abs(np.trace(rho)-self.trace_val)>self.tol):
                return False
        evs=eigvalsh(rho)
        return evs[0]>-self.tol
    
    @staticmethod
    def closest_point_simplex(p,a):
        r'''
        Algorithm from Held, Wolfe and Crowder (1974) to project onto simplex :math:`\{q:q_{i}\qeq 0,\sum q_{i}=a\}`.
        It uses that p is already sorted in increasing order.

        Parameters
        ---------
        p: numpy.ndarray
            Input point sorted in increasing order
        a: float
            positive value that is the sum of the elements in the result
        '''
        p_flipped=np.flip(p)
        comp_vals=(np.cumsum(p_flipped)-a)/np.arange(1,p.shape[0]+1)
        k=np.where(comp_vals<p_flipped)[0][-1]
        t=comp_vals[k]
        return np.maximum(p-t,0)
        

    def _eval(self, x):
        if(self.is_in_essential_domain(x)):
            return np.sum(np.abs(x)**2)/2
        else:
            return np.inf

    def _proximal(self, x, tau):
        evs,U=eigh(x)
        evs/=(1+tau)
        if(self.has_trace_constraint):
            proj_evs=QuadraticPositiveSemidef.closest_point_simplex(evs,self.trace_val)
        else:
            proj_evs=np.maximum(0,evs)
        return U@np.diag(proj_evs)@np.conj(U).T
        
    def _subgradient(self, x):
        if(self.is_in_essential_domain(x)):
            return np.copy(x)
        else:
            return NotInEssentialDomainError
        
    def _hessian(self,x):
        if(self.is_in_essential_domain(x)):
            return self.domain.identity
        else:
            return NotInEssentialDomainError

        
    def _conj(self,xstar):
        evs=eigvalsh(xstar)
        if(self.has_trace_constraint):
            cps=QuadraticPositiveSemidef.closest_point_simplex(evs,self.trace_val)
            return (np.sum(evs**2)+np.sum((cps-evs)**2))/2
        else:
            return (np.sum(evs**2)+np.sum(evs**2,where=evs<0))/2

class L1Generic(Functional):
    r"""Generic :math:`L ^1` Functional. Proximal implemented for default :math:`L^2` as `h_domain`.

    Parameters
    ----------
    domain : regpy.vecsps.VectorSpace
        Domain on which to define the generic L1.
    """
    def __init__(self, domain):
        super().__init__(domain)

    def _eval(self, x):
        return np.sum(np.abs(x))

    def _subgradient(self, x):
        return np.sign(x)

    def _hessian(self, x):
        # Even approximate Hessians don't work here.
        raise NotImplementedError

    def _proximal(self, x, tau):
        return np.maximum(0, np.abs(x)-tau)*np.sign(x)


class TVGeneric(Functional):
    r"""Generic TV Functional. Proximal implemented for default L2 h_space

    NotImplemented yet!
    """
    def __init__(self, domain, h_domain=hilbert.L2):
        super().__init__(domain,h_domain=h_domain)

    def _subgradient(self, x):
        return NotImplementedError

    def _hessian(self, x):
        return NotImplementedError
    
    def _proximal(self, x, tau):
        return NotImplementedError

class TVUniformGridFcts(Functional):
    r"""Total Variation Norm: For C^1 functions the l1-norm of the gradient on a Uniform Grid

    Parameters
    ----------
    domain : regpy.vecsps.UniformGridFcts
        Underlying domain. 
    h_domain : regpy.hilbert.HilbertSapce (defaul: L2)
        Underlying Hilbert space for proximal. 
    """
    def __init__(self, domain, h_domain=None):
        self.dim = np.size(domain.shape)
        """Dimension of the Uniform Grid functions.
        """
        assert isinstance(domain, vecsps.UniformGridFcts)
        super().__init__(domain,h_domain=h_domain)

    def _eval(self, x):
        if self.dim==1:
            return np.sum(np.abs(self._gradientuniformgrid(x)))
        else:
            return np.sum(np.linalg.norm(self._gradientuniformgrid(x), axis=0))

    def _subgradient(self, x):
        if self.dim==1:
            return np.sign(self._gradientuniformgrid(x))
        else:
            grad = self._gradientuniformgrid(x)
            grad_norm = np.linalg.norm(grad, axis=0)
            toret = np.zeros(x.shape)
            toret = np.where(grad_norm != 0, np.sum(grad, axis=0) / grad_norm, toret)
            return toret

    def _hessian(self, x):
        raise NotImplementedError

    def _proximal(self, x, tau, stepsize=0.1, maxiter=10):
        shape = [self.dim]+list(x.shape)
        p = np.zeros(shape)
        for i in range(maxiter):
            update = stepsize*self._gradientuniformgrid( self.h_domain.gram_inv( self._divergenceuniformgrid(p))-x/tau)
            p = (p+update) / (1+np.abs(update))
        return x-tau*self._divergenceuniformgrid(p)

    def _gradientuniformgrid(self, u):
        r"""Computes the gradient of field given by 'u'. 'u' is defined on a 
        equidistant grid. Returns a list of vectors that are the derivatives in each 
        dimension."""
        # Need to reshape spacing otherwise getting braodcasting error
        shape = [self.domain.ndim]+[1 for _ in self.domain.shape]
        return 1/self.domain.spacing.reshape(shape)*np.array(np.gradient(u))

    def _divergenceuniformgrid(self, u):
        r"""Computes the divergence of a vector field 'u'. 'u' is assumed to be
        a list of matrices u=(u_x, u_y, u_z, ...) holding the values for u on a
        regular grid"""
        return np.ufunc.reduce(np.add, [np.gradient(u[i], axis=i)/h for i,h in enumerate(self.domain.spacing)])

def as_functional(func, vecsp):
    r"""Convert `func` to Functional instance on vecsp.

    - If func is a `HilbertSpace` then it generated the `HilbertNormGeneric`.
    - If func is an Operator, it's wrapped in a `GramHilbertSpace` and then `HilbertNormGeneric` functional.
    - If func is callable, e.g. an `hilbert.AbstractSpace` or `AbstractFunctional`, it is called on `vecsp` to construct the concrete functional or Hilbert space. In the later case the functional will be the `HilbertNormGeneric`

    Parameters
    ----------
    func : Functional or HilbertSapce or regpy.operators.Operator or callable
        Functional or object from which to construct the Functional.
    vecsp : VectorSpace
        Underlying vector space for the functional. 

    Returns
    -------
    Functional
        Constructed Functional on the underlying vectorspace. 
    """
    from regpy.operators import Operator  # imported here to avoid circular dependency
    if not isinstance(func,Functional):
        if isinstance(func, operators.Operator):
            func = HilbertNormGeneric(hilbert.GramHilbertSpace(func))
        elif callable(func):
            func = func(vecsp)
        if isinstance(func, hilbert.HilbertSpace):
            func = HilbertNormGeneric(func)
    assert isinstance(func,Functional)
    assert func.domain == vecsp or (isinstance(func,Composed) and func.func.domain == vecsp), "Given Vector space and the one of the functional do not match."
    return func

def HilbertNormOnAbstractSpace(vecsp, h_space=hilbert.L2):
    return HilbertNorm(h_space(vecsp))


def _register_functionals():
    r"""Auxiliary method to register abstract functionals for various vector spaces. Using the decorator
    method described in `AbstractFunctional` does not work due to circular depenencies when
    loading modules.

    This is called from the `regpy` top-level module once, and can be ignored otherwise.
    """
    HilbertNorm.register(hilbert.HilbertSpace, HilbertNormGeneric)
    HilbertNorm.register(vecsps.VectorSpace,HilbertNormOnAbstractSpace)

    L1.register(vecsps.VectorSpace, L1Generic)
    L1.register(vecsps.MeasureSpaceFcts, L1MeasureSpace)

    TV.register(vecsps.VectorSpace, TVGeneric)
    TV.register(vecsps.UniformGridFcts, TVUniformGridFcts)
