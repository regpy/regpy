from collections import defaultdict

from copy import copy

import numpy as np

from regpy import operators, util, vecsps
from regpy import hilbert


class Functional:
    r"""
    Base class for implementation of a functional. Subsclasses should at least implement the 
        `_eval` :  evaluating the funcitonal
    and 
        `_gradient` or `_deriv` : returning the gradient or derivative at `x`.
    
    The evalution of a specific functional on some element of the `domain` can be done by
    simply caling the functional on that element. 
        
    Funcationals can be added by taking `LinearCombination` of them. The `domain` has to be the
    same for each functional. 

    They can also be multiplied by scalars or `np.ndarrays`of `domain.shape`or multiplied by 
    `regpy.operators.Operator`. This leads to a functional that is composed with the operator
    \(F\circ O\) where \(F\) is the functional and \(O)\ some operator. Multiplying by a scalar
    results in a composition with the `PtwMultiplication` operator.

    Parameters
    ----------
    domain : regpy.vecsps.VectorSpace
        The uncerlying vector space for the function space on which it is defined.
    h_domain : regpy.hilbert.HilbertSpace (default: `L2(domain)`)
        The underlying Hilbert wrt which the proximal and the conjugate are computed.
    """
    def __init__(self, domain, h_domain=None, linear = False):
        assert isinstance(domain, vecsps.VectorSpace)
        self.domain = domain
        """The underlying vector space."""
        self.h_domain = hilbert.as_hilbert_space(h_domain,domain) or hilbert.L2(domain)
        """The underlying Hilbert space."""
        self.linear = linear
        """boolean indicating if the functional is linear"""

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
        Linearizes the functional at `x` given by the value at that point and the gradient which is considered as
        \[
            F(x+\epsilon h) = F(x) + \epsilon  \nabla F[x]^T h + \mathcal{o}(\epsilon)
        \]
        Requires the implementation of either `_gradient` or `_linearize`.

        Parameter
        ----------
        x : in self.domain
            Element at which will be linearized

        Return
        ----------
        y 
            Value of \(F(x)\).
        grad : in self.domain
            Gradient of \(F\) at \(x\).        
        """
        assert x in self.domain
        try:
            y, grad = self._linearize(x)
        except NotImplementedError:
            y = self._eval(x)
            grad = self._gradient(x)
        assert isinstance(y, float)
        assert grad in self.domain
        return y, grad

    def gradient(self, x):
        r"""
        Gradient \(\nabla F[x])\ of the functional at `x` characterized by
        \[
            F(x+\epsilon h) = F(x) + \epsilon \nabla F[x])^T h  + \mathcal{O}(\epsilon^2)
        \]
        Requires the implementation of either `_gradient` or `_linearize`.

        Parameter
        ----------
        x : in self.domain
            Element at which will be linearized

        Return
        ----------
        grad : in self.domain
            Gradient of \(F\) at \(x\).        
        """
        assert x in self.domain
        try:
            grad = self._gradient(x)
        except NotImplementedError:
            _, grad = self._linearize(x)
        assert grad in self.domain
        return grad

    def hessian(self, x,recursion_safeguard=False):
        r"""The hessian of the functional at `x` as an `regpy.operators.Operator` maping form the 
        functionals `domain` to it self. Requires the implementation of `_hessian` or by default
        computes the `regpy.operators.ApproximateHessian`. It is defined by 
        \[
         F(x+h) = F(x) + (\nabla F)(x)^T h + \frac{1}{2} h^T Hess F(x) h + \mathcal{o}(\|h\|^2)
        \]

        Parameter:
        ----------
        `x` : `self.domain`
            Point in `domain` at which to compute the hessian. 

        Returns:
        ----------
        `h` : `regpy.operators.Operator` (Default: `regpy.operators.ApproximateHessian`)
            Hessian operator at the point `x`. 
        """
        assert x in self.domain
        try:
            h = self._hessian(x)
        except NotImplementedError:
            if recursion_safeguard:
                raise NameError("Neither hessian nor conj_hessian are implemented.")
            else:
                h = self.conj_hessian(self.gradient(x),recursion_safeguard=True).inverse
        assert isinstance(h, operators.Operator)
        assert h.linear
        assert h.domain == h.codomain == self.domain
        return h

    def conj(self, xstar):
        r"""The conjugate functional 
        \[
            F^*(x^*) = \sup_{x\in {\mathcal{X}}}((x^*)^T x - F(x) )
        \]
        
        Parameter:
        -------------
        `xstar` : `self.domain`
            Point in `domain` at which to compute the conjugate functional. 
        
        Returns:
        -------------
        `y`: float
            Value of the conjugate functional at `xstar`
        """
        assert xstar in self.domain
        y = self._conj(xstar)
        assert isinstance(y, float)
        return y        

    def conj_gradient(self, xstar):
        r"""
        Gradient \(\nabla F^*[x^*])\ of the conjugate functional F^* at `x^*`. 
        Requires the implementation of either `_conj_gradient`.

        Parameter
        ----------
        xstar : in self.domain
            Element at which will be linearized

        Return
        ----------
        grad : in self.domain
            Gradient of \(F^*\) at \(x^*\).        
        """
        assert xstar in self.domain
        try:
            grad = self._conj_gradient(xstar)
        except NotImplementedError:
            _, grad = self._conj_linearize(xstar)
        assert grad in self.domain
        return grad

    def conj_hessian(self,xstar, recursion_safeguard=False):
        r"""The hessian of the functional at `x` as an `regpy.operators.Operator` maping form the 
        functionals `domain` to it self. Requires the implementation of `_conj_hessian` or 
        _hessian and _conj_gradient
        
        Parameter:
        ----------
        `x` : `self.domain`
            Point in `domain` at which to compute the hessian. 

        Returns:
        ----------
        `h` : `regpy.operators.Operator` (Default: `regpy.operators.ApproximateHessian`)
            Hessian operator at the point `x`. 
        """
        assert xstar in self.domain
        try:
            h = self._conj_hessian(xstar)
        except NotImplementedError:
            if recursion_safeguard:
                raise NameError("Neither hessian nor conj_hessian are implemented.")
            else:
                h = self.hessian(self.conj_gradient(xstar),recursion_safeguard=True).inverse
        assert isinstance(h, operators.Operator)
        assert h.linear
        assert h.domain == h.codomain == self.domain
        return h

    def conj_linearize(self, xstar):
        r"""
        Linearizes the conjugate functional \(F^*\) at `xstar` given by the value at that point and the gradient 
        Requires the implementation of either `_conj_gradient` or `_conj_linearize`.

        Parameter
        ----------
        xstar : in self.domain
            Element at which will be linearized

        Return
        ----------
        y 
            Value of \(F^*(x^*)\).
        grad : in self.domain
            Gradient of \(F^*\) at \(x^*\).        
        """
        assert xstar in self.domain
        try:
            y, grad = self._conj_linearize(xstar)
        except NotImplementedError:
            y = self._conj(xstar)
            grad = self._conj_gradient(xstar)
        assert isinstance(y, float)
        assert grad in self.domain
        return y, grad

    def proximal(self, x, tau, recursion_safeguard = False, proximal_pars = None):
        r"""Proximal operator 
        \[
            \mathrm{prox}_{\tau F}(x)=\arg \min _{v\in {\mathcal {X}}}(F(v)+{\frac{1}{2\tau}}\Vert v-x\Vert_{\mathcal {X}}^{2}).
        \]
        Requires an implementation of `_proximal`.

        Parameters
        ----------
        x : `self.domain`
            Point at which to compute proximal.
        tau : `np.number`
            Regularization parameter for the proximal. 
        proximal_pars : any, optional
            parameters handed to the implementation of `_proximal`, by default None

        Returns
        -------
        proximal : `self.domain`
            the computed proximal at \(x\) with parameter \(\tau\).
        """
        assert x in self.domain
        if proximal_pars == None:
            proximal_pars = {}
        self.proximal_pars = proximal_pars
        try: 
            proximal = self._proximal(x, tau, **proximal_pars)
        except NotImplementedError:
            # evaluation by Moreau's identity
            if recursion_safeguard: 
                raise NameError("Neither proximal nor proximal_conj are implemented.")
            else:
                gram = self.h_domain.gram
                proximal = x - tau *gram.inverse(self.proximal_conj(gram(x)/tau,1/tau,recursion_safeguard=True))
        assert proximal in self.domain
        return proximal

    def proximal_conj(self, xstar, tau, recursion_safeguard = False, proximal_pars = None):
        r"""Proximal operator of conjugate functional 
        \[
            \mathrm{prox} _{\tau F^*}(x^*)=\arg \min _{v\in {\mathcal {X}}}(F^*(v^*)+{\frac{1}{2\tau}}\Vert v^*-x^*\Vert_{\mathcal {X}}^{2}).
        \]
        Requires an implementation of `_proximal` (in this case proximal_conj is evaluated by Moreau's identity) or an alternative implementation of ` _proximal_conj`.

        Parameters
        ----------
        xstar : `self.domain`
            Point at which to compute proximal_star.
        tau : `np.number`
            Regularization parameter for the proximal. 
        proximal_pars : any, optional
            parameters handed to the implementation of `_proximal`, by default None

        Returns
        -------
        proximal : `self.domain`
            the computed proximal of the conjugate functional at \(x^*\) with parameter \(\tau\).
        """
        assert xstar in self.domain
        if proximal_pars == None:
            proximal_pars = {}
        self.proximal_pars = proximal_pars
        #proximal = self._proximal_conj(xstar, tau, **proximal_pars)
        try:
            proximal = self._proximal_conj(xstar, tau, **proximal_pars)
        except NotImplementedError:
            if recursion_safeguard: 
                raise ValueError("neither proximal nor proximal_conj are implemented")
            else:
                gram = self.h_domain.gram
             #proximal = xstar - tau * gram(self.proximal(gram.inverse(xstar),1/tau,recursion_safeguard=True))
        assert proximal in self.domain
        return proximal 


    def _eval(self, x):
        raise NotImplementedError

    def _linearize(self, x):
        raise NotImplementedError

    def _gradient(self, x):
        raise NotImplementedError

    def _hessian(self, x):
        raise NotImplementedError
    
    def _conj(self, xstar):
        raise NotImplementedError

    def _conj_linearize(self, xstar):
        raise NotImplementedError
    
    def _conj_gradient(self, xstar):
        raise NotImplementedError

    def _conj_hessian(self, xstar):
        raise NotImplementedError

    def _proximal(self, x, tau):
        raise NotImplementedError

    def _proximal_conj(self, xstar, tau):
        raise NotImplementedError
#         gram = self.h_domain.gram
#         return xstar-tau*gram(self._prox(gram.inverse(xstar)/tau,1/tau))

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
            return VerticalShift(self, other)
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


class LinearFunctional(Functional):
    r"""Linear functionals
    Linear functional given by
        F(x) = np.dot(a, x)
    
    Parameters: 
    gradient: domain
        The gradient of the linear functional. \(a=gradient\) if gradient_in_dual_space == True

    domain: regpy.vctspc.VectorSpace, optional
        The VectorSpace on which the functional is defined

    h_domain: regpy.hilbert.HilbertSpace (default: `L2(domain)`)
        Hilbert space for proximity operator

    gradient_in_dual_space: bool (default: False)
        If false, the argument gradient is considered as an element of the primal space, 
        and \(a = h_domain.gram(gradient).\).
    
    """
    def __init__(self,gradient,domain=None,h_domain = None,gradient_in_dual_space = False):
        if domain is None:
            domain = vecsps.VectorSpace(shape=gradient.shape,dtype=float)
        super().__init__(domain=domain,h_domain=h_domain,linear=True)
        assert gradient in self.domain
        if gradient_in_dual_space:
            self._gradient_vector = gradient
        else:
            self._gradient_vector = self.h_domain.gram(gradient)

    def _eval(self,x):
        return np.dot(self._gradient_vector,x)

    @property
    def gradient_vector(self):
        return self._gradient_vector.copy()

    def _gradient(self,x):
        return self._gradient_vector.copy()
    
    def _hessian(self, x):
        return operators.Zero(self.domain)

    def _conj(self,x_star):
        0 if x_star == self._gradient_vector else np.inf

    def _proximal(self, x, tau):
        return x-tau*self._gradient_vector

    def _proximal_conj(self, xstar, tau):
        return self._gradient_vector.copy()


class LinearCombination(Functional):
    """Linear combination of functionals. 

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
        self.linear_table = []
        for func, coeff in coeff_for_func.items():
            self.coeffs.append(coeff)
            self.funcs.append(func)
            self.linear_table.append(func.linear)        

        domains = [func.domain for func in self.funcs if func.domain]
        if domains:
            domain = domains[0]
            assert all(d == domain for d in domains)
        else:
            domain = None

        super().__init__(domain, linear = all(self.linear_table))

        if self.linear_table.count(False)<=1 and self.linear_table.count(True)>=1:
            self.grad_sum = self.domain.zeros()
            for coeff,func,linear in zip(self.coeffs,self.funcs,self.linear_table):
                if linear:
                    self.grad_sum += coeff * func.gradient_vector

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

    def _gradient(self, x):
        grad = self.domain.zeros()
        for coeff, func in zip(self.coeffs, self.funcs):
            grad += coeff * func.gradient(x)
        return grad

    def _hessian(self, x):
        return operators.LinearCombination(
            *((coeff, func.hessian(x)) for coeff, func in zip(self.coeffs, self.funcs))
        )

    def _conj(self, x):
        if len(self.funcs) == 1:
            return self.coeffs[0]*self.funcs[0].conj(x/self.coeffs[0])
        elif self.linear_table.count(False)==0:
            return 0 if x == self.grad_sum else np.inf
        elif self.linear_table.count(False)==1:
            j = self.linear_table.index(False)
            return self.coeffs[j]*self.funcs[j].conj((x-self.grad_sum)/self.coeffs[j])
        else:
            return NotImplementedError

    def _proximal(self, x, tau, proximal_params):
        if len(self.funcs) == 1:
            return self.funcs[0].proximal(x,self.coeffs[0]*tau)
        elif self.linear_table.count(False)==0:
            return x-tau*self.grad_sum
        elif self.linear_table.count(False)==1:
            return self.funcs[0].proximal(x-tau*self.grad_sum,self.coeffs[0]*tau)
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
        super().__init__(func.domain)
        self.func = func
        """Functional to be offset.
        """
        self.offset = offset
        """Offset added to the evaluation of the functional.
        """

    def _eval(self, x):
        return self.func(x) + self.offset

    def _linearize(self, x):
        return self.func.linearize(x)

    def _gradient(self, x):
        return self.func.gradient(x)

    def _hessian(self, x):
        return self.func.hessian(x)

    def _conj(self,x):
        return self.func.conj(x) - self.offset
    
    def _proximal(self, x, tau):
        return self.func.proximal(x, tau)

class HorizontalShiftDilation(Functional):
    r"""Implements a horizontal shift and/or a horizontal translation of the graph of a functional \(F\), i.e. replaces 
    \(F(x)\) by \(F(dilation(x-shift)))
    
    Parameters
    --------
    dilation: float
        dilation factor
    shift: self.domain
        shift vector
    """
    def __init__(self, F, dilation =1., shift = None):
        super().__init__(F.domain, F.h_domain, F.linear)
        assert shift is None or shift in self.domain
        assert np.isscalar(dilation) and util.is_real_dtype(dilation)
        self.F = F
        self.dilation = dilation
        self.shift = shift

    def _eval(self, x):
        if self.shift is None:
            return self.F(self.dilation * x)
        else:
            return self.F(self.dilation * (x-self.shift))
         
    def _conj(self,x_star):
        if self.shift is None:
            return self.F.conj(x_star/self.dilation)             
        else:
            return self.F.conj(x_star/self.dilation) + np.dot(x_star,self.shift)
        
    def _proximal(self, x, tau):
        if self.shift is None:
            return (1./self.dilation) * self.F.proximal(self.dilation*x,tau*self.dilation**2)
        else:
            return self.shift + (1./self.dilation) * self.F.proximal(self.dilation*(x-self.shift),tau*self.dilation**2)
 

class Composed(Functional):
    r"""Composition of an operator with a functional \(F\circ O\). This should not be called
    directly but rather used by multiplying the `Functional` object with an `Operator`.

    Parameters
    ----------
    func : `regpy.functionals.Functional`
        Functional to be composed with. 
    op : `regpy.operators.Operator`
        Operator to be composed with. 
    """
    def __init__(self, func, op):
        assert isinstance(func, Functional)
        assert isinstance(op, operators.Operator)
        assert func.domain == op.codomain
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

    def _eval(self, x):
        return self.func(self.op(x))

    def _linearize(self, x):
        y, deriv = self.op.linearize(x)
        z, grad = self.func.linearize(y)
        return z, deriv.adjoint(grad)

    def _gradient(self, x):
        y, deriv = self.op.linearize(x)
        return deriv.adjoint(self.func.gradient(y))

    def _hessian(self, x):
        if self.op.linear:
            return self.op.adjoint * self.func.hessian(x) * self.op
        else:
            # TODO this can be done slightly more efficiently
            return super()._hessian(x)

    def _conj(self,x):
        if self.op.linear:
            return self.func.conj(self.op.adjoint.inverse(x))

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
    """Class representing abstract functionals without reference to a concrete implementation.

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
    """An abstract functional that can be called on a vector space to get the corresponding
    concrete implementation.

    AbstractFunctionals provides two kinds of functionality:

    - A decorator method `register(vecsp_type)` that can be used to declare some class or function
      as the concrete implementation of this abstract functional for vector spaces of type `vecsp_type`
      or subclasses thereof, e.g.:

              @TV.register(vecsps.UniformGridFcts)
              class TVUniformGridFcts(HilbertSpace):
                  ...

    - AbstractFunctionals are callable. Calling them on a vector space and arbitrary optional
      keyword arguments finds the corresponding concrete `regpy.functionals.Functional` among all
      registered implementations. If there are implementations for multiple base classes of the
      vector space type, the most specific one will be chosen. The chosen implementation will
      then be called with the vector space and the keyword arguments, and the result will be
      returned.

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
        """Either registers a new implementation on a specific `regpy.vecsps.VectorSpace` 
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
    r"""Abstract analogue to `Composed`. Composition of an operator with a functional \(F\circ O\). This should not be called
    directly but rather used by multiplying the `AbstractFunctional` object with an `Operator`.

    Parameters
    ----------
    func : `regpy.functionals.AbstractFunctional`
        Functional to be composed with. 
    op : `regpy.operators.Operator`
        Operator to be composed with. 
    """
    def __init__(self, func, op):
        assert isinstance(func, AbstractFunctional), "func not a AbstractFunctional"
        assert isinstance(op, operators.Operator), "op not a Operator"
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
    

class FunctionalProductSpace(Functional):
    """Helper to define Functionals with respective prox-operators on product spaces (vecsps.DirectSum objects).
    The functionals are given as a list of the functionals on the summands of the product space.

    Parameters
    ----------
    funcs : [regpy.functionals.Functional, ...]
        List of functionals each defined on one summand of the direct sum of vector spaces.
    domain : regpy.vecsps.DirectSum
        Domain on which the combined functional is defined. 
    """
    def __init__(self, funcs, domain):
        assert isinstance(domain, vecsps.DirectSum)
        self.length = len(domain.summands)
        """Number of the summands in the direct sum domain. 
        """
        for i in range(self.length):
            assert isinstance(funcs[i], Functional)
            assert funcs[i].domain == domain.summands[i] 
        self.funcs = funcs
        """List of the functionals on each summand of the direct sum domain.
        """
        super().__init__(domain)

    def _eval(self, x):
        splitted = self.domain.split(x)
        toret = 0 
        for i in range(self.length):
            toret += self.funcs[i](splitted[i])
        return toret

    def _gradient(self, x):
        splitted = self.domain.split(x)
        gradients = []
        for i in range(self.length):
            gradients.append( self.funcs[i](splitted[i]) )
        return np.asarray(gradients).flatten()

    def _hessian(self, x):
        raise NotImplementedError

    def _proximal(self, x, taus):
        assert len(taus) == self.length
        splitted = self.domain.split(x)
        proximals = []
        for i in range(self.length):
            proximals.append( self.funcs[i].proximal(splitted[i], taus[i]) )
        return np.asarray(proximals).flatten()


class Indicator(Functional):
    r"""Indicator function on the domain defined by some function evaluation to `True` on some subset of the `domain`
    \[
        \chi_f(x) := 
        \begin{cases}
        0\;\; if\;f(x)\;is\,true \\
        \infty\;\; else
        \end{cases}.
    \]

    Parameters
    ----------
    domain : regpy.vecsps.VectorSpace
        Underlying domain on which the functional is defined.
    predicate : (regpy.vecsps.VectorSpace -> boolean)
        Function evaluating the truth value of elements in the domain.    
    """
    def __init__(self, domain, predicate):
        super().__init__(domain)
        self.predicate = predicate
        """Function evaluating the truth value of elements in the domain.
        """

    def _eval(self, x):
        if self.predicate(x):
            return 0
        else:
            return np.inf

    def _gradient(self, x):
        # This is of course not correct, but lets us use an Indicator functional to force
        # rejecting an MCMC proposals without altering the gradient.
        return self.domain.zeros()

    def _hessian(self, x):
        return operators.Zero(self.domain)

    def _proximal(self, x, tau):
        return NotImplementedError


class ErrorToInfinity(Functional):
    """Can be used in cases when a functional will most likely throw an exception. In such a case the return
    value will be `np.inf`. The gradient will in such a case be zero. 

    Parameters
    ----------
    func : regpy.functionals.Functional
        Functional to be modified to have infinity whenever an exception is thrown. 
    """
    def __init__(self, func):
        super().__init__(func.domain)
        self.func = func
        """Functional to be modified to have infinity whenever an exception is thrown.
        """

    def _eval(self, x):
        try:
            return self.func(x)
        except:
            return np.inf

    def _gradient(self, x):
        try:
            return self.func.gradient(x)
        except:
            return self.domain.zeros()

class HilbertNormGeneric(Functional):
    r"""Generic implementation of the HilbertNorm \(1/2*\Vert x\Vert^2\). Proximal operator defined on `h_space`.

    Parameters
    ----------
    h_space : regpy.hilbert.HilbertSpace
        Hilbert space used for norm. 
    h_domain : regpy.hilbert.HilbertSpace
        Hilbert Space wrt the proximal operator gets computed. (Defaults : h_space)
    """
    def __init__(self, h_space, h_domain=None):
        assert isinstance(h_space, hilbert.HilbertSpace)
        super().__init__(h_space.vecsp, h_domain= h_domain or h_space)
        self.h_space = h_space
        """ Hilbert space used for norm.
        """

    def _eval(self, x):
        return np.real(np.vdot(x, self.h_space.gram(x))) / 2

    def _linearize(self, x):
        gx = self.h_space.gram(x)
        y = np.real(np.vdot(x, gx)) / 2
        return y, gx

    def _gradient(self, x):
        return self.h_space.gram(x)

    def _hessian(self, x):
        return self.h_space.gram

    def _proximal(self, x, tau, cg_pars=None):
        if self.h_domain == self.h_space:
            return 1/(1+tau)*x
        else:
            op = self.h_domain.gram+tau*self.h_space.gram
            inverse = operators.CholeskyInverse(op)
            return inverse(self.h_domain.gram(x))
        

class IntegralFunctionalBase(Functional):
    r"""
    This class provides a general framework for Integral functionals of the type
    \[
    F\colon X \to \mathbb{R}
    \]
    \[
    v\mapsto \Int_\Omega f(v(x),w(x))\mathrm{d}x
    \]
    with \(f\colon \mathbb{R}^2\to \mathbb{R})\ some function and \(w\colon\Omega\to\mathbb{R})\
    defining some reference function. 

    Subclasses defining explicit functionals of this type have to implement
        `_f` evaluation the function \(f)\
        `_f_deriv` giving the derivative \(\partial_1 f)\
        `_f_prox` giving the prox of \(v>->f(v,w))\
    since 
    \[
    F'[g]h = \int_\Omega h(x)(\partial_1 f)(g(x),w(x))
    \]
    is a functional of the same type and
    \[
    \mathrm{prox}_F(v)(x) = \mathrm{prox}_f(v(x),x).
    \]

    Parameters
    ----------
    domain : `regpy.vecsps.MeasureSpaceFcts`
        Domain on which it is defined. Needs some Measure therefore a MeasureSpaceFcts
    h_domain : `regpy.hilbert.HilbertSpace`
        Hilbert Space defined on `domain`. Proximal operator needs to be computed 
    wrt to that.
    """

    def __init__(self,domain,h_domain,wref=None):
        assert isinstance(domain,vecsps.MeasureSpaceFcts)
        assert domain == h_domain.vecsp
        assert wref is None or np.isscalar(wref) or wref.shape == domain.shape
        self.wref = wref
        super().__init__(domain)
        self.h_domain = h_domain
        """ Hilbert space on `domain` wrt to which is the prox computed."""

    def _eval(self, v):
        return np.sum(self._f(v,self.wref)*self.domain.measure)

    def _conj(self,vstar):
        return np.sum(self._f_conj(vstar/self.domain.measure,self.wref)*self.domain.measure)

    def _gradient(self, v):
        return self._f_deriv(v,self.wref)*self.domain.measure

    def _hessian(self, v):
        return operators.PtwMultiplication(self.domain,self._f_second_deriv(v,self.wref))

    def _proximal(self, v, tau):
        return self._f_prox(v,self.wref,tau)
    
    def _proximal_conj(self, vstar, tau):
        return self._f_prox_conj(vstar/self.domain.measure,self.wref,tau)*self.domain.measure
    
    def _conj_gradient(self, vstar):
        return self._f_conj_deriv(vstar/self.domain.measure,self.wref)
    
    def _conj_hessian(self, vstar):
        return operators.PtwMultiplication(self.domain,self._f_conj_second_deriv(vstar/self.domain.measure,self.wref))

    def _f(self,v,w):
        raise NotImplementedError
    
    def _f_deriv(self,v,w):
        raise NotImplementedError

    def _f_second_deriv(self,v,w):
        raise NotImplementedError

    def _f_prox(self,v,w,tau):
        """TODO: write default implementation by Newton's method"""
        raise NotImplementedError
    
    def _f_conj(self,vstar,w):
        raise NotImplementedError
    
    def _f_conj_deriv(self,vstar,w):
        raise NotImplementedError

    def _f_conj_second_deriv(self,vstar,w):
        raise NotImplementedError

    def _f_prox_conj(self,vstar,w,tau):
        raise NotImplementedError
#        """default implementation by Moreau's identity """
#        return vstar-tau*self._f_prox(vstar/tau,w,1/tau)
    
class LppPower(IntegralFunctionalBase):
    r"""
    Implements the \(p)\-power of the \(L^p)\ norm on some domain in `MeasureSpaceFcts`
    as an integral functional.

    Parameters
    ----------
    domain : `regpy.vecsps.MeasureSpaceFcts`
        Domain on which it is defined. Needs some Measure therefore a MeasureSpaceFcts
    """

    def __init__(self, domain, p=2):
        assert np.isscalar(p) and p >1
        self.p = p
        self.q = p/(p-1)
        super().__init__(domain, hilbert.L2(domain))

    def _f(self,v,w):
        return np.abs(v)**self.p/self.p
    
    def _f_deriv(self, v,w):
        return np.abs(v)**(self.p-1)*np.sign(v)
    
    def _f_second_deriv(self, v,w):
        return (self.p-1)*np.abs(v)**(self.p-2)
    
    def _f_conj(self, vstar,w):
        return np.abs(vstar)**self.q/self.q

    def _f_conj_deriv(self, vstar,w):
        return np.abs(vstar)**(self.q-1)*np.sign(vstar)
    
    def _f_conj_second_deriv(self, vstar,w):
        return (self.q-1)*np.abs(vstar)**(self.q-2)
    

    def _f_prox(self, x_hat,tau,w):
        raise NotImplementedError

class L1MeasureSpace(IntegralFunctionalBase):
    r"""\(L ^1\) Functional on `MeasureSpace`. Proximal implemented for default \(L^2\) as `h_domain`.

    Parameters
    ----------
    domain : regpy.vecsps.VectorSpace
        Domain on which to define the generic L1.
    """
    def __init__(self, domain):
        super().__init__(domain,hilbert.L2(domain))

    def _f(self, v,w):
        return np.abs(v)

    def _f_deriv(self, v,w):
        assert not np.any(v==0)
        return np.sign(v)

    def _f_second_deriv(self, v, w):
        assert not np.any(v==0)
        return np.zeros_like(v)

    def _f_prox(self, v,w, tau):
        return np.maximum(0, np.abs(v)-tau)*np.sign(v)

    def _f_conj(self, v_star,w):
        ind = (np.abs(v_star)>1)
        res = np.zeros_like(v_star)
        res[ind]=np.inf
        return res
    
    def _f_conj_der(self, v_star,w):
        assert abs(v_star)<=1
        return np.zeros_like(v_star)

class KullbackLeibler(IntegralFunctionalBase):
    r"""Kullback-Leiber divergence define by
    \[ 
        F(u,w) = KL(w,u) = \int (u(x) -w(x) - w(x)\ln \frac{u(x)}{w(x)}) dx
    \]

    Parameters
    ----------
    domain : regpy.vecsps.VectorSpace
        Domain on which to define the Kullback-Leibler divergence
    """

    def __init__(self, domain,w):
        super().__init__(domain,hilbert.L2(domain))
        assert w in domain
        assert np.min(w)>=0
        self.wref = w

    def _f(self, u,w):
        res = u-w - w * np.log(u/w)
        ind_uneg = (u<0)
        res[ind_uneg] = np.inf
        ind_uzero = np.logical_and(u==0,np.logical_not(w==0))
        res[ind_uzero] = np.inf
        return res    
   
    def _f_deriv(self, u,w):
        assert np.min(u)>=0
        assert np.all(np.logical_or(np.logical_not(u==0),w==0))
        res = np.ones_like(u)-w/u
        res[u==0] = 1
        return res

    def _f_second_deriv(self, u, w):
        assert np.min(u)>0
        return w/u**2

    def _f_conj(self, u_star,w):
        if np.any(u_star)>1:
            return np.inf 
        elif np.any(np.logical_and(u_star == 1,np.logical_not(w==0))):
            return np.inf 
        else:
            return -w*np.log(1-u_star)

    def _f_conj_deriv(self, u_star,w):
        assert np.max(u_star)<=1
        assert np.all(np.logical_or(np.logical_not(u_star==1),w==0))
        return w/(1-u_star)
    
    def _f_conj_second_deriv(self, u_star,w):
        assert np.max(u_star)<=1
        assert np.all(np.logical_or(np.logical_not(u_star==1),w==0))
        return w/(1-u_star)**2

class RelativeEntropy(IntegralFunctionalBase):
    r"""Kullback-Leiber divergence define by
    \[ 
        F(u,w) = \int (u(x)\ln \frac{u(x)}{w(x)}) dx
    \]

    Parameters
    ----------
    domain : regpy.vecsps.VectorSpace
        Domain on which to define the Kullback-Leibler divergence
    """

    def  __init__(self, domain,w):
        super().__init__(domain,hilbert.L2(domain))
        assert w in domain
        assert np.min(w)>0
        self.wref = w

    def _f(self, u,w):
        res =  u * np.log(u/w)
        ind_uneg = (u<0)
        res[ind_uneg] = np.inf
        res[u==0] = 0
        return res    
   
    def _f_deriv(self, u,w):
        assert np.min(u)>0
        res = np.ones_like(u)+np.log(u/w)
        return res

    def _f_second_deriv(self, u, w):
        assert np.min(u)>0
        return 1/u

    def _f_conj(self, u_star,w):
        return w*(np.exp(u_star-1))

    def _f_conj_deriv(self, u_star,w):
        return w*np.exp(u_star-1)
    
    def _f_conj_second_deriv(self, u_star,w):
        return w*np.exp(u_star-1)

class L1Generic(Functional):
    r"""Generic \(L ^1\) Functional. Proximal implemented for default \(L^2\) as `h_domain`.

    Parameters
    ----------
    domain : regpy.vecsps.VectorSpace
        Domain on which to define the generic L1.
    """
    def __init__(self, domain):
        super().__init__(domain)

    def _eval(self, x):
        return np.sum(np.abs(x))

    def _gradient(self, x):
        return np.sign(x)

    def _hessian(self, x):
        # Even approximate Hessians don't work here.
        raise NotImplementedError

    def _proximal(self, x, tau):
        return np.maximum(0, np.abs(x)-tau)*np.sign(x)


class TVGeneric(Functional):
    """Generic TV Functional. Proximal implemented for default L2 h_space

    NotImplemented yet!
    """
    def __init__(self, domain, h_domain=hilbert.L2):
        super().__init__(domain,h_domain=h_domain)

    def _gradient(self, x):
        return NotImplementedError

    def _hessian(self, x):
        return NotImplementedError
    
    def _proximal(self, x, tau):
        return NotImplementedError

class TVUniformGridFcts(Functional):
    """Total Variation Norm: For C^1 functions the l1-norm of the gradient on a Uniform Grid

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

    def _gradient(self, x):
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
        """Computes the gradient of field given by 'u'. 'u' is defined on a 
        equidistant grid. Returns a list of vectors that are the derivatives in each 
        dimension."""
        # Need to reshape spacing otherwise getting braodcasting error
        shape = [self.domain.ndim]+[1 for _ in self.domain.shape]
        return 1/self.domain.spacing.reshape(shape)*np.array(np.gradient(u))

    def _divergenceuniformgrid(self, u):
        """Computes the divergence of a vector field 'u'. 'u' is assumed to be
        a list of matrices u=(u_x, u_y, u_z, ...) holding the values for u on a
        regular grid"""
        return 1/self.domain.spacing*np.ufunc.reduce(np.add, [np.gradient(u[i], axis=i) for i in range(self.dim)])

def as_functional(func, vecsp):
    r"""Convert `func` to Functional instance on vecsp.

    - If func is a `HilbertSpace` then it generated the `HilbertNormGeneric`.
    - If func is an Operator, it's wrapped in a `GramHilbertSpace` and then `HilbertNormGeneric` functional.
    - If func is callable, e.g. an `hilbert.AbstractSpace` or `AbstractFunctional`, it is called on `vecsp` to construct the concrete functional or Hilbert space. In the later case the functional will be the `HilbertNormGeneric`

    Parameters
    ----------
    func : Functional or HilbertSapce or Operator or callable
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
    """Auxiliary method to register abstract functionals for various vector spaces. Using the decorator
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
