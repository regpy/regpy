from collections import defaultdict

from copy import copy

import numpy as np

from regpy import operators, util, vecsps, hilbert


class Functional:
    r"""
    Base class for implementation of a functional. Subsclasses should at least implement the 
        `_eval` :  evaluating the funcitonal
    and 
        `_gradient` or `_deriv` : returning the gradtient or derivative at `x`.
    
    The evalution of a specific functional on some element of the `domain` can be done by
    simply caling the functional on that element. 
        
    Funcationals can be added by taking `LinearCombination` of them. The `domain` has to be the
    same for each functional. 

    They can also be multiplied by scalars or `np.ndarrays`of `domain.shape`or multiplied by 
    `regpy.operators.Operator`. This leads to a functional that is composed with the operator
    \(F\circ O\) where \(F\) is the functional and $O$ some operator. Multiplying by a scalar
    results in a composition with the `PtwMultiplication` operator.

    Parameters
    ----------
    daomain : regpy.vecsps.VectorSpace
        The uncerlying vector space for the function space on which it is defined.
    h_domain : regpy.hilbert.HilbertSpace (default: `L2(domain)`)
        The underlying Hilbert wrt which the proximal is conputed.
    """
    def __init__(self, domain, h_domain=None):
        # TODO implement domain=None case
        assert isinstance(domain, vecsps.VectorSpace)
        self.domain = domain
        """The underlying vector space."""
        self.h_domain = hilbert.as_hilbert_space(h_domain,domain) or hilbert.L2(domain)
        """The underlying Hilbert space."""

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
            F(x+\epsilon h) = F(x) + \epsion \langle \nabla F[x],h\rangle + \mathcal{O}(\epsilon^2)
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
        Gradient of the functional at `x` where
        \[
            F(x+\epsilon h) = F(x) + \epsion \langle \nabla F[x],h\rangle + \mathcal{O}(\epsilon^2)
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

    def hessian(self, x):
        """The hessian of the functional at `x` as an `regpy.operator.Operator` maping form the 
        functionals `domain` to it self. Requires the implementation of `_hessian` or by default
        computes the `regpy.operators.ApproximateHessian`.

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
        h = self._hessian(x)
        assert isinstance(h, operators.Operator)
        assert h.linear
        assert h.domain == h.codomain == self.domain
        return h

    def proximal(self, x, tau, proximal_pars = None):
        r"""Proximal operator 
        \[
            \mathrm{prox} _{F}(x)=\arg \min _{v\in {\mathcal {X}}}(F(v)+{\frac{1}{2\tau}}\Vert v-x\Vert_{\mathcal {X}}^{2}).
        \]
        Requires and implementation of `_proximal`.

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
        proximal = self._proximal(x, tau, **proximal_pars)
        assert proximal in self.domain
        return proximal

    def _eval(self, x):
        raise NotImplementedError

    def _linearize(self, x):
        raise NotImplementedError

    def _gradient(self, x):
        raise NotImplementedError

    def _hessian(self, x):
        return operators.ApproximateHessian(self, x)

    def _proximal(self, x, tau):
        return NotImplementedError

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
            return Shifted(self, other)
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
        for func, coeff in coeff_for_func.items():
            self.coeffs.append(coeff)
            self.funcs.append(func)

        domains = [func.domain for func in self.funcs if func.domain]
        if domains:
            domain = domains[0]
            assert all(d == domain for d in domains)
        else:
            domain = None

        super().__init__(domain)

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

    def _proximal(self, x, tau, proximal_params):
        if len(self.funcs) == 1:
            return self.funcs[0].proximal(x,self.coeffs[0]*tau)
        else:
            return NotImplementedError


class Shifted(Functional):
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

    def _proximal(self, x, tau):
        return self.func.proximal(x, tau)


class Composed(Functional):
    """Composition of an operator with a functional \(F\circ O\). This should not be called
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

    def _proximal(self, x, tau, cg_params={}):
        # In case it is a functional 1/2||Tf-g^delta||^2 can approximated by a Tikhonov solver
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
            return AbstractShifted(self, other)
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

class AbstractShifted(AbstractFunctional):
    r"""Abstract analogue to `Shifted` class. Shifting a functional by some offset. Should not be used directly but rather by adding some scalar to the functional. 

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
        return Shifted(func=self.func(vecsp),offset=self.offset)
    
class AbstractComposed(AbstractFunctional):
    """Abstract analogue to `Composed`. Composition of an operator with a functional \(F\circ O\). This should not be called
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

    Notes
    -----
    The proximal operator is the projection on the set predicate.
    However, it is more natural to implement indicator function constraints in Tikhonov 
    regularization by semismooth approaches. See semismooth Newton method.
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
    $$
    F\colon X \to \mathbb{R}
    $$
    $$
    v\mapsto \Int_\Omega f(w(x)v(x))\mathrm{d}x
    $$
    with $f\colon \mathbb{R}\ro \mathbb{R}$ some function and $w\colon\Omega\to\mathbb{R}$
    defining some whieght function. 

    Subclasses defining explicit functionals of this type have to implement
        `_f` evaluation the function $f$
        `_f_deriv` giving the derivative $f'$
        `_f_porx` giving the prox of $f$
    since 
    $$
    F'[g]h = \int_\Omega h(x)w(x)f'(w(x)g(x))
    $$
    is a functional of the same type and
    $$
    \mathrm{prox}_F(v)(x) = \mathrm{prox}_f(w(x)v(x)).
    $$

    Parameters
    ----------
    domain : `regpy.vecsps.MeasureSpaceFcts`
        Domain on which it is defined. Needs some Measure therefore a MeasureSpaceFcts
    h_domain : `regpy.hilbert.HilbertSpace`
        Hilbert Space defined on `domain`. Proximal operator needs to be computed 
    wrt to that.
    """

    def __init__(self,domain,h_domain,weight = 1):
        assert isinstance(domain,vecsps.MeasureSpaceFcts)
        assert domain == h_domain.vecsp
        assert np.isscalar(weight) or weight.shape == domain.shape
        self.weight = weight
        """ Weights to multipy. """
        super().__init__(domain)
        self.h_domain = h_domain
        """ Hilbert space on `domain` wrt to which is the prox computed."""

    def _eval(self, x):
        return np.sum(self._f(self.weight*x)*self.domain.measure)

    def _gradient(self, x):
        return self.weight*self._f_deriv(self.weight*x)

    def _hessian(self, x):
        raise NotImplementedError

    def _proximal(self, x, tau):
        return self._f_prox(self.weight*x,tau)
    
    def _f(self,x_hat):
        raise NotImplementedError
    
    def _f_deriv(self,x_hat):
        raise NotImplementedError
    
    def _f_prox(self,x_hat,tau):
        raise NotImplementedError
    
class LppPower(IntegralFunctionalBase):
    r"""
    Implements the $p$-power of the $L^p$ norm on some domain in `MeasureSpaceFcts`
    as an integral functional.

    Parameters
    ----------
    domain : `regpy.vecsps.MeasureSpaceFcts`
        Domain on which it is defined. Needs some Measure therefore a MeasureSpaceFcts
    """

    def __init__(self, domain, p=2):
        assert np.isscalar(p) and p >1
        self.p = p
        super().__init__(domain, hilbert.L2(domain))

    def _f(self,x_hat):
        return np.abs(x_hat)**self.p
    
    def _f_deriv(self, x_hat):
        return 1/self.p*np.abs(x_hat)**(self.p-1)*np.sign(x_hat)
    
    def _f_prox(self, x_hat,tau):
        raise NotImplementedError

class L1MeasureSpace(IntegralFunctionalBase):
    """\(L ^1\) Functional on `MeasureSpace`. Proximal implemented for default \(L^2\) as `h_domain`.

    Parameters
    ----------
    domain : regpy.vecsps.VestorSpace
        Domain on which to define the generic L1.
    """
    def __init__(self, domain):
        super().__init__(domain,hilbert.L2(domain))

    def _f(self, x):
        return np.abs(x)

    def _f_deriv(self, x):
        return np.sign(x)

    def _f_prox(self, x, tau):
        return np.maximum(0, np.abs(x)-tau)*np.sign(x)


class L1Generic(Functional):
    """Generic \(L ^1\) Functional. Proximal implemented for default \(L^2\) as `h_domain`.

    Parameters
    ----------
    domain : regpy.vecsps.VestorSpace
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

'''
Total Variation Norm: For C^1 functions the l1-norm of the gradient on a Uniform Grid
'''
from regpy.util import gradientuniformgrid
from regpy.util import divergenceuniformgrid
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
            return np.sum(np.abs(gradientuniformgrid(x, spacing=self.domain.spacing)))
        else:
            return np.sum(np.linalg.norm(gradientuniformgrid(x, spacing=self.domain.spacing), axis=0))

    def _gradient(self, x):
        if self.dim==1:
            return np.sign(gradientuniformgrid(x, spacing=self.domain.spacing))
        else:
            grad = gradientuniformgrid(x, spacing=self.domain.spacing)
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
            update = stepsize*gradientuniformgrid( self.h_domain.gram_inv( divergenceuniformgrid(p, self.dim, spacing=self.domain.spacing))-x/tau, spacing=self.domain.spacing)
            p = (p+update) / (1+np.abs(update))
        return x-tau*divergenceuniformgrid(p, self.dim, spacing=self.domain.spacing)


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
