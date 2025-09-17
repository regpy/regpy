r"""
Forward operators
=================

This module provides the basis for defining forward operators, and implements some simple
auxiliary operators. Actual forward problems are implemented in submodules.

The base class is `Operator`.
"""

from collections import defaultdict
from copy import deepcopy

from math import sqrt,inf

import numpy as np
from scipy.sparse.linalg import LinearOperator

from regpy import util, vecsps

__all__ = ["Operator", "Pow", "Identity", "CoordinateProjection", "CoordinateMask", "PtwMultiplication", "OuterShift", "InnerShift", "DirectSum", "VectorOfOperators", "MatrixOfOperators", "Sum", "Product", "RealPart", "ImaginaryPart", "SquaredModulus", "Zero", "ApproximateHessian", "SciPyLinearOperator"]


class _Revocable:
    def __init__(self, val):
        self.__val = val

    @classmethod
    def take(cls, other):
        return cls(other.revoke())

    def get(self):
        try:
            return self.__val
        except AttributeError:
            raise RuntimeError('Attempted to use revoked reference') from None

    def revoke(self):
        val = self.get()
        del self.__val
        return val

    @property
    def valid(self):
        try:
            self.__val
            return True
        except AttributeError:
            return False


class Operator:
    r"""Base class for forward operators. Both linear and non-linear operators are handled. Operator
    instances are callable, calling them with an array argument evaluates the operator.

    Subclasses implementing non-linear operators should implement the following methods:

        _eval(self, x, differentiate=False)
        _derivative(self, x)
        _adjoint(self, y)

    These methods are not intended for external use, but should be invoked indirectly via calling
    the operator or using the `Operator.linearize` method. They must not modify their argument, and
    should return arrays that can be freely modified by the caller, i.e. should not share data
    with anything. Usually, this means they should allocate a new array for the return value.

    Implementations can assume the arguments to be part of the specified vector spaces, and return
    values will be checked for consistency.

    In some cases a solver only requires the application of the composition of the adjoint with the 
    derivative. When this should be used i.e. in cases when setting up elements in the codomain is 
    not feasible one can implement the `_adjoint_derivative` method and when linearizing can set the
    flag `return_adjoint_eval = True`.

    The mechanism for derivatives and their adjoints is this: whenever a derivative is to be
    computed, `_eval` will be called first with `differentiate=True` or `return_adjoint_eval=True`, 
    and should produce the operator's value and perform any precomputation needed for evaluating 
    the derivative. Any subsequent invocation of `_derivative`, `_adjoint` and `_adjoint_derivative`
    should evaluate the  derivative, its adjoint or their composition at the same point `_eval` was 
    called. The reasoning is this

     * In most cases, the derivative alone is not useful. Rather, one needs a linearization of the
       operator around some point, so the value is almost always needed.
     * Many expensive computations, e.g. assembling of finite element matrices, need to be carried
       out only once per linearization point, and can be shared between the operator and the
       derivative, so they should only be computed once (in `_eval`).
    
    For callers, this means that since the derivative shares data with the operator, it can't be
    reliably called after the operator has been evaluated somewhere else, since shared data may
    have been overwritten. The `Operator`, `Derivative` and `Adjoint` classes ensure that an
    exception is raised when an invalidated derivative is called.

    If derivatives at multiple points are needed, a copy of the operator should be performed using
    `copy.deepcopy`. For efficiency, subclasses can add the names of attributes that are considered
    as constants and should not be deepcopied to `self._consts` (a `set`). By default, `domain` and
    `codomain` will not be copied, since `regpy.vecsps.VectorSpaceBase` instances should never
    change in-place.

    If no derivative at some point is needed, `_eval` will be called with `differentiate=False`,
    allowing it to save on precomputations. It does not need to ensure that data shared with some
    derivative remains intact; all derivative instances will be invalidated regardless.

    Linear operators should implement

        _eval(self, x)
        _adjoint(self, y)

    Here the logic is simpler, and no sharing of precomputations is needed (unless it applies to the
    operator as a whole, in which case it should be performed in `__init__`).

    Note that the adjoint should be computed with respect to the standard real inner product on the
    domain / codomain, given as

        real(domain.vdot(x, y)) or real(codomain.vdot(x, y))

    Other inner products on vector spaces are independent of both vector spaces and operators,
    and are implemented in the `regpy.hilbert` module.

    Basic operator algebra is supported:

        a * op1 + b * op2    # linear combination
        op1 * op2            # composition
        op * arr             # composition with array multiplication in domain
        op + arr             # operator shifted in codomain
        op + scalar          # dto.

    Parameters
    ----------
    domain, codomain : regpy.vecsps.VectorSpaceBase or None
        The vector space on which the operator's arguements / values are defined. Using `None`
        suppresses some consistency checks and is intended for ease of development, but should 
        not be used except as a temporary measure. Some constructions like direct sums will fail
        if the vector spaces are unknown.
    linear : bool, optional
        Whether the operator is linear. Default: `False`.
    """

    log = util.ClassLogger()

    def __init__(self, domain=None, codomain=None, linear=False, inverse=None):
        assert not domain or isinstance(domain, vecsps.VectorSpaceBase)
        assert not codomain or isinstance(codomain, vecsps.VectorSpaceBase)
        self.domain = domain
        r"""The vector space on which the operator is defined. Either a
        subclass of `regpy.vecsps.VectorSpaceBase` or `None`."""
        self.codomain = codomain
        r"""The vector space on which the operator values are defined. Either
        a subclass of `regpy.vecsps.VectorSpaceBase` or `None`."""
        self.linear = linear
        r"""Boolean indicating whether the operator is linear."""
        self._constants = {}
        """A dictionary containing constants set to inputs by the `set_constant`"""        
        self._consts = {'domain', 'codomain','lu'}
        r"""properties that are handled differently when copying the operator."""
        if inverse is not None and not isinstance(inverse, Operator):
            raise TypeError("The inverse has to be an Operator instance or None.")
        self._inverse = inverse

    def __deepcopy__(self, memo):
        cls = type(self)
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k in self._consts:
                setattr(result, k, v)
            else:
                setattr(result, k, deepcopy(v, memo))
        return result

    @property
    def attrs(self):
        r"""The set of all instance attributes. Useful for updating the `_consts` attribute via

            self._consts.update(self.attrs)

        to declare every current attribute as constant for deep copies.
        """
        return set(self.__dict__)

    def __call__(self, x):
        assert not self.domain or x in self.domain, "x of type {} is not in domain {}".format(type(x),self.domain)
        if self.linear:
            y = self._eval(self._insert_constants(x))
        else:
            self.__revoke()
            y = self._eval(self._insert_constants(x), differentiate=False)
        assert not self.codomain or y in self.codomain, "y of type {} is not in codomain {}".format(type(y),self.domain)
        return y

    def linearize(self, x, return_adjoint_eval = False):
        r"""Linearize the operator around some point.

        Parameters
        ----------
        x : array-like
            The point around which to linearize.

        adjoint_deriv : boolean (Default: False)
            Flag to determine if AdjointDerivative should be returned as additional output argument. 
            This can be used if AdjointDerivative has an a more efficient implementation than by composition 
            or if the image space of the operator is too large to store vectors in this space. 

        Returns
        -------
        if return_adjoint_eval==False: 
          array, Derivative:
             The value and the derivative at `x`, the latter as an `Operator` instance.
        if return_adjoint_eval ==True: 
           array, Derivative
               array is :math:`F'[x]^\ast F(x)`, Derivative is as above. The adjoint derivative
               AdjointDerivative that is an efficient implementation of the composition Derivative.adjoint * Derivative is accessible by Derivative.adjoint_eval
        """
        if self.linear:
            if not return_adjoint_eval:
                return self(x), self
            else:
                return self.adjoint_eval(x), self
        else:            
            assert not self.domain or x in self.domain, "x of type {} is not in domain {}".format(type(x),self.domain)
            self.__revoke()
            if not return_adjoint_eval:
                y = self._eval(self._insert_constants(x), differentiate=True)
                if self.codomain and not y in self.codomain:
                    raise RuntimeError("y of type {} is not in codomain {}".format(type(x),self.domain))
                deriv = Derivative(self.__get_handle())
                return y, deriv
            else:
                Fstar_y = self._adjoint_eval(self._insert_constants(x))
                if self.domain and not Fstar_y in self.domain: 
                    raise RuntimeError("Fstar_y of type {} is not in domain {}".format(type(x),self.domain))
                deriv = Derivative(self.__get_handle()) 
                return Fstar_y, deriv

    @util.memoized_property
    def adjoint(self):
        r"""For linear operators, this is the adjoint as a linear `regpy.operators.Operator`
        instance. Will only be computed on demand and saved for subsequent invocations.

        Returns
        -------
        Adjoint
            The adjoint as an `Operator` instance.
        """
        if not self.linear:
            raise RuntimeError('Operator is not linear.')
        return Adjoint(self)
    
    @util.memoized_property
    def adjoint_eval(self):
        r"""This is only available for linear operators, it is the composition of adjoint and 
        eval as a `regpy.operators.AdjointEval` instance. Will only be computed on
         demand and saved for subsequent invocations.

        Returns
        -------
        Adjoint
            The adjoint as an `Operator` instance.
        """
        if not self.linear:
            raise RuntimeError('Operator is not linear.')
        return AdjointEval(self)

    def __revoke(self):
        try:
            self.__handle = _Revocable.take(self.__handle)
        except AttributeError:
            pass

    def __get_handle(self):
        try:
            return self.__handle
        except AttributeError:
            self.__handle = _Revocable(self)
            return self.__handle

    def _eval(self, x, differentiate=False, return_adjoint_eval = False):
        raise NotImplementedError

    def _derivative(self, x):
        raise NotImplementedError

    def _adjoint(self, y):
        raise NotImplementedError
    
    def adjoint_data(self, data):
        return self._adjoint(data)
    
    def _adjoint_eval(self, x):
        if self.linear:
            return self._adjoint(self._eval(x))
        else:
            y,deriv = self.linearize(x)
            return deriv.adjoint(y)

    def _adjoint_derivative(self, x):
        if self.linear:
            return self._adjoint(self._eval(x))
        else:
            return self._adjoint(self._derivative(x))

    @property
    def inverse(self):
        r"""A property containing the  inverse as an `Operator` instance. In most cases this will
        just raise a `NotImplementedError`, but subclasses may override this if possible and useful.
        To avoid recomputing the inverse on every access, `regpy.util.memoized_property` may be
        useful."""
        if self._inverse is None:
            raise NotImplementedError("The inverse of the operator {} is not known.".format(self))
        return self._inverse
    
    @inverse.setter
    def inverse(self, inv):
        if inv is None:
            self._inverse = None
            self.log.info("Setting the inverse of the operator {} to None".format(self))
        elif not isinstance(inv, Operator):
            raise TypeError("The inverse has to be an Operator instance.")
        self.log.info("Setting the inverse of the operator {} to {} overwriting the old {}.".format(self,inv,self._inverse))
        self._inverse = inv

    def as_linear_operator(self):
        r"""Creating a `scipy.linalg.LinearOperator` from the defined linear operator.  

        Returns
        -------
        scipy.linalg.LinearOperator 
            The linear operator as a scipy linear operator.

        Raises
        ------
        RuntimeError
            If operator flag `linear` is False. 
        """
        if self.linear:
            return SciPyLinearOperator(self)
        else:
            raise RuntimeError('Operator is not linear.')
        
    def norm(self,h_domain=None,h_codomain=None,method=None,use_adjoint_derivative=False):
        r"""Approximate the operator norm of  a linear operator with respect to the vector norms of h_domain and h_codomain. 
        By default this is achieved by computing the largest eigenvalue of \(T^*T\) using eigsh from scipy. 
        # To-do: Test making this a memoized property (should only be recomputed if non-linear, should be possible for user to input if analytically known).    
        #@memoized_property
 
        Parameters
        ----------
        h_domain: Hilbert space on the domain. Defaults to L2 if None.
        h_codomain: Hilbert space on the codomain. Defaults to L2 if None.
        method: string [default: None]
            Method by which an approximation of the operator norm is computed. If None uses self.default_norm_method or 'lanczos'
              if this is not set. Alternative: "power" for power method
        Returns
        -------
        scalar
            Approximation of the norm of T^*T. 

        Raises
        ------
        NotImplementedError
            If the operator is nonlinear or the method is not implemented.
        """

        if(not self.linear):
            raise NotImplementedError
        from regpy.hilbert import L2
        if(h_domain is None):
            h_domain=L2(self.domain)
        else:
            assert h_domain.vecsp==self.domain
        if(h_codomain is None):
            h_codomain=L2(self.codomain)
        else:
            if use_adjoint_derivative:
                raise Warning('value of h_codomain will be ignored!')
            else:
                assert h_codomain.vecsp==self.codomain
        method=getattr(self,'default_norm_method','lanczos') if method is None else method
        if method == "power":
            return self._power_method(h_domain,h_codomain,use_adjoint_derivative=use_adjoint_derivative)
        elif method == "lanczos":
            from scipy.sparse.linalg import eigsh
            if use_adjoint_derivative:
                op = self.adjoint_eval
            else:
                op = self.adjoint * h_codomain.gram * self
            return sqrt(eigsh(SciPyLinearOperator(op), 1, M=SciPyLinearOperator(h_domain.gram),tol=0.01)[0][0])
        else:
            raise NotImplementedError

    def _power_method(self,h_domain,h_codomain,max_iter=int(1e2),stopping_rule=1e-12,use_adjoint_derivative = False):
        r"""Approximation of operator norm by the power method. Should not be used directly and only be called via norm.

        Parameters
        ----------
        h_domain: Hilbert space on the domain.
        h_codomain: Hilbert space on the codomain.
        max_iter: int maximum number of iterations
        stopping_rule: float Iteration is stopped if relative residual is smaller than this value.
        """
        x = self.domain.rand()
        relative_residual = inf
        if use_adjoint_derivative:
            op = self.adjoint_eval
        else:
            op = self.adjoint * h_codomain.gram * self
        for _ in range(max_iter):
            if relative_residual < stopping_rule:
                break
            ystar = op(x)
            y = h_domain.gram_inv(ystar)
            lmb = sqrt(self.domain.vdot(y, ystar).real)
            relative_residual = h_domain.norm(y - lmb * x)
            x = y/lmb
        return sqrt(lmb)
    
    def set_constant(self,c,index):
        """Assuming the operator you defined has a domain that is composed of multiple
        inputs and thus a direct sum, this method allows you to fix one of that inputs
        as a constant. 

        This method changes the domain to either a direct sum of the remaining
        components or just the remaining component.

        Moreover, it asserts if the remaining operator is linear using the utility method
        and changes the linearity flag.

        Parameters
        ----------
        c : array-type or scalar
            The constant array to be set.
        index : int
            The index to be set as a constant. The index is with respect to the full domain.
        """
        self.__revoke()
        if not hasattr(self, "full_domain"):
            if not isinstance(self.domain,vecsps.DirectSum):
                raise TypeError("Cannot set a constant when domain is {}, require the domain to be a DirectSum".format(type(self.domain)))
            self.full_domain = deepcopy(self.domain)
        
        if not isinstance(index,int):
            raise TypeError("The index has to be an integer, was given {}".format(type(index)))
        if index<0 or index>=len(self.full_domain):
            raise IndexError("The used index is {} is out of range.".format(index))
        if len(set(range(len(self.full_domain)))-self._constants.keys()-{index}) == 0:
            raise ValueError("By setting the index {} their is no input remaining please choose another index or release some other constant.".format(index))
        if c in self.full_domain[index]:
            pass
        elif isinstance(c,int) or isinstance(c,float) or (isinstance(c,complex) and self.full_domain[index].is_complex):
            c = c*self.full_domain[index].ones()
        else:
            raise ValueError("The given constant is not in the {} component of type {}. Was given something of type {}".format(index,type(self.full_domain[index]),type(c)))
        
        self._constants[index] = c
        self.domain = vecsps.DirectSum(*[d_i for i,d_i in enumerate(self.full_domain) if i not in self._constants.keys()])
        if len(self.domain) == 1:
            self.domain = self.domain[0]

        self.adjoint.codomain = self.domain
        
        if not self.linear:
            self.linear = util.operator_tests.test_linearity(self)
    
    def reset_constants(self):
        """Resets the constants set by `set_constant` to an empty dictionary. This will
        also reset the domain to the full domain.
        """
        self._constants = {}
        if hasattr(self, "full_domain"):
            self.domain = self.full_domain
            self.adjoint.codomain = self.full_domain
        else:
            raise RuntimeError("Cannot reset constants, no full domain set.")
        self.linear = util.operator_tests.test_linearity(self)

    def get_constants(self):
        """Returns the constants set by `set_constant` as a dictionary. The keys are the indices
        of the full domain and the values are the constants set.
        
        Returns
        -------
        dict
            The dictionary with the constants set by `set_constant`.
        """
        return self._constants

    def _insert_constants(self,x):
        """Inserts the constants into the vector.

        Parameters
        ----------
        x : array-type
            The vector in the reduced domain that to be completed with constants.

        Returns
        -------
        array-type
            The vector in the full domain with the constants put into the places 
            to be kept constant. If no constants are set return x. 
        """
        assert x in self.domain, "Somehow the passed vector does not belong to the reduced domain."
        if hasattr(self, "full_domain") and len(self._constants)>0:
            x_full_split = self.full_domain.zeros()
            if isinstance(self.domain,vecsps.DirectSum):
                x_split = x
            else:
                x_split = [x]
            m = 0
            for i in range(len(self.full_domain)):
                if i in self._constants.keys():
                    x_full_split[i] = self._constants[i]
                else:
                    x_full_split[i] = x_split[m]
                    m += 1
            return x_full_split
        else:
            return x
     
    def _reduce_to_domain(self,y):
        """Remove the constant coefficients in the vector where one has specified 
        constants in the original operator.

        Parameters
        ----------
        y : array-type
            The vector in the full codomain that to be reduced to the new domain.

        Returns
        -------
        array-type
            The vector in the reduced domain with the the places 
            to be kept constant removed. If no constants are set return x. 
        """
        if hasattr(self, "full_domain") and len(self._constants)>0:
            assert y in self.full_domain, "Expected the vector to be in the full codomain of the Adjoint."
            y_full_split = self.full_domain.split(y)
            if isinstance(self.domain,vecsps.DirectSum):
                return self.domain.join(*[y_full_split[i] for i in set(range(len(self.full_domain)))-self._constants.keys()])
            else:
                return [y_full_split[i] for i in set(range(len(self.full_domain)))-self._constants.keys()][0]
        else:
            return y
    
    def __mul__(self, other):
        if np.isscalar(other) and other == 1:
            return self
        elif isinstance(other, Operator):
            return Composition(self, other)
        elif np.isscalar(other) or other in self.domain:
            return self * PtwMultiplication(self.domain, other)
        else:
            return NotImplemented

    def __rmul__(self, other):
        if np.isscalar(other):
            if other == 1:
                return self
            else:
                return LinearCombination((other, self))         
        elif other in self.codomain:
            return PtwMultiplication(self.codomain, other) * self
        elif isinstance(other, Operator):
            return Composition(other, self) 
        else:
            return NotImplemented
    
    def __imul__(self,other):
        return self*other

    def __add__(self, other):
        if np.isscalar(other) and other == 0:
            return self
        elif isinstance(other, Operator):
            return LinearCombination(self, other)
        elif np.isscalar(other) or other in self.codomain:
            return OuterShift(self, other)
        else:
            return NotImplemented

    def __radd__(self, other):
        return self + other
    
    def __iadd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other
    
    def __isub__(self,other):
        return self - other

    def __neg__(self):
        return (-1) * self

    def __pos__(self):
        return self
    
    def __pow__(self, power):
        return Pow(self, power)
    
    def __getitem__(self,val):
        if val is None:
            return self
        return PartOfOperator(self,val)


class Adjoint(Operator):
    r"""An proxy class wrapping a linear operator. Calling it will evaluate the operator's
    adjoint. This class should not be instantiated directly, but rather through the
    `Operator.adjoint` property of a linear operator.

    Parameters
    ----------
    op : Operator
        The base operator giving rise to this adjoint.
    """

    def __init__(self, op):
        assert op.linear
        self.op = op
        r"""The underlying operator."""
        super().__init__(op.codomain, op.domain, linear=True)
        if hasattr(self.op,"full_domain"):
            self._constants = {}
            """The constant inputs of the op need to be set in adjoint evaluation
            """
            self.full_domain = op.codomain

    def _eval(self, x):
        return self.op._reduce_to_domain(self.op._adjoint(x))

    def _adjoint(self, x):
        return self.op._eval(self._insert_constants(x))

    @util.memoized_property
    def adjoint(self):
        return self.op

    @Operator.inverse.getter
    def inverse(self):
        if self._inverse is not None:
            return self._inverse
        try:
            return self.op.inverse.adjoint
        except NotImplementedError:
            raise NotImplementedError("The inverse of the adjoint operator {} is not known.".format(self))

    def __repr__(self):
        return util.make_repr(self, self.op)


class Derivative(Operator):
    r"""An proxy class wrapping a non-linear operator. Calling it will evaluate the operator's
    derivative. This class should not be instantiated directly, but rather through the
    `Operator.linearize` method of a non-linear operator.

    Parameters
    ----------
    op : Operator
        The base operator giving rise to this derivative.
    """

    def __init__(self, op):
        if not isinstance(op, _Revocable):
            # Wrap plain operators in a _Revocable that will never be revoked to
            # avoid case distinctions below.
            op = _Revocable(op)
        self.op = op
        r"""The underlying operator."""
        _op = op.get()
        """The underlying operator."""
        super().__init__(_op.domain, _op.codomain, linear=True)
        # Setting the corresponding constants of op to zero
        if hasattr(_op,"full_domain"):
            self.full_domain = _op.full_domain
            self._constants = {index : self.full_domain[index].zeros() for index in _op._constants}

    def _eval(self, x):
        return self.op.get()._derivative(x)

    def _adjoint(self, x):
        return self._reduce_to_domain(self.op.get()._adjoint(x))
    
    def adjoint_data(self, x):
        return self._reduce_to_domain(self.op.get().adjoint_data(x))
    
    def _adjoint_eval(self, x):
        return self._reduce_to_domain(self.op.get()._adjoint_derivative(x))

    def __repr__(self):
        return util.make_repr(self, self.op.get())


class AdjointEval(Operator):
    r"""A proxy class wrapping a linear operator :math:`F`. Calling it will evaluate the
     composition of the operator's adjoint with its evaluation :math:`F^\ast\circ F`. This
     class should not be instantiated directly, but rather through the `Operator.
     adjoint_eval` method of a linear operator.
    The `_eval` and `_adjoint` require the implementation of `_adjoint_eval` note that only 
    one implementation is needed as it is a selfadjoint operator.

    Parameters
    ----------
    op : Operator
        The base operator giving rise to this combination of adjoint and derivative.
    """

    def __init__(self, op):
        if not isinstance(op, Operator):
            raise TypeError("The input has to be an Operator instance.")
        if not op.linear:
            raise RuntimeError('Operator is not linear cannot create AdjointEval.')
        self.op = op
        super().__init__(op.domain, op.domain, linear=True)
        # Setting the corresponding constants of op to zero
        if hasattr(self.op,"full_domain"):
            self.full_domain = self.op.full_domain
            self._constants = {index : self.full_domain[index].zeros() for index in self.op._constants}

    def _eval(self, x):
        return self._reduce_to_domain(self.op._adjoint_eval(self._insert_constants(x)))

    def _adjoint(self, x):
        return self._reduce_to_domain(self.op._adjoint_eval(self._insert_constants(x)))
    
    def adjoint_data(self, x):
        return self._reduce_to_domain(self.op.adjoint_data(x))

    def __repr__(self):
        return util.make_repr(self, self.op)


class AdjointDerivative(Operator):
    r"""A proxy class wrapping a non-linear operator :math:`F`. Calling it will evaluate the composition of the operator's
    derivative adjoint with its derivative :math:`F'^\ast\circ F'`. This class should not be instantiated directly, 
    but rather through the `Operator.linearize` method of a non-linear operator with the flag `return_adjoint_eval = True`.
    The `_eval` and `_adjoint` require the implementation of `return_adjoint_eval` note that only one implementation is 
    needed as it is a selfadjoint operator.

    Parameters
    ----------
    op : Operator
        The base operator giving rise to this combination of adjoint and derivative.
    """

    def __init__(self, op):
        if not isinstance(op, _Revocable):
            # Wrap plain operators in a _Revocable that will never be revoked to
            # avoid case distinctions below.
            op = _Revocable(op)
        self.op = op
        r"""The underlying operator."""
        _op = op.get()
        super().__init__(_op.domain, _op.domain, linear=True)
        # Setting the corresponding constants of op to zero
        if hasattr(self.op,"full_domain"):
            self.full_domain = self.op.full_domain
            self._constants = {index : self.full_domain[index].zeros() for index in self.op._constants}

    def _eval(self, x):
        return self._reduce_to_domain(self.op.get()._adjoint_derivative(self._insert_constants(x)))

    def _adjoint(self, x):
        return self._reduce_to_domain(self.op.get()._adjoint_derivative(self._insert_constants(x)))
    
    def adjoint_data(self, x):
        return self._reduce_to_domain(self.op.get().adjoint_data(x))

    def __repr__(self):
        return util.make_repr(self, self.op.get())


class LinearCombination(Operator):
    r"""A linear combination of operators. This class should normally not be instantiated directly,
    but rather through adding and multiplying `Operator` instances and scalars.
    
    .. code-block::python

        op_composed = a_1 * op_1 + a_2 * op_2 + ... + a_n * op_n

    Parameters
    ----------
    *args : tuple
        Variable number of scalar and operators to be put in a linear combination. Each can be either:
        - A tuple `(scalar, Operator)` representing a scalar and an operator to be combined linearly.
        - An `Operator` to be included directly in the linear combination.
    """

    def __init__(self, *args):
        coeff_for_op = defaultdict(lambda: 0)
        for arg in args:
            if isinstance(arg, tuple):
                coeff, op = arg
            else:
                coeff, op = 1, arg
            assert isinstance(op, Operator), "Given input {} is not an operator please use either [(coeff,operator), ...] or [operator,...]".format(type(op))
            assert np.isscalar(coeff), "coefficient is not a scalar but of type {}".format(type(coeff))
            if isinstance(coeff,complex):
                assert (op.codomain.is_complex), "Complex coefficients can only be used for operators with complex codomains"
            if isinstance(op, type(self)):
                for c, o in zip(op.coeffs, op.ops):
                    coeff_for_op[o] += coeff * c
            else:
                coeff_for_op[op] += coeff
        self.coeffs = []
        """List of coefficients of the combined operators."""
        self.ops = []
        """List of combined operators."""
        for op, coeff in coeff_for_op.items():
            self.coeffs.append(coeff)
            self.ops.append(op)

        domains = [op.domain for op in self.ops if op.domain]
        if domains:
            domain = domains[0]
            assert all(d == domain for d in domains), "All domains have to be the same"
        else:
            domain = None

        codomains = [op.codomain for op in self.ops if op.codomain]
        if codomains:
            codomain = codomains[0]
            assert all(c == codomain for c in codomains), "All codomains have to be the same"
        else:
            codomain = None

        super().__init__(domain, codomain, linear=all(op.linear for op in self.ops))

    def _eval(self, x, differentiate=False, return_adjoint_eval=False):
        y = self.codomain.zeros()
        if differentiate:
            self._derivs = []
        for coeff, op in zip(self.coeffs, self.ops):
            if differentiate:
                tup = op.linearize(x,return_adjoint_eval=return_adjoint_eval)
                z = tup[0]
                self._derivs.append(tup[1])
            else:
                z = op(x)
            y += coeff * z
        return y

    def _derivative(self, x):
        y = self.codomain.zeros()
        for coeff, deriv in zip(self.coeffs, self._derivs):
            y += coeff * deriv(x)
        return y

    def _adjoint(self, y):
        if self.linear:
            ops = self.ops
        else:
            ops = self._derivs
        x = self.domain.zeros()
        for coeff, op in zip(self.coeffs, ops):
            x += coeff.conjugate() * op.adjoint(y)
        return x
    
    def _adjoint_eval(self, x):
        if self.linear:
            if not hasattr(self,"_adjoint_evals"):
                self._adjoint_evals = [op.adjoint_eval for op in self.ops]
            y = self.domain.zeros()
            for coeff, adjoint_eval in zip(self.coeffs, self._adjoint_evals):
                y += abs(coeff)**2 * adjoint_eval(x)
            return y
        else:
            raise RuntimeError(f"Tying to compute an adjoint_eval of a non-linear LinearCombination is not allowed")
    
    def _adjoint_derivative(self, x):
        y = self.domain.zeros()
        for coeff, adjoint_deriv in zip(self.coeffs, self._adjoint_derivs):
            y += abs(coeff)**2 * adjoint_deriv(x)
        return y
    
    def _adjoint_data(self, x):
        y = self.domain.zeros()
        for coeff, op in zip(self.coeffs, self.ops):
            y += abs(coeff)**2 * op._adjoint_data(x)
        return y

    @Operator.inverse.getter
    def inverse(self):
        if self._inverse is not None:
            return self._inverse
        if len(self.ops) > 1:
            raise NotImplementedError(f"The inverse of the linear combination {self} is not defined for operators with.")
        return (1 / self.coeffs[0]) * self.ops[0].inverse

    def __repr__(self):
        return util.make_repr(self, *zip(self.coeffs, self.ops))

    def __str__(self):
        reprs = []
        for coeff, op in zip(self.coeffs, self.ops):
            if coeff == 1:
                reprs.append(repr(op))
            else:
                reprs.append('{} * {}'.format(coeff, op))
        return ' + '.join(reprs)


class Composition(Operator):
    r"""A composition of operators. This class should normally not be instantiated directly,
    but rather through multiplying `Operator` instances.

    .. code-block::python

        op_composed = op_n * ... * op_2 * op_1

    Parameters
    ----------
    *ops : tuple
        Variable number of Operator instanced to be composed. Each Operators domain has to 
        match the next ones codomain. 
    """

    def __init__(self, *ops):
        if not isinstance(ops[0],Operator):
            raise ValueError("The first entry of operators is not an Operator but a {}.".format(type(f)))
        for i,(f, g) in enumerate(zip(ops, ops[1:])):
            if not isinstance(g,Operator):
                raise ValueError( "The {}-th  entry of operators is not an Operator but a {} ".format(i+2,type(g)))  
            if f.domain != g.codomain:
                raise ValueError("The domain of {} and codomain of {} do not match up. \n Domain is \n {} \n Codomain is \n {}".format(f,g, f.domain,g.codomain))
        self.ops = []
        """The list of composed operators."""
        for op in ops:
            if isinstance(op, Composition):
                self.ops.extend(op.ops)
            else:
                self.ops.append(op)
        super().__init__(
            self.ops[-1].domain, self.ops[0].codomain,
            linear=all(op.linear for op in self.ops))

    def _eval(self, x, differentiate=False, return_adjoint_eval = False):
        y = x
        if return_adjoint_eval:
            return self._adjoint_eval(x)
        else:
            if differentiate:
                self._derivs = []
                for op in self.ops[:0:-1]:
                    y, deriv = op.linearize(y)
                    self._derivs.insert(0,deriv)
                tup = self.ops[0].linearize(y)
                y = tup[0]
                self._derivs.insert(0,tup[1])
            else:
                for op in self.ops[::-1]:
                    y = op(y)
            return y

    def _derivative(self, x):
        y = x
        for deriv in self._derivs[::-1]:
            y = deriv(y)
        return y

    def _adjoint(self, y):
        x = y
        if self.linear:
            ops = self.ops
        else:
            ops = self._derivs
        for op in ops:
            x = op.adjoint(x)
        return x
    
    def _adjoint_eval(self, x):
        y = x
        self._derivs = []
        for op in self.ops[:0:-1]:
            y, deriv = op.linearize(y)
            self._derivs.insert(0,deriv)
        tup = self.ops[0].linearize(y,return_adjoint_eval=True)
        y = tup[0]
        self._derivs.insert(0,tup[1])

        for op in self._derivs[1:]:
            y = op.adjoint(y)
        return y
    
    def _adjoint_data(self, data):
        back = self.ops[0]._adjoint_data(data)
        for op in self.ops[1:]:
            back = op.adjoint(back)
        return back
    
    def _adjoint_derivative(self, x):
        y = x
        for deriv in self._derivs[:0:-1]:
            y = deriv(y)
        y = self._derivs[0].adjoint_eval(y)
        for deriv in self._derivs[1:]:
            y = deriv.adjoint(y)
        return y

    @Operator.inverse.getter
    def inverse(self):
        if self._inverse is not None:
            return self._inverse
        try:
            return Composition(*(op.inverse for op in self.ops[::-1]))
        except NotImplementedError:
            raise NotImplementedError("The inverse of the composition {} is not known since one of the operators has not a well defined inverse.".format(self))

    def __repr__(self):
        return util.make_repr(self, *self.ops)



class PartOfOperator(Operator):
    r"""Slcing the output of an operator. Given an operator

    .. math::
        F\colon X \to (Y_1,\dots,Y_n)

    One can slice the operator to a subset of the direct sum of :math:`(Y_1,\dots,Y_n)` 
    by defining an index set :math:`I\subset (1,\dots,n)`.
        
    Parameters
    ----------
    Operator : Operator
        The operator to be sliced. 
    index : int, slice, tuple(int)
        The subset of indices. 
    """
    def __init__(self,base_op,index):
        assert isinstance(base_op.codomain,vecsps.DirectSum)
        self.base_op=base_op
        """The base operator being sliced.
        """
        n_codim = len(base_op.codomain.summands)
        if(isinstance(index,int)):
            assert -n_codim<=index and index<n_codim
            self.index=index
        elif(isinstance(index,slice)):
            assert index.stop is None or -n_codim<=index.stop and index.stop<n_codim
            assert index.start is None or -n_codim<=index.start and index.start<n_codim
            index_list=list(range(n_codim)[index])
            assert len(index_list)>0
            if(len(index_list)==1):
                self.index=index_list[0]
            else:
                self.index=index_list
        elif(isinstance(index,tuple)):
            assert all(isinstance(i,int) for i in index)
            assert -n_codim<=min(index) and max(index)<n_codim
            if(len(index)==1):
                self.index=index[0]
            else:
                self.index=index
        else:
            raise ValueError(f"Invalid type {type(index)} for index")
        if(isinstance(self.index,int)):
            codomain=base_op.codomain.summands[self.index]
        else:
            codomain=vecsps.DirectSum(*[base_op.codomain.summands[i] for i in self.index])
        super().__init__(self.base_op.domain,codomain,linear=self.base_op.linear)

    def _get_codomain_part(self,y):
        if(isinstance(self.index,int)):
            return self.base_op.codomain.split(y)[self.index]
        else:
            y_parts=self.base_op.codomain.split(y)
            return self.codomain.join(*[y_parts[i] for i in self.index])

    def _eval(self, x, differentiate=False):
        if(self.base_op.linear):
            y=self.base_op._eval(x)
        else:
            y=self.base_op._eval(x,differentiate=differentiate)
        return self._get_codomain_part(y)
    
    def _derivative(self, x):
        y=self.base_op._derivative(x)
        return self._get_codomain_part(y)

    def _adjoint(self, y):
        if(isinstance(self.index,int)):
            y_base_op= self.base_op.codomain.zeros()
            y_base_op[self.index] = y
        else:
            y_base_op= self.base_op.codomain.zeros()
            for i,y_i in enumerate(y):
                y_base_op[self.index[i]]+=y_i
        return self.base_op._adjoint(y_base_op)
    
    def __getitem__(self, val):#TODO add checks for ranges
        assert isinstance(self.index,tuple)
        if(isinstance(val,int) or isinstance(val,slice)):
            return PartOfOperator(self.base_op,self.index[val])
        elif(isinstance(val,tuple)):
            return PartOfOperator(self.base_op,tuple(self.index[v] for v in val))


class Pow(Operator):
    r"""Power of a linear operator `A`, mapping a domain into itself, i.e. 
    `A * A * ... * A`

    Parameters
    ----------
    op : Operator
        The Operoter raised to the power of `exponent`
    exponent :  int
        The power. Is required to be a positive interger.
    """
    def __init__(self, op, exponent):
        assert op.linear, "The operator has to be linear."
        assert op.domain == op.codomain, "Domain and codomain have to match."
        assert type(exponent)==int and exponent>=0, "The exponent has to be of int type"
        super().__init__(op.domain,op.domain,linear=True)
        self.op = op
        self.exponent = exponent

    def _eval(self,x):
        res = x
        for j in range(self.exponent):
            res = self.op(res)
        return res

    def _adjoint(self,x):
        res = x
        for j in range(self.exponent):
            res = self.op.adjoint(res)
        return res
    
    @Operator.inverse.getter
    def inverse(self):
        if self._inverse is not None:
            return self._inverse
        try:
            return Pow(self.op.inverse,self.exponent)
        except NotImplementedError:
            raise NotImplementedError("The inverse of the power {} is not known since the operator has not a well defined inverse.".format(self))


class Identity(Operator):
    r"""The identity operator on a vector space. 
    By default, a copy is performed to prevent callers from
    accidentally modifying the argument when modifying the return value.

    Parameters
    ----------
    domain : regpy.vecsps.VectorSpaceBase
        The underlying vector space.
    """

    def __init__(self, domain, copy=True):
        self.copy = copy
        super().__init__(domain, domain, linear=True)

    def _eval(self, x):
        if self.copy:
            return x.copy()
        else:
            return x

    def _adjoint(self, x):
        if self.copy:
            return x.copy()
        else:
            return x
        
    def _adjoint_eval(self, x):
        if self.copy:
            return x.copy()
        else:
            return x

    @Operator.inverse.getter
    def inverse(self):
        return self

    def __repr__(self):
        return util.make_repr(self, self.domain)

class CoordinateProjection(Operator):
    r"""A projection operator onto a subset of the domain. The codomain is a one-dimensional
    `regpy.vecsps.VectorSpaceBase` of the same dtype as the domain.

    Parameters
    ----------
    domain : regpy.vecsps.VectorSpaceBase
        The underlying vector space
    mask : array-like
        Boolean mask of the subset onto which to project.
    """
    def __init__(self, domain, mask):
        if isinstance(domain,vecsps.NumPyVectorSpace):
            mask = np.broadcast_to(mask, domain.shape)
            assert mask.dtype == bool
        else:
            x = domain.rand()
            _ = x[mask]
            x[mask] = domain.ones()[mask]
        self.mask = mask
        super().__init__(
            domain=domain,
            codomain=domain.masked_space(mask),
            linear=True
        )

    def _eval(self, x):
        return x[self.mask]

    def _adjoint(self, x):
        y = self.domain.zeros()
        y[self.mask] = x
        return y
    
    def _adjoint_eval(self, x):
        y = x.copy()
        y[~self.mask] = 0
        return y

    def __repr__(self):
        return util.make_repr(self, self.domain, self.mask)

class CoordinateMask(Operator):
    """A projection operator onto a subset of the domain. The remaining array elements are set to zero.

    Parameters
    ----------
    domain : regpy.vecsps.VectorSpace
        The underlying vector space
    mask : array-like
        Boolean mask of the subset onto which to project.
    """
    def __init__(self, domain, mask):
        self.mask = mask
        super().__init__(
            domain=domain,
            codomain=domain,
            linear=True
        )

    def _eval(self, x):
        res = self.domain.zeros()
        res[self.mask] = x[self.mask]
        return res

    def _adjoint(self, x):
        res = self.domain.zeros()
        res[self.mask] = x[self.mask]
        return res
    
    def _adjoint_eval(self, x):
        res = self.domain.zeros()
        res[self.mask] = x[self.mask]
        return res

    def __repr__(self):
        return util.make_repr(self, self.domain)


class PtwMultiplication(Operator):
    r"""A multiplication operator by a constant factor where each vector entry is multiplied 
    by the vector entry of `factor`. 

    Parameters
    ----------
    domain : regpy.vecsps.VectorSpaceBase
        The underlying vector space
    factor : array-like
        The factor by which to multiply. In case of domain being NumPyVectorSpace it can be anything that can be broadcast to `domain.shape`.
    """
    def __init__(self, domain, factor):
        # Check that factor can broadcast against domain elements without
        # increasing their size.
        if isinstance(domain,vecsps.NumPyVectorSpace):
            factor = np.broadcast_to(factor, domain.shape)
        if domain:
            assert np.isscalar(factor) or factor in domain
        self.factor = factor
        super().__init__(domain, domain, linear=True)

    def _eval(self, x):
        return self.factor * x

    def _adjoint(self, x):
        if self.domain.is_complex:
            return self.factor.conj() * x
        else:
            return self.factor * x

    @Operator.inverse.getter
    def inverse(self):
        if self._inverse is not None:
            return self._inverse
        sav = np.seterr(divide='raise')
        try:
            return PtwMultiplication(self.domain, 1 / self.factor)
        finally:
            np.seterr(**sav)

    def __repr__(self):
        return util.make_repr(self, self.domain)


class OuterShift(Operator):
    r"""Shift an operator by a constant offset in the codomain.

    Parameters
    ----------
    op : Operator
        The underlying operator.
    offset : op.codomain
        The offset by which to shift. 
    """
    def __init__(self, op, offset):
        assert isinstance(op,Operator)
        assert offset in op.codomain or np.isscalar(offset)
        offset = op.codomain.ones()*offset if np.isscalar(offset) else offset
        super().__init__(op.domain, op.codomain)
        if isinstance(op, type(self)):
            offset = offset + op.offset
            op = op.op
        self.op = op
        self.offset = offset.copy()

    def _eval(self, x, differentiate=False, return_adjoint_eval=False):
        if differentiate:
            tup = self.op.linearize(x, return_adjoint_eval= return_adjoint_eval)
            y = tup[0]
            self._deriv = tup[1]
            if not return_adjoint_eval:
                return y + self.offset
            else:
                return y+self._adjoint(self.offset)
        else:
            return self.op(x) + self.offset

    def _derivative(self, x):
        return self._deriv(x)

    def _adjoint(self, y):
        return self._deriv.adjoint(y)
    
    def _adjoint_derivative(self, x):
        return self._deriv.adjoint_eval(x)


class InnerShift(Operator):
    r"""Shift an operator by a constant offset in the domain.

    Parameters
    ----------
    op : Operator
        The underlying operator.
    offset : op.domain
        The offset by which to shift. 
    """
    def __init__(self, op, offset):
        assert offset in op.domain or np.isscalar(offset)
        offset = op.domain.ones()*offset if np.isscalar(offset) else offset
        super().__init__(op.domain, op.codomain)
        if isinstance(op, type(self)):
            offset = offset + op.offset
            op = op.op
        self.op = op
        self.offset = offset.copy()

    def _eval(self, x, differentiate=False, return_adjoint_eval=False):
        if differentiate or return_adjoint_eval:
            y, self._deriv = self.op.linearize(x-self.offset, return_adjoint_eval=return_adjoint_eval)
            return y 
        else:
            return self.op(x - self.offset)

    def _derivative(self, h):
        return self._deriv(h)

    def _adjoint(self, y):
        return self._deriv.adjoint(y)
    
    def _adjoint_derivative(self, x):
        return self._deriv.adjoint_eval(x)

class DirectSum(Operator):
    r"""The direct sum of operators. For

    .. math::
        T_i \colon X_i \to Y_i 

    the direct sum

    .. math::
        T := DirectSum(T_i) \colon DirectSum(X_i) \to DirectSum(Y_i) 

    is given by :math:`T(x)_i := T_i(x_i)`. As a matrix, this is the block-diagonal
    with blocks :math:`(T_i)`.

    Parameters
    ----------
    *ops : tuple(Operator)
        Variable number of Operator instances to be composed to a direct sum.
    flatten : bool, optional
        If True, summands that are themselves direct sums will be merged with
        this one. Default: False.
    domain, codomain : vecsps.VectorSpaceBase or callable, optional
        Either the underlying vector space or a factory function that will be called with all
        summands' vector spaces passed as arguments and should return a vecsps.DirectSum instance.
        The resulting vector space should be iterable, yielding the individual summands.
        Default: vecsps.DirectSum.
    """

    def __init__(self, *ops, flatten=False, domain=None, codomain=None):
        assert all(isinstance(op, Operator) for op in ops)
        self.ops = []
        r""" List of all operators :math:`(T_1,\dots,T_n)`"""
        for op in ops:
            if flatten and isinstance(op, type(self)):
                self.ops.extend(op.ops)
            else:
                self.ops.append(op)
        if isinstance(domain,vecsps.DirectSum):
            if any([d != op.domain for d,op in zip(domain.summands,self.ops)]):
                raise ValueError(f"Was given a DirectSum {domain} whos components do not match with the domain of the operators. \n The csummands of the given domaina are {domain.summands} \n The domain of the operator are {[op.domain for op in self.ops]}")
            else:
                pass
        elif domain is None:
            domain = vecsps.DirectSum(*[op.domain for op in self.ops])
        elif callable(domain):
            domain = domain(*(op.domain for op in self.ops))
            assert isinstance(domain,vecsps.DirectSum) and all([d == op.domain for d,op in zip(domain.summands,self.ops)]), "Domain constructur failed to construct correct domain."
        else:
            raise TypeError('domain={} is neither a VectorSpaceBase nor callable'.format(domain))
        
        if isinstance(codomain,vecsps.DirectSum):
            if any([d != op.codomain for d,op in zip(codomain.summands,self.ops)]):
                raise ValueError(f"Was given a DirectSum {codomain} whos components do not match with the domain of the operators. \n The csummands of the given domaina are {codomain.summands} \n The domain of the operator are {[op.codomain for op in self.ops]}")
            else:
                pass
        elif codomain is None:
            codomain = vecsps.DirectSum(*[op.codomain for op in self.ops])
        elif callable(codomain):
            codomain = codomain(*(op.codomain for op in self.ops))
            assert isinstance(codomain,vecsps.DirectSum) and all([cd == op.codomain for cd,op in zip(codomain.summands,self.ops)]), "Codomain constructur failed to construct correct codomain."
        else:
            raise TypeError('codomain={} is neither a VectorSpaceBase nor callable'.format(codomain))
        
        super().__init__(domain=domain, codomain=codomain, linear=all(op.linear for op in self.ops))

    def _eval(self, x, differentiate=False, return_adjoint_eval=False):
        if hasattr(self,"full_domain"):
            assert x in self.full_domain, "{} is not in the full_domain {}".format(x,type(self.full_domain))
        else:
            assert x in self.domain, "{} is not in domain {}".format(x,type(self.domain))
        if differentiate:
            linearizations = [op.linearize(x_i,return_adjoint_eval=return_adjoint_eval) for op, x_i in zip(self.ops, x)]
            self._derivs = [l[1] for l in linearizations]
            return self.codomain.join(*(l[0] for l in linearizations))
        elif return_adjoint_eval:
            linearizations = [op.linearize(elm,return_adjoint_eval=True) for op, elm in zip(self.ops, x)]
            self._adjoint_derivs = [l[1] for l in linearizations]
            return self.codomain.join(*(l[0] for l in linearizations))
        else:
            return self.codomain.join(*(op(x_i) for op, x_i in zip(self.ops, x)))

    def _derivative(self, x):
        if hasattr(self,"full_domain"):
            assert x in self.full_domain, "{} is not in full_domain {}".format(x,type(self.full_domain))
        else:
            assert x in self.domain, "{} is not in domain {}".format(x,type(self.domain))
        return self.codomain.join(
            *(deriv(x_i) for deriv, x_i in zip(self._derivs, x))
        )

    def _adjoint(self, y):
        assert y in self.codomain, f"{y} is not in codomain {type(self.codomain)} of shape {self.codomain.shape}"
        if self.linear:
            ops = self.ops
        else:
            ops = self._derivs
        if hasattr(self,"full_domain"):
            return self.full_domain.join(
                *(op.adjoint(y_i) for op, y_i in zip(ops, y))
            )
        else:
            return self.domain.join(
                *(op.adjoint(y_i) for op, y_i in zip(ops, y))
            )
    
    def _adjoint_derivative(self, x):
        if hasattr(self,"full_domain"):
            assert x in self.full_domain, "{} is not in full_domain {}".format(x,type(self.full_domain))
        else:
            assert x in self.domain, "{} is not in domain {}".format(x,type(self.domain))
        return self.domain.join(
            *(deriv.adjoint_eval(x_i) for deriv, x_i in zip(self._derivs, x))
        )

    @Operator.inverse.getter
    def inverse(self):
        """The component-wise inverse as a `DirectSum`, if all of them exist."""
        if self._inverse is not None:
            return self._inverse
        try:
            return DirectSum(
                *(op.inverse for op in self.ops),
                domain=self.codomain,
                codomain=self.domain
            )
        except NotImplementedError:
            raise NotImplementedError("The inverse of the direct sum {} is not known since one of the operators has not a well defined inverse.".format(self))

    def __repr__(self):
        return util.make_repr(self, *self.ops)

    def __getitem__(self, item):
        if item is None:
            return self
        return self.ops[item]

    def __iter__(self):
        return iter(self.ops)


class VectorOfOperators(Operator):
    r"""Vector of operators. For

    .. math::
        T_i \colon X \to Y_i

    we define

    .. math::
        T := VectorOfOperators(T_i) \colon X \to DirectSum(Y_i)

    by :math:`T(x)_i := T_i(x)`. 
    
    Parameters
    ----------
    *ops : tuple(Operator)
        Variable number of Operator instances to be put together to a Vector. Each of the Operators
        is required to have the same domain.
    codomain : vecsps.VectorSpaceBase or callable, optional
        Either the underlying vector space or a factory function that will be called with all
        summands' vector spaces passed as arguments and should return a vecsps.DirectSum instance.
        The resulting vector space should be iterable, yielding the individual summands.
        Default: vecsps.DirectSum.
    """

    def __init__(self, ops,  domain=None, codomain=None):
        assert all([isinstance(op, Operator) for op in ops]), "ops must be a list of `Operator` instances"
        assert ops
        self.ops = ops
        r"""List of all Operators :math:`(T_1,\dots,T_n)`"""

        if domain is None:
            self.domain = self.ops[0].domain
        else:
            self.domain = domain
        assert all(op.domain == self.domain for op in self.ops), "All operators in `ops` must have same domain {}".format(type(self.domain))

        if codomain is None:
            codomain = vecsps.DirectSum(*tuple([op.codomain for op in ops]))
        if isinstance(codomain, vecsps.VectorSpaceBase):
            pass
        elif callable(codomain):
            codomain = codomain(*(op.codomain for op in self.ops))
        else:
            raise TypeError('codomain={} is neither a VectorSpaceBase nor callable'.format(codomain))
        assert isinstance(codomain,vecsps.DirectSum), "Codomain must be a `DirectSum`"
        assert all(op.codomain == c for op, c in zip(ops, codomain)), "Codomains of Operators do not match constructed codomain"

        super().__init__(domain=self.domain, codomain=codomain, linear=all(op.linear for op in ops))

    def _eval(self, x, differentiate=False, return_adjoint_eval=False):
        assert x in self.domain, "{} is not in domain {}".format(x,type(self.domain))
        if differentiate:
            linearizations = [op.linearize(x,return_adjoint_eval=return_adjoint_eval) for op in self.ops]
            self._derivs = [l[1] for l in linearizations]
            return self.codomain.join(*(l[0] for l in linearizations))
        else:
            return self.codomain.join(*(op(x) for op in self.ops))

    def _derivative(self, x):
        assert x in self.domain, "{} is not in domain {}".format(x,type(self.domain))
        return self.codomain.join(
            *(deriv(x) for deriv in self._derivs)
        )

    def _adjoint(self, y):
        assert y in self.codomain, "{} is not in codomain {}".format(y,type(self.codomain))
        if self.linear:
            ops = self.ops
        else:
            ops = self._derivs
        result = self.domain.zeros()    
        for op, y_i in zip(ops, y):
            result += op.adjoint(y_i)
        return result
    
    def _adjoint_derivative(self, x):
        assert x in self.domain, "{} is not in domain {}".format(x,type(self.domain))
        result = self.domain.zeros() 
        for deriv in self._derivs:
            result += deriv.adjoint_eval(x)
        return result

    def __repr__(self):
        vec_repr = "[" +", ".join([repr(op) for op in self.ops])+"]"
        return util.make_repr(self, *self.ops)

    def __getitem__(self, item):
        return self.ops[item]

    def __iter__(self):
        return iter(self.ops)


class MatrixOfOperators(Operator):
    r"""Matrix of operators. For

    .. math::
        T_ij \colon X_j \to Y_i

    we define

    .. math::
        T := MatrixOfOperators(T_ij) \colon DirectSum(X_j) \to DirectSum(Y_i)

    by :math:`T(x)_i := \sum_j T_ij(x_j)`. 
    
    Parameters
    ----------
    *ops : tuple(tuple(Operator) 
        Variable number of tuples of Operator instances to build the matrix. Each tuple has to have 
        the same length and a zero operators should be given by None.
    domain, codomain : vecsps.VectorSpaceBase or callable, optional
        Either the underlying vector space or a factory function that will be called with all
        summands' vector spaces passed as arguments and should return a vecsps.DirectSum instance.
        The resulting vector space should be iterable, yielding the individual summands.
        Default: vecsps.DirectSum.
    """

    def __init__(self, ops,  domain=None, codomain=None):
        assert all((isinstance(op_col,list) and len(op_col) == len(ops[0]) for op_col in ops))
        ops_flat = [op for op_col in ops for op in op_col]
        assert all((isinstance(op, Operator) or op==None) for op in ops_flat)
        self.ops = ops
        r""" Matrix of Operators :math:`(T_ij)`"""

        domains = [None]*len(ops)
        for j in range(len(ops)):
            for i in range(len(ops[0])):
                if ops[j][i]:
                    if domains[j]:
                        assert domains[j] == ops[j][i].domain
                    else:    
                        domains[j] = ops[j][i].domain
        assert None not in domains

        if domain is None:
            domain = vecsps.DirectSum
        if isinstance(domain, vecsps.VectorSpaceBase):
            pass
        elif callable(domain):
            domain = domain(*tuple(domains))
        else:
            raise TypeError('domain={} is neither a VectorSpaceBase nor callable'.format(domain))

        codomains = [None]*len(ops[0])
        for i in range(len(ops[0])):
            for j in range(len(ops)):
                if ops[j][i]:
                    if codomains[i]:
                        assert codomains[i] == ops[j][i].codomain
                    else:
                        codomains[i] = ops[j][i].codomain
        assert None not in codomains

        if codomain is None:
            codomain = vecsps.DirectSum
        if isinstance(codomain, vecsps.VectorSpaceBase):
            pass
        elif callable(codomain):
            codomain = codomain(*tuple(codomains))
        else:
            raise TypeError('codomain={} is neither a VectorSpaceBase nor callable'.format(domain))
        
        super().__init__(domain=domain, codomain=codomain, linear=all(op==None or op.linear for op in ops_flat))
        assert isinstance(self.domain,vecsps.DirectSum)
        assert isinstance(self.codomain,vecsps.DirectSum)

    def _eval(self, x, differentiate=False):
        res = self.codomain.zeros()
        Tprime = []
        Tadjprime = []
        for T_j, x_j in zip(self.ops,x):
            Tprime_j = []
            Tadjprime_j = []
            for T_ij,res_i in zip(T_j,res):
                if differentiate:
                    if T_ij:
                        res_deriv,deriv = T_ij.linearize(x_j)
                        res_i += res_deriv
                        Tprime_ij = deriv
                    else:
                        Tprime_ij = None
                        Tadjprime_ij = None
                    Tprime_j.append(Tprime_ij)
                else:   
                    if T_ij:
                        res_i += T_ij(x_j)
            Tprime.append(Tprime_j)
            Tadjprime.append(Tadjprime_j)
        if differentiate:
            self._derivs = Tprime
        return res

    def _derivative(self, x):
        res = self.codomain.zeros()
        for Tprime_j, x_j in zip(self._derivs,x):
            for Tprime_ij,res_i in zip(Tprime_j,res):
                if Tprime_ij:
                    res_i += Tprime_ij(x_j)
        return res

    def _adjoint(self, y):
        if self.linear:
            ops = self.ops
        else:
            ops = self._derivs
        res = self.domain.zeros() 
        for Tprime_j, res_j in zip(ops, res):
            for Tprime_ij, y_i in zip(Tprime_j,y):
                if Tprime_ij:
                    res_j += Tprime_ij.adjoint(y_i)
        return res

    def __repr__(self):
        mat_repr = "[[" +"],\n[".join([", ".join([repr(op) if op else "0" for op in row]) for row in self.ops])+"]"
        return util.make_repr(self, mat_repr)

    def __getitem__(self, item):
        return self.ops[item]

    def __iter__(self):
        return iter(self.ops)

class Sum(Operator):
    r"""Maps element in direct sum of vector spaces to their sum.

    Parameters
    ----------
    domain : vecsps.DirectSum
        The domain of the operator. Summands have to have the same shape.
    codomain : vecsps.VectorSpace or None, optional
        The codomain of the operator. Has to have same shape as a summand of the domain.
        If set to None the first summand of the domain is chosen instead. Defaults to None.
    """

    def __init__(self, domain,codomain=None):
        assert isinstance(domain,vecsps.DirectSum)
        assert isinstance(codomain,vecsps.VectorSpace) or codomain is None
        assert all(domain.summands[0].shape==summand.shape for summand in domain.summands)
        assert codomain is None or domain.summands[0].shape==codomain.shape
        if(codomain is None):
            codomain=domain.summands[0]
        super().__init__(domain, codomain, True)

    def _eval(self,x):
        return sum(self.domain.split(x))
    
    def _adjoint(self,y):
        return self.domain.join(*[np.real(y) if summand.dtype==float else y for summand in self.domain.summands])
    
class Product(Operator):
    r"""Maps element in direct sum of vector spaces to their product.

    Parameters
    ----------
    domain : vecsps.DirectSum
        The domain of the operator. Summands have to have the same shape.
    codomain : vecsps.VectorSpace or None, optional
        The codomain of the operator. Has to have same shape as a summand of the domain.
        If set to None the first summand of the domain is chosen instead. Defaults to None.
    """

    def __init__(self, domain,codomain=None):
        assert isinstance(domain,vecsps.DirectSum)
        assert isinstance(codomain,vecsps.VectorSpace) or codomain is None
        assert all(domain.summands[0].shape==summand.shape for summand in domain.summands)
        assert codomain is None or domain.summands[0].shape==codomain.shape
        if(codomain is None):
            codomain=domain.summands[0]
        super().__init__(domain, codomain, False)

    def _eval(self,x,differentiate=False):
        x_split=self.domain.split(x)
        y=x_split[0].copy()
        for i in range(1,len(x_split)):
            y*=x_split[i]
        if(differentiate):
            self.deriv_data=[y/x_j for x_j in x_split]
        return y
    
    def _derivative(self, x):
        x_split=self.domain.split(x)
        y=self.deriv_data[0]*x_split[0]
        for i in range(1,len(x_split)):
            y+=self.deriv_data[i]*x_split[i]
        return y

    def _adjoint(self,y):
        y_parts=[np.real(y*np.conj(self.deriv_data[i])) if summand.dtype==float else y*np.conj(self.deriv_data[i]) for i,summand in enumerate(self.domain.summands)]
        return self.domain.join(*y_parts)


class RealPart(Operator):
    r"""The pointwise real part operator.

    Parameters
    ----------
    domain : regpy.vecsps.VectorSpaceBase
        The underlying vector space. The codomain will be the corresponding
        `regpy.vecsps.VectorSpaceBase.real_space`.
    """

    def __init__(self, domain):
        if domain:
            codomain = domain.real_space()
        else:
            codomain = None
        super().__init__(domain, codomain, linear=True)

    def _eval(self, x):
        return x.real.copy()

    def _adjoint(self, y):
        return y.copy()


class ImaginaryPart(Operator):
    r"""The pointwise imaginary part operator.

    Parameters
    ----------
    domain : regpy.vecsps.VectorSpaceBase
        The underlying vector space. The codomain will be the corresponding
        `regpy.vecsps.VectorSpaceBase.real_space`.
    """

    def __init__(self, domain):
        if domain:
            assert domain.is_complex
            codomain = domain.real_space()
        else:
            codomain = None
        super().__init__(domain, codomain, linear=True)

    def _eval(self, x):
        return x.imag.copy()

    def _adjoint(self, y):
        return 1j * y


class SquaredModulus(Operator):
    r"""The pointwise squared modulus operator.

    Parameters
    ----------
    domain : regpy.vecsps.VectorSpaceBase
        The underlying vector space. The codomain will be the corresponding
        `regpy.vecsps.VectorSpaceBase.real_space`.
    """

    def __init__(self, domain):
        if domain:
            codomain = domain.real_space()
        else:
            codomain = None
        super().__init__(domain, codomain)

    def _eval(self, x, differentiate=False, return_adjoint_eval=False):
        if differentiate or return_adjoint_eval:
            self._factor = 2 * x
        return x.real**2 + x.imag**2

    def _derivative(self, h):
        return (self._factor.conj() * h).real

    def _adjoint(self, y):
        return self._factor * y


class Zero(Operator):
    r"""The constant zero operator.

    Parameters
    ----------
    domain : regpy.vecsps.VectorSpaceBase
        The underlying vector space.
    codomain : regpy.vecsps.VectorSpaceBase, optional
        The vector space of the codomain. Defaults to `domain`.
    """
    def __init__(self, domain, codomain=None):
        if codomain is None:
            codomain = domain
        super().__init__(domain, codomain, linear=True)

    def _eval(self, x):
        return self.codomain.zeros()

    def _adjoint(self, x):
        return self.domain.zeros()


class ApproximateHessian(Operator):
    r"""An approximation of the Hessian of a `regpy.functionals.Functional` at some point, computed
    using finite differences of it `gradient` if it is implemented for that functional.

    Parameters
    ----------
    func : regpy.functionals.Functional
        The functional.
    x : array-like
        The point at which to evaluate the Hessian.
    stepsize : float, optional
        The stepsize for the finite difference approximation.
    """
    def __init__(self, func, x, stepsize=1e-8):
        from regpy.functionals import Functional
        assert isinstance(func, Functional)
        assert hasattr(func,"gradient")
        self.gradx = func.gradient(x)
        """The gradient at `x`"""
        self.func = func
        self.x = x.copy()
        self.stepsize = stepsize
        # linear=True is a necessary lie
        super().__init__(func.domain, func.domain, linear=True)
        self.log.info('Using approximate Hessian of functional {}'.format(self.func))

    def _eval(self, h):
        grad = self.func.gradient(self.x + self.stepsize * h)
        return grad - self.gradx

    def _adjoint(self, x):
        return self._eval(x)


class SciPyLinearOperator(LinearOperator):
    r"""A class wrapping a linear operator \(F\) into a scipy.sparse.linalg.LinearOperator so that it can be used conveniently in scipy methods.
    The domain and codomain are flattened.

    Parameters
    ----------
    op2 : Operator
        The operator to be put into a scipy.linalg.LinearOperator. 
    """
    def __init__(self, op2):
        self.op2 = op2
        r"""the wrapped operator"""
        domain_shape=op2.domain.realsize
        codomain_shape=op2.codomain.realsize
        super().__init__(np.float64, (codomain_shape,domain_shape))

    
    def _matvec(self, x):
        r"""Applies the operator.
        
        Parameters
        ----------
        x : numpy.ndarray
            Flattened element from domain of operator.
        
        Returns
        -------
        numpy.ndarray
        """
        op2 = self.op2
        return op2.codomain.flatten(op2(op2.domain.fromflat(x)))
    
    def _rmatvec(self, y):
        r"""Applies the adjoint operator.
        
        Parameters
        ----------
        y : numpy.ndarray
            Flattened element from codomain of operator.
        
        Returns
        -------
        numpy.ndarray
        """
        op2 = self.op2
        return op2.domain.flatten(op2.adjoint(op2.codomain.fromflat(y)))

