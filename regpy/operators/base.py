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

__all__ = ["Operator", "Pow", "Identity", "CoordinateProjection", "CoordinateMask", "PtwMultiplication", "OuterShift", "InnerShift", "DirectSum", "VectorOfOperators", "MatrixOfOperators", "Sum","Product", "RealPart", "ImaginaryPart", "SplitRealImag","SquaredModulus", "Zero", "ApproximateHessian", "SciPyLinearOperator"]


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
            raise RuntimeError(util.Errors._compose_message("REVOKED",
                'Attempted to use revoked reference')) from None

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
    r"""Base class for (forward) operators. Both linear and non-linear operators are handled. Operator
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
     * Many expensive computations, e.g. assembly and factorization of finite element matrices, need to be carried
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
        The vector space on which the operator's arguments / values are defined. Using `None`
        suppresses some consistency checks and is intended for ease of development, but should 
        not be used except as a temporary measure. Some constructions like direct sums will fail
        if the vector spaces are unknown.
    linear : bool, optional
        Whether the operator is linear. Default: `False`.
    """

    log = util.ClassLogger()

    def __init__(self, domain=None, codomain=None, linear=False, inverse=None):
        if not isinstance(domain,vecsps.VectorSpaceBase):
            raise ValueError(util.Errors.not_a_vecsp(domain,vecsps.VectorSpaceBase,"The domain of an operator has to be some derivative of VectorSpaceBases."))
        if not isinstance(codomain,vecsps.VectorSpaceBase):
            raise ValueError(util.Errors.not_a_vecsp(codomain,vecsps.VectorSpaceBase,"The codomain of an operator has to be some derivative of VectorSpaceBases."))
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
            raise TypeError(util.Errors._compose_message("INVERSE NOT EXISTENT","The inverse has to be an Operator instance or None."))
        self._inverse = inverse

        self.reset_detect_loop()

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
    
    def reset_detect_loop(self):
        self._detect_loop = [False]*4    

    def __call__(self, x, out = None, **kwargs):
        if x not in self.domain and (not hasattr(self, "full_domain") or x not in self.full_domain):
            raise ValueError(util.Errors.not_in_vecsp(
                x,
                self.domain,
                add_info=f"Evaluation of {self} not possible! The vector has to be either in domain or full domain."
                ))
        if out is None:
            if self.linear:
                y = self._eval(self._insert_constants(x),**kwargs)
            else:
                self.__revoke()
                y = self._eval(self._insert_constants(x), differentiate=False, **kwargs)
            if y not in self.codomain:
                raise RuntimeError(util.Errors.not_in_vecsp(
                    y,
                    self.codomain,
                    add_info=f" Evaluation of {self} went wrong! Please analyse your evaluation method _eval it does not return a proper element in the codomain."
                    ))
            self._detect_loop = [False]*4
            return y
        elif out in self.codomain:
            if self.linear:
                out = self._ieval(self._insert_constants(x), out,**kwargs)
            else:
                self.__revoke()
                out = self._ieval(self._insert_constants(x), out, differentiate=False, **kwargs)
            self.reset_detect_loop()
            return out
        else:
            raise ValueError(util.Errors.not_in_vecsp(
                out,
                self.codomain,
                add_info=f"Given output does not belong to codomain evaluation of {self} not possible!"
                ))
            

    def linearize(self, x, out = None, return_adjoint_eval = False, **kwargs):
        r"""Linearize the operator around some point.

        Parameters
        ----------
        x : array-like
            The point around which to linearize.

        return_adjoint_eval : boolean (Default: False)
            Flag to determine if the adjoint of the evaluation should be returned. That is the 
            first output will be :math:`F'[x]^\ast F(x)` rather then F(x). In particular, if the image 
            space of the operator is too large to store vectors in this space the operator can have 
            an :meth:`_adjoint_eval` that has an efficient implementation for that and 
            additionally a :meth:`_adjoint_derivative` that efficiently implements :math:`F'[x]^\ast F'[x]` .

        Returns
        -------
        if return_adjoint_eval==False: 
          array, Derivative:
             The value and the derivative at `x`, the latter as an `Operator` instance.
        if return_adjoint_eval ==True: 
           array, Derivative
               array is :math:`F'[x]^\ast F(x)`, Derivative is as above. The adjoint derivative
               that is an efficient implementation of the composition Derivative.adjoint * 
               Derivative is accessible by Derivative.adjoint_eval given an AdjointEval instance.
        """
        if x not in self.domain and (not hasattr(self, "full_domain") or x not in self.full_domain):
            raise ValueError(util.Errors.not_in_vecsp(
                x,
                self.domain,
                add_info=f"Evaluation of {self} not possible! The vector has to be either in domain or full domain."
                ))
        if out is None or (isinstance(out,list) and out[0] is None):
            if self.linear:
                if not return_adjoint_eval:
                    return self(x, **kwargs), self
                else:
                    return self.adjoint_eval(x, **kwargs), self
            else:
                self.__revoke()
                if not return_adjoint_eval:
                    y = self._eval(self._insert_constants(x), differentiate=True, **kwargs)
                    if y not in self.codomain:
                        raise RuntimeError(util.Errors.not_in_vecsp(
                            y,
                            self.codomain
                            ))
                    deriv = Derivative(self.__get_handle())
                    self.reset_detect_loop()
                    return y, deriv
                else:
                    Fstar_y = self._adjoint_eval(self._insert_constants(x),**kwargs)
                    if Fstar_y not in self.domain: 
                        raise RuntimeError(util.Errors.not_in_vecsp(
                            Fstar_y,
                            self.domain
                        ))
                    deriv = Derivative(self.__get_handle()) 
                    self.reset_detect_loop()
                    return Fstar_y, deriv
        else:
            if not isinstance(out,list):
                out = [out, None]
            if return_adjoint_eval and out[0] not in self.domain:
                raise ValueError(util.Errors.value_error(
                    f"The output must be a list of two elements [vec,any] or just vec where vec belongs to the domain if linearize with return_adjoint_eval = True. Evaluation of {self} not possible! You gave "+"\n\t "+f"out = {out}"
                    ))            
            if not return_adjoint_eval and out[0] not in self.codomain:
                raise ValueError(util.Errors.value_error(
                    f"The output must be a list of two elements [vec,any] or just vec where vec belongs to the codomain. \n Evaluation of {self} not possible! You gave:"+"\n\t "+f"out = {out}"
                    ))
            if self.linear:
                # Out in domain or codomain is validated in __call__ of self or adjoint_eval
                if not return_adjoint_eval:
                    out[0] = self(x, out = out[0], **kwargs)
                    out[1] = self
                else:
                    out[0] = self.adjoint_eval(x, out = out[0], **kwargs)
                    out[1] = self
            else:
                self.__revoke()
                if not return_adjoint_eval:
                    out[0] = self._ieval(self._insert_constants(x), out = out[0], differentiate=True, **kwargs)
                    out[1] = Derivative(self.__get_handle())
                else:
                    self._iadjoint_eval(self._insert_constants(x), out = out[0],**kwargs)
                    out[1] = Derivative(self.__get_handle()) 
            self.reset_detect_loop()
            return out[0],out[1]

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
            raise RuntimeError(util.Errors.not_linear_op(
                self,
                "To construct an adjoint the Operator has to be linear!"
                ))
        return Adjoint(self)
    
    @util.memoized_property
    def adjoint_eval(self):
        r"""This is only available for linear operators, it is the composition of adjoint and 
        eval as a `regpy.operators.AdjointEval` instance. Will only be computed on
        demand and saved for subsequent invocations.

        Returns
        -------
        AdjointEval
            The combination of the adjoint with the forward model T*T as an `Operator` instance.
        """
        if not self.linear:
            raise RuntimeError(util.Errors.not_linear_op(
                self,
                "To construct an adjoint the Operator has to be linear!"
                ))
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

    def _eval(self, x, **kwargs):
        self.log.debug(f"eval: {self._detect_loop}")
        if self._detect_loop[0]:
            raise NotImplementedError(util.Errors._compose_message(
                "NOT DEFINED METHOD",
                "By default the method _eval or _ieval are not implemented for an Operator!\n You as a user has to define it!"
            ))
        self._detect_loop[0] = True
        out = self.codomain.zeros()
        return self._ieval(x, out, **kwargs)
    
    def _ieval(self, x, out, **kwargs):
        r"""In place/ out evaluation. A method that ideally provides an in place evaluation into the specified out
        value. The default implementation uses _eval. Note If eval is equally not defined the it will raise an Error.

        Parameters
        ----------
        x : array-like
            The point at which to evaluate in the domain.
        out : array-like
            The output vector in which to write the result.
        differentiate : bool, optional
            Determines if precomputation for derivate is needed, by default False
        """
        self.log.debug(f"eval: {self._detect_loop}")
        if self._detect_loop[0]:
            raise NotImplementedError(util.Errors._compose_message(
                "NOT DEFINED METHOD",
                "By default the method _eval or _ieval are not implemented for an Operator!\n You as a user has to define it!"
            ))
        self._detect_loop[0] = True
        if out is x:
            x = x.copy()
        out *= 0
        out += self._eval(x, **kwargs)
        return out


    def _derivative(self, x, **kwargs):
        if self._detect_loop[1]:
            raise NotImplementedError(util.Errors._compose_message(
                "NOT DEFINED METHOD",
                "By default the method _derivative or _iderivative are not implemented for an Operator!\n You as a user has to define it!"
            ))
        self.log.debug(f"deriv: {self._detect_loop}")
        self._detect_loop[1] = True
        out = self.codomain.zeros()
        return self._iderivative(x, out, **kwargs)
    
    def _iderivative(self, x, out, **kwargs):
        r"""In place/ out evaluation of derivative. A method that ideally provides an in place evaluation into the 
        specified out value of the derivative. The default implementation uses _derivative. Note If _derivative is 
        equally not defined the it will raise an Error.

        Parameters
        ----------
        x : array-like
            The point at which to evaluate in the domain.
        out : array-like
            The output vector in which to write the result.
        differentiate : bool, optional
            Determines if precomputation for derivate is needed, by default False
        """
        if self._detect_loop[1]:
            raise NotImplementedError(util.Errors._compose_message(
                "NOT DEFINED METHOD",
                "By default the method _derivative or _iderivative are not implemented for an Operator!\n You as a user has to define it!"
            ))
        self._detect_loop[1] = True
        if out is x:
            x = x.copy()
        out *= 0
        out += self._derivative(x, **kwargs)
        return out

    def _adjoint(self, y, **kwargs):
        if self._detect_loop[2]:
            raise NotImplementedError(util.Errors._compose_message(
                "NOT DEFINED METHOD",
                "By default the method _adjoint or _iadjoint are not implemented for an Operator!\n You as a user has to define it!"
            ))
        self._detect_loop[2] = True
        out = self.domain.zeros()
        return self._iadjoint(y, out, **kwargs)
    
    def _iadjoint(self,y, out, **kwargs):
        if self._detect_loop[2]:
            raise NotImplementedError(util.Errors._compose_message(
                "NOT DEFINED METHOD",
                "By default the method _adjoint or _iadjoint are not implemented for an Operator!\n You as a user has to define it!"
            ))
        self._detect_loop[2] = True
        if out is y:
            y = y.copy()
        out *= 0
        out += self._adjoint(y, **kwargs)
        return out

    def _adjoint_data(self, data, **kwargs):
        if self._detect_loop[3]:
            raise NotImplementedError(util.Errors._compose_message(
                "NOT DEFINED METHOD",
                "By default the method _adjoint_data or _iadjoint_data are not implemented for an Operator!\n You as a user has to define it!"
            ))
        self.log.debug(f"adjdata: {self._detect_loop}")
        self._detect_loop[3] = True
        out = self.domain.zeros()
        return self._iadjoint_data(data, out, **kwargs)
    
    def _iadjoint_data(self, data, out, **kwargs):
        if self._detect_loop[3]:
            raise NotImplementedError(util.Errors._compose_message(
                "NOT DEFINED METHOD",
                "By default the method _adjoint_data or _iadjoint_data are not implemented for an Operator!\n You as a user has to define it!"
            ))
        self.log.debug(f"adjdata: {self._detect_loop}")
        self._detect_loop[3] = True
        if out is data:
            data = data.copy()
        out *= 0
        out += self._adjoint_data(data, **kwargs)
        return out
        
    def adjoint_data(self, data, out = None, **kwargs):
        if out is None:
            try:
                res = self._adjoint_data(data, **kwargs)
            except NotImplementedError:
                res = self._adjoint(data,**kwargs)
            return res
        else:
            if out not in self.domain:
                raise ValueError(util.Errors.value_error(
                f"The output must be a an element of the domain when calling constructing adjoint data. You gave:"+"\n\t"+f"out = {out}"
                )) 
            try:
                return self._iadjoint_data(data, out, **kwargs)
            except NotImplementedError:
                return self._iadjoint(data, out,**kwargs)
    
    def _adjoint_eval(self, x, **kwargs):
        self.log.warning("The default implementation of _adjoint_eval is not optimised to exclude constructing elements in the codomain!")
        if len(kwargs) != 0:
            self.log.warning(f"In the default implementation of _adjoint_eval it is unclear where to put keyword arguments. Maybe implement your own _adjoint_eval and process the kwargs Thus ignoring yours: kwargs= {kwargs}")
        if self.linear:
            return self.adjoint(self(x))
        else:
            y,deriv = self.linearize(x)
            return deriv.adjoint(y)
    
    def _iadjoint_eval(self, x, out, **kwargs):
        self.log.warning("The default implementation of _iadjoint_eval is not optimised to exclude constructing elements in the codomain!")
        if len(kwargs) != 0:
            self.log.warning(f"In the default implementation of _adjoint_eval it is unclear where to put keyword arguments. Maybe implement your own _adjoint_eval and process the kwargs Thus ignoring yours: kwargs= {kwargs}")
        if self.linear:
            return self.adjoint(self(x), out = out)
        else:
            y, deriv = self.linearize(x)
            return deriv.adjoint(y, out = out)
            

    def _adjoint_derivative(self, x, **kwargs):
        self.log.warning(f"The default implementation of _adjoint_derivative is not optimised to exclude constructing elements in the codomain!")
        if len(kwargs) != 0:
            self.log.warning(f"In the default implementation of _adjoint_derivative it is unclear where to put keyword arguments.Maybe implement your own _adjoint_derivative and process the kwargs Thus ignoring yours: kwargs= {kwargs}")
        if self.linear:
            return self._adjoint(self._eval(x))
        else:
            return self._adjoint(self._derivative(x))
        
    def _iadjoint_derivative(self, x, out, **kwargs):
        self.log.warning(f"The default implementation of _adjoint_derivative is not optimised to exclude constructing elements in the codomain!")
        if len(kwargs) != 0:
            self.log.warning(f"In the default implementation of _adjoint_derivative it is unclear where to put keyword arguments.Maybe implement your own _adjoint_derivative and process the kwargs Thus ignoring yours: kwargs= {kwargs}")
        if self.linear:
            return self._iadjoint(self._eval(x), out)
        else:
            return self._iadjoint(self._derivative(x), out)

    @property
    def inverse(self):
        r"""A property containing the  inverse as an `Operator` instance. In most cases this will
        just raise a `NotImplementedError`, but subclasses may override this if possible and useful.
        To avoid recomputing the inverse on every access, `regpy.util.memoized_property` may be
        useful."""
        if self._inverse is None:
            raise NotImplementedError(util.Errors._compose_message(
                "NOT EXISTENT INVERSE",
                "The inverse of the operator {} is not known.".format(self)))
        return self._inverse
    
    @inverse.setter
    def inverse(self, inv):
        if inv is None:
            self._inverse = None
            self.log.info("Setting the inverse of the operator {} to None".format(self))
        elif not isinstance(inv, Operator):
            raise TypeError(util.Errors.not_instance(
                inv,
                Operator,
                "The inverse has to be an Operator instance."
                ))
        self.log.info("Setting the inverse of the operator {} to {} overwriting the old {}.".format(self,inv,self._inverse))
        self._inverse = inv

    @property
    def invertible(self):
        return self._inverse is not None

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
            raise RuntimeError(util.Errors.not_linear_op(
                self,
                "To construct an linear SciPy operator the Operator has to be linear!"
                ))
        
    def norm(self,h_domain=None,h_codomain=None,method=None,without_codomain_vectors=False):
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
        without_codomain_vectors: bool [default: False]
            If true, the method avoids any use of vectors in the image space of the operator by using 
            adjoint_derivative. (Useful if the latter method in more efficient than the composition of
            adjoint and derivative or if vectors in the image space are too large to be stored.)
        Returns
        -------
        scalar
            Approximation of the norm of T^*T. 

        Raises
        ------
        NotImplementedError
            If the operator is non-linear or the method is not implemented.
        """

        if(not self.linear):
            raise NotImplementedError(util.Errors.not_linear_op(
                self,
                f"To compute the norm of the Operator {self} it has to be linear!"
                ))
        from regpy.hilbert import L2
        if(h_domain is None):
            h_domain=L2(self.domain)
        elif h_domain.vecsp != self.domain:
            raise ValueError(util.Errors.not_equal(
                h_domain.vecsp,
                self.domain,
                add_info=f"Trying to compute the norm of the operator {self} \n with a given Hilbert space {h_domain} on domain."))
        if(h_codomain is None):
            h_codomain=L2(self.codomain)
        elif not without_codomain_vectors and h_codomain.vecsp != self.codomain:
            raise ValueError(util.Errors.not_equal(
                h_domain.vecsp,
                self.domain,
                add_info=f"Trying to compute the norm of the operator {self} with a given Hilbert space {h_domain} on codomain."))
        method=getattr(self,'default_norm_method','lanczos') if method is None else method
        if method == "power":
            return self._power_method(h_domain,h_codomain,without_codomain_vectors=without_codomain_vectors)
        elif method == "lanczos":
            from scipy.sparse.linalg import eigsh
            if without_codomain_vectors:
                op = self.adjoint_eval
            else:
                op = self.adjoint * h_codomain.gram * self
            # eigsh fails if the operator has a null spaces. That's why we compute the largest eigenvalue op+h_domain_gram
            return sqrt(eigsh(SciPyLinearOperator(op+h_domain.gram), 1, M=SciPyLinearOperator(h_domain.gram),
                              Minv=SciPyLinearOperator(h_domain.gram_inv),
                              tol=0.01)[0][0]-1.)
        else:
            raise NotImplementedError(util.Errors._compose_message(
                "NOT DEFINED METHOD",
                f"The method {method} is unknown to compute the norm of an Operator {self}!"
            ))

    def _power_method(self,h_domain,h_codomain,max_iter=int(1e2),stopping_rule=1e-12,without_codomain_vectors = False):
        r"""Approximation of operator norm by the power method. Should not be used directly and only be called via norm.

        Parameters
        ----------
        h_domain: Hilbert space on the domain.
        h_codomain: Hilbert space on the codomain.
        max_iter: int maximum number of iterations
        stopping_rule: float Iteration is stopped if relative residual is smaller than this value.
        without_codomain_vectors: bool [default: False]
            see method norm        
        """
        x = self.domain.rand()
        relative_residual = inf
        if without_codomain_vectors:
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

        Moreover, it validates if the remaining operator is linear using the utility method
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
                raise TypeError(util.Errors.not_instance(
                    self.domain,
                    vecsps.DirectSum,
                    "Setting constants for an Operator is only allowed if that Operator has a domain that is a DirectSum."
                    ))
            self.full_domain = deepcopy(self.domain)
        
        if not isinstance(index,int):
            raise TypeError(util.Errors.indexation(
                index,
                self,
                "The index has to be an integer!"
            ))
        if index<0 or index>=len(self.full_domain):
            raise IndexError(util.Errors.indexation(
                index,
                self,
                f"The used index is out of range 0, ..., {len(self.full_domain)}."
                ))
        if len(set(range(len(self.full_domain)))-self._constants.keys()-{index}) == 0:
            raise ValueError(util.Errors.indexation(
                index,
                self,
                "By setting the index there is no input remaining please choose another index or release some other constant."
                ))
        if c in self.full_domain[index]:
            pass
        elif isinstance(c,int) or isinstance(c,float) or (isinstance(c,complex) and self.full_domain[index].is_complex):
            c = c*self.full_domain[index].ones()
        else:
            raise ValueError(util.Errors.not_in_vecsp(
                c,
                self.full_domain[index],
                vec_name="constant",
                space_name="full domain",
                add_info= f"The given constant is not in the component of the full domain of index={index}"
                ))
        
        self._constants[index] = c
        self.domain = vecsps.DirectSum(*[d_i for i,d_i in enumerate(self.full_domain) if i not in self._constants.keys()])
        if len(self.domain) == 1:
            self.domain = self.domain[0]

        del self.adjoint
        del self.adjoint_eval
        
        if not self.linear:
            self.linear = util.operator_tests.test_linearity(self)
    
    def reset_constants(self):
        """Resets the constants set by `set_constant` to an empty dictionary. This will
        also reset the domain to the full domain.
        """
        if hasattr(self, "full_domain"):
            try:
                del self.adjoint
                del self.adjoint_eval
                self._constants = {}
                self.domain = self.full_domain
            except Exception as e:
                raise RuntimeError(util.Errors._compose_message(
                    "ERROR WHILE RESETTING OPERATOR CONSTANTS",
                    f"Something went wrong while resetting. Cannot reset constants! Got an exception:"+"\n\t"+ f"{e}"))
            self.linear = util.operator_tests.test_linearity(self)
        elif len(self._constants)!=0:
            raise RuntimeError(util.Errors._compose_message(
                    "ERROR WHILE RESETTING OPERATOR CONSTANTS",f"Something went wrong while resetting. Their exists constants {self._constants} but no full_domain"))
        else:
            self.log.warning("Resetting constants while non are specified might not be necessary!")

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
        if hasattr(self, "full_domain") and len(self._constants)>0:
            if x in self.domain:
                x_full_split = self.full_domain.zeros()
                if isinstance(self.domain,vecsps.DirectSum):
                    x_split = x
                else:
                    x_split = [x]
                for ind, constant in self._constants.items():
                    x_full_split[ind] = constant
                other_inds = list(set(range(self.full_domain.n_components))-self._constants.keys())
                other_inds.sort()
                x_full_split[other_inds] = x_split
                return x_full_split
            elif x in self.full_domain:
                for ind, constant in self._constants.items():
                    x[ind] = constant
                return x
            else:
                raise RuntimeError(util.Errors.runtime_error(f"Trying to insert constants failed.x = {x} has to belong either to the domain {self.domain} or the full_domain {self.full_domain}."))
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
            if y in self.full_domain:
                y_full_split = self.full_domain.split(y)
                if isinstance(self.domain,vecsps.DirectSum):
                    return self.domain.join(*[y_full_split[i] for i in set(range(len(self.full_domain)))-self._constants.keys()])
                else:
                    return [y_full_split[i] for i in set(range(len(self.full_domain)))-self._constants.keys()][0]
            elif y in self.domain:
                return y
            else:
                raise RuntimeError(util.Errors.runtime_error(
                    f"The vector supposed to be reduced to the domain is not in the full domain or already in the domain."
                ))
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
        elif np.isscalar(other) or other in self.codomain:
            return OuterShift(self, other)
        elif isinstance(other, Operator):
            return LinearCombination(self, other)
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
        if not isinstance(op,Operator):
            raise ValueError(util.Errors.not_instance(
                op,
                Operator,
                "The Adjoint can only be constructed for a regpy Operator!"
            ))
        if not op.linear:
            raise ValueError(util.Errors.not_linear_op(
                op,
                "An adjoint operator can only be constructed from a linear operator. Please use linearize to get the derivative and use its adjoint!"))
        self.op = op
        r"""The underlying operator."""
        super().__init__(op.codomain, op.domain, linear=True)
        if hasattr(self.op,"full_domain"):
            self._constants = {}
            """The constant inputs of the op need to be set in adjoint evaluation
            """
            self.full_domain = op.codomain

    def _eval(self, x, **kwargs):
        self.op.reset_detect_loop()
        return self.op._reduce_to_domain(self.op._adjoint(x, **kwargs))
        
    def _ieval(self,x, out,**kwargs):
        self.op.reset_detect_loop()
        return self.op._reduce_to_domain(self.op._iadjoint(x, out, **kwargs))

    def _adjoint(self, x, **kwargs):
        self.op.reset_detect_loop()
        return self.op._eval(self._insert_constants(x), **kwargs)

    def _iadjoint(self, x, out, **kwargs):
        self.op.reset_detect_loop()
        return self.op._ieval(self._insert_constants(x), out, **kwargs)

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
            raise NotImplementedError(util.Errors._compose_message(
                    "INVERSE NOT DEFINED",
                    "The inverse of the adjoint of operator {} is not known.".format(self.op)))

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

    def _eval(self, x, **kwargs):
        self.op.get().reset_detect_loop()
        return self.op.get()._derivative(x, **kwargs)

    def _ieval(self, x, out, **kwargs):
        self.op.get().reset_detect_loop()
        return self.op.get()._iderivative(x, out, **kwargs)

    def _adjoint(self, x, **kwargs):
        self.op.get().reset_detect_loop()
        return self._reduce_to_domain(self.op.get()._adjoint(x, **kwargs))

    def _iadjoint(self, x, out, **kwargs):
        self.op.get().reset_detect_loop()
        return self._reduce_to_domain(self.op.get()._iadjoint(x, out, **kwargs))
    
    def adjoint_data(self, x, out = None, **kwargs):
        if out is None:
            return self._reduce_to_domain(self.op.get().adjoint_data(x, **kwargs))
        else:
            return self._reduce_to_domain(self.op.get().adjoint_data(x, out = out, **kwargs))
    
    def _adjoint_eval(self, x, **kwargs):
        self.op.get().reset_detect_loop()
        return self._reduce_to_domain(self.op.get()._adjoint_derivative(x, **kwargs))

    def _iadjoint_eval(self, x, out, **kwargs):
        self.op.get().reset_detect_loop()
        return self._reduce_to_domain(self.op.get()._iadjoint_derivative(x, out, **kwargs))

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
            raise TypeError(util.Errors.not_instance(
                op,
                Operator,
                add_info="The input has to be an Operator instance."
            ))
        if not op.linear:
            raise RuntimeError(util.Errors.not_linear_op(
                op,
                add_info='Operator is not linear cannot create AdjointEval.'
            ))
        self.op = op
        super().__init__(op.domain, op.domain, linear=True)
        # Setting the corresponding constants of op to zero
        if hasattr(self.op,"full_domain"):
            self.full_domain = self.op.full_domain
            self._constants = {index : self.full_domain[index].zeros() for index in self.op._constants}

    def _eval(self, x, **kwargs):
        self.op.reset_detect_loop()
        return self._reduce_to_domain(self.op._adjoint_eval(x, **kwargs))

    def _ieval(self, x, out, **kwargs):
        self.op.reset_detect_loop()
        return self._reduce_to_domain(self.op._iadjoint_eval(x, out, **kwargs))

    def _adjoint(self, x, **kwargs):
        return self._eval(x, **kwargs)

    def _iadjoint(self, x, out, **kwargs):
        return self._ieval(x, out, **kwargs)


    def adjoint_data(self, x, out = None, **kwargs):
        if out is None:
            return self._reduce_to_domain(self.op.adjoint_data(x, **kwargs))
        else:
            return self._reduce_to_domain(self.op.adjoint_data(x, out = out, **kwargs))

    def __repr__(self):
        return util.make_repr(self, self.op)


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
            if not isinstance(op,Operator):
                raise ValueError(util.Errors.not_instance(
                    op,
                    Operator,
                    "The LinearCombination can only be constructed for inputs given by \n w[(coeff,Operator), ...] or [Operator,...]. "
                ))
            if not np.isscalar(coeff):
                raise ValueError(util.Errors.not_instance(
                    coeff,
                    np.ScalarType,
                    "The coefficients in [(coeff,Operator), ...] to construct a LinearCombination \n have to be scalars! "
                ))

            if isinstance(coeff,complex):
                if not op.codomain.is_complex:
                    self.log.warning("Given a complex coefficient for an operator with non complex domain. Casting the coefficient to real!")
                    coeff = coeff.real
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
        if len(domains) == 0:
            raise RuntimeError(util.Errors._compose_message(
                    "",
                    "While constructing a Linear combination the domains list remained empty!"))
        domain = domains[0]
        if any(d != domain for d in domains):
            raise ValueError(util.Errors.not_equal(
                domain,
                domains,
                add_info="The Operators to be taken into a LinearCombination do not have\n matching domains:\n\t"
                                                +"\n\t".join([f"{op} with domain {op.domain}," for op in self.ops])))

        codomains = [op.codomain for op in self.ops if op.codomain]
        if len(codomains) == 0:
            raise RuntimeError(util.Errors._compose_message(
                    "",
                    "While constructing a Linear combination the codomains list remained empty!"))
        codomain = codomains[0]
        if any(cd != codomain for cd in codomains):
            raise ValueError(util.Errors.not_equal(
                codomain,
                codomains,
                add_info="The Operators to be taken into a LinearCombination do not have\n matching domains:\n\t"
                                                +"\n\t".join([f"{op} with domain {op.codomain}," for op in self.ops])))

        super().__init__(domain, codomain, linear=all(op.linear for op in self.ops))

    def _eval(self, x, differentiate=False,  **kwargs):
        if differentiate:
            self._derivs = []
        out = self.codomain.zeros()
        for coeff, op in zip(self.coeffs, self.ops):
            if differentiate:
                tup = op.linearize(x,**kwargs)
                z = tup[0]
                self._derivs.append(tup[1])
            else:
                z = op(x,**kwargs)
            z *= coeff
            out += z
        return out
    
    def _iadjoint_eval(self, x, out, **kwargs):
        if len(self.ops) == 1:
            if self.linear:
                out = self.ops[0].adjoint_eval(x, out = out, **kwargs)
            else:
                out, deriv = self.ops[0].linearize(x,return_adjoint_eval = True, out = out)
                self._derivs = [deriv]
            out *= np.abs(self.coeffs[0])**2
            return out
        self.log.warning("A default fast implementation for an adjoint evaluation of a linear combination is only available for a single scalar multiplication! Consider defining your own operator with a faster method!")
        return super()._iadjoint_eval(x, out, **kwargs)
    
    def _iadjoint_derivative(self, x, out, **kwargs):
        if len(self.ops) == 1:
            out = self._derivs[0].adjoint_eval(x, out = out, **kwargs)
            out *= np.abs(self.coeffs[0])**2
            return out
        self.log.warning("A default fast implementation for an adjoint derivative of a linear combination is only available for a single scalar multiplication! Consider defining your own operator with a faster method!")
        return super()._iadjoint_derivative(x, out,**kwargs)

    def _iderivative(self, x, out, **kwargs):
        for coeff, deriv in zip(self.coeffs, self._derivs):
            out += coeff * deriv(x,**kwargs)
        return out

    def _iadjoint(self, y, out, **kwargs):
        if y is out:
            y = y.copy()
        if self.linear:
            ops = self.ops
        else:
            ops = self._derivs
        for coeff, op in zip(self.coeffs, ops):
            out += coeff.conjugate() * op.adjoint(y,**kwargs)
        return out
       
    def _iadjoint_data(self, x, out, **kwargs):
        if x is out:
            x = x.copy()
        for coeff, op in zip(self.coeffs, self.ops):
            out += coeff.conj() * op.adjoint_data(x,**kwargs)
        return out

    @Operator.inverse.getter
    def inverse(self):
        if self._inverse is not None:
            return self._inverse
        if len(self.ops) > 1:
            raise NotImplementedError(util.Errors._compose_message(
                    "INVERSE NOT DEFINED",
                f"The inverse of the linear combination {self} is not defined. Since it was not explicitly defined and automatically computing it with more then one operator is ambiguous. You may specify an explicit inverse by setting self.inverse for this operator."
                ))
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
            raise ValueError(util.Errors.not_instance(
                ops[0],
                Operator,
                add_info="The first argument of the list of operators is not an Operator."
                ))
        for i,(f, g) in enumerate(zip(ops, ops[1:])):
            if not isinstance(g,Operator):
                raise ValueError(util.Errors.not_instance(
                    g,
                    Operator,
                    add_info=f"The {i+2}-th entry of operators is not an Operator."
                    ))  
            if not f.domain.compatible(g.codomain):
                raise ValueError(util.Errors.not_equal(
                    f,
                    g,
                    add_info=f"The domain of the {i+1}-th and codomain of {i+2}-th entry do not match up."
                    ))
        self.ops = []
        """The list of composed operators."""
        self.d_eq_cd = []
        """Stores the information if domain and codomain are equal to possible have in-place evaluations."""
        for op in ops:
            if isinstance(op, Composition):
                self.ops.extend(op.ops)
                self.d_eq_cd.extend(op.d_eq_cd)
            else:
                self.ops.append(op)
                self.d_eq_cd.append(op.domain.compatible(op.codomain))
        super().__init__(
            self.ops[-1].domain, self.ops[0].codomain,
            linear=all(op.linear for op in self.ops))

    def _ieval(self, x, out, differentiate=False, **kwargs):
        if differentiate:
            self._derivs = []
            x, deriv = self.ops[-1].linearize(x,**kwargs)
            self._derivs.append(deriv)
            for d_eq_cd, op in zip(self.d_eq_cd[-2:0:-1],self.ops[-2:0:-1]):
                if d_eq_cd:
                    x, deriv = op.linearize(x, out = x,**kwargs)
                else:
                    x, deriv = op.linearize(x,**kwargs)
                self._derivs.insert(0,deriv)
            out, deriv = self.ops[0].linearize(x, out = out,**kwargs)
            self._derivs.insert(0,deriv)
            return out
        else:
            x = self.ops[-1](x,**kwargs)
            for d_eq_cd, op in zip(self.d_eq_cd[-2:0:-1],self.ops[-2:0:-1]):
                if d_eq_cd:
                    x = op(x, out = x,**kwargs)
                else:
                    x = op(x,**kwargs)
            return self.ops[0](x, out = out,**kwargs)

    def _iderivative(self, x, out, **kwargs):
        x = self._derivs[-1](x, **kwargs)
        for d_eq_cd, deriv in zip(self.d_eq_cd[-2:0:-1],self._derivs[-2:0:-1]):
            if d_eq_cd:
                x = deriv(x, out = x, **kwargs)
            else:
                x = deriv(x, **kwargs)
        return self._derivs[0](x, out = out,**kwargs)

    def _iadjoint(self, y, out, **kwargs):
        if self.linear:
            ops = self.ops
        else:
            ops = self._derivs
        y = ops[0].adjoint(y, **kwargs)
        for d_eq_cd,op in zip(self.d_eq_cd[1:-1],ops[1:-1]):
            if d_eq_cd:
                y = op.adjoint(y, out = y, **kwargs)
            else:
                y = op.adjoint(y, **kwargs)
        out = ops[-1].adjoint(y, out = out,**kwargs)
        return out
    
    def _iadjoint_eval(self, x, out, **kwargs):
        if len(kwargs) != 0:
            self.log.warning(f"In the default implementation of _adjoint_eval it is unclear where to put keyword arguments.Maybe implement your own _adjoint_eval and process the kwargs Thus ignoring yours: kwargs= {kwargs}")
        if self.linear:
            x = self.ops[-1](x)
            for d_eq_cd,op in zip(self.d_eq_cd[-2:0:-1],self.ops[-2:0:-1]):
                self.log.debug(f"domain = {op.domain}, codomain = {op.codomain}")
                if d_eq_cd:
                    x = op(x, out = x)
                else:
                    x = op(x)
            x = self.ops[0].adjoint_eval(x, out = x)
            for d_eq_cd,op in zip(self.d_eq_cd[1:-1],self.ops[1:-1]):
                if d_eq_cd:
                    x = op.adjoint(x, out = x)
                else:
                    x = op.adjoint(x)
            return self.ops[-1].adjoint(x, out = out)
        else:
            self._derivs = []
            x, deriv = self.ops[-1].linearize(x)
            self._derivs.append(deriv)
            for d_eq_cd,op in zip(self.d_eq_cd[-2:0:-1],self.ops[-2:0:-1]):
                if d_eq_cd and not first:
                    x, deriv = op.linearize(x, out = x)
                else:
                    x, deriv = op.linearize(x)
                self._derivs.insert(0,deriv)
            x, deriv = self.ops[0].linearize(x, out = x,return_adjoint_eval=True)
            self._derivs.insert(0,deriv)
            for d_eq_cd,deriv in zip(self.d_eq_cd[1:-1],self._derivs[1:-1]):
                if d_eq_cd:
                    x = deriv.adjoint(x, out = x)
                else:
                    x = deriv.adjoint(x)
            return self._derivs[-1].adjoint(x, out = out)
    
    def _iadjoint_data(self, data, out, **kwargs):
        if self.linear:
            ops = self.ops
        else:
            ops = self._derivs
        if self.d_eq_cd[0]:
            data = ops[0].adjoint_data(data, out = data, **kwargs)
        else:
            data = ops[0].adjoint_data(data, **kwargs)
        for d_eq_cd,op in zip(self.d_eq_cd[1:-1],ops[1:-1]):
            if d_eq_cd:
                data = op.adjoint(data, out = data, **kwargs)
            else:
                data = op.adjoint(data, **kwargs)
        return ops[-1].adjoint(data, out = out, **kwargs)
    
    def _iadjoint_derivative(self, x, out, **kwargs):
        if len(kwargs) != 0:
            self.log.warning(f"In the default implementation of _adjoint_eval it is unclear where to put keyword arguments.Maybe implement your own _adjoint_eval and process the kwargs Thus ignoring yours: kwargs= {kwargs}")
        x = self._derivs[-1](x)
        for d_eq_cd,deriv in zip(self.d_eq_cd[-2:0:-1],self._derivs[-2:0:-1]):
            if d_eq_cd:
                x = deriv(x, out = x)
            else:
                x = deriv(x)
        x = self._derivs[0].adjoint_eval(x, out = x)
        for d_eq_cd,deriv in zip(self.d_eq_cd[1:-1],self._derivs[1:-1]):
            if d_eq_cd:
                x = deriv.adjoint(x, out = x)
            else:
                x = deriv.adjoint(x)
        return self._derivs[-1].adjoint(x, out = out)

    @Operator.inverse.getter
    def inverse(self):
        if self._inverse is not None:
            return self._inverse
        try:
            return Composition(*(op.inverse for op in self.ops[::-1]))
        except NotImplementedError:
            raise NotImplementedError(util.Errors._compose_message(
                    "INVERSE NOT DEFINED",
                f"The inverse of the composition {self} is not well-defined since one or more of the operators does not have an inverse."
                ))

    def __repr__(self):
        if hasattr(self,"ops"):
            return util.make_repr(self, *[type(op).__qualname__ for op in self.ops])
        else:
            return util.make_repr(self)


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
        if not isinstance(base_op.codomain,vecsps.DirectSum):
            raise ValueError(util.Errors.not_instance(
                base_op,
                vecsps.DirectSum,
                f"To construct a PartOfOperator the codomain has to be a DirectSum!"))
        self.base_op=base_op
        """The base operator being sliced.
        """
        n_codim = len(base_op.codomain.summands)
        if(isinstance(index,int)):
            if index<-n_codim or n_codim<=index:
                raise IndexError(util.Errors.indexation(
                    index,
                    self,
                    f"The given integer index is out of the range of {-n_codim},...,{n_codim}."))
            self.index=index
        elif(isinstance(index,slice)):
            if index.stop is not None and (index.stop<-n_codim or n_codim<=index.stop):
                raise IndexError(util.Errors.indexation(
                    index,
                    self,
                    f"The given index slice stops out of the range of {-n_codim},...,{n_codim}."))
            if index.start is not None and (index.start<-n_codim or n_codim<=index.start):
                raise IndexError(util.Errors.indexation(
                    index,
                    self,
                    f"The given index slice starts out of the range of {-n_codim},...,{n_codim}."))
            index_list=list(range(n_codim)[index])
            if len(index_list) == 0:
                raise IndexError(util.Errors.indexation(
                    index,
                    self,
                    f"The given index slice generated an empty index list."
                ))
            if(len(index_list)==1):
                self.index=index_list[0]
            else:
                self.index=tuple(index_list)
        elif(isinstance(index,(tuple,list))):
            if any(not isinstance(i,int) for i in index) or min(index)<-n_codim or n_codim<=max(index):
                raise IndexError(util.Errors.indexation(
                    index,
                    self,
                    f"The given tuple/list of indeces has to be a list of integers in the range of {-n_codim},...,{n_codim}."))
            if(len(index)==1):
                self.index=index[0]
            else:
                self.index=tuple(index)
        else:
            raise ValueError(util.Errors.indexation(f"Invalid type {type(index)} for index in PartOfOperator"))
        if(isinstance(self.index,int)):
            codomain=base_op.codomain.summands[self.index]
        else:
            codomain=vecsps.DirectSum(*[base_op.codomain.summands[i] for i in self.index])
        super().__init__(self.base_op.domain,codomain,linear=self.base_op.linear)

    def _get_codomain_part(self,y):
        if(isinstance(self.index,int)):
            return y[self.index]
        else:
            return self.codomain.join(*[y[i] for i in self.index])
        
    def _extend_to_full(self, out):
        if isinstance(self.index,int):
            return self.base_op.codomain.join(*[cd.zeros() if i!=self.index else out for i,cd in enumerate(self.base_op.codomain)])
        else:
            return self.base_op.codomain.join(*[cd.zeros() if i not in self.index else out[self.index.index(i)] for i,cd in enumerate(self.base_op.codomain)])

    def _ieval(self, x, out, differentiate=False, **kwargs):
        if(self.base_op.linear):
            y=self.base_op._ieval(x, self._extend_to_full(out), **kwargs)
        else:
            y=self.base_op._ieval(x, self._extend_to_full(out), differentiate=differentiate, **kwargs)
        return self._get_codomain_part(y)
    
    def _iderivative(self, x, out, **kwargs):
        y = self.base_op._iderivative(x, self._extend_to_full(out), **kwargs)
        return self._get_codomain_part(y)

    def _iadjoint(self, y, out, **kwargs):
        out = self.base_op._iadjoint(self._extend_to_full(y), out, **kwargs)
        return out
    
    def __getitem__(self, val):
        if not isinstance(self.index,tuple):
            raise IndexError(util.Errors.indexation(
                val,
                self,
                "Cannot index a PartOfOperator Further then to a signle element!"
            ))
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
        if not op.linear:
            raise ValueError(util.Errors.not_linear_op(
                op,
                "To construct an power of an operator the operator has to be linear."
            ))
        if op.domain != op.codomain:
            raise ValueError(util.Errors.not_equal(
                op.domain,
                op.codomain,
                add_info=f"The domain and codomain of the operator {op} have to match to construct a power of it."
            ))
        if not isinstance(exponent,int) or exponent<0:
            raise ValueError(util.Errors.not_instance(
                exponent,
                int,
                f"The exponent of a power of an operator has to be a non-negative integer."
            ))
        super().__init__(op.domain,op.domain,linear=True)
        self.op = op
        self.exponent = exponent
    
    def _eval(self,x,**kwargs):
        res = x
        for _ in range(self.exponent):
            res = self.op(res, **kwargs)
        return res

    def _ieval(self, x, out, **kwargs):
        if out is x:
            for _ in range(self.exponent):
                out = self.op(out, out = out, **kwargs)
            return out
        else:
            out *= 0
            out += x
            for _ in range(self.exponent):
                out = self.op(out, out = out, **kwargs)
            return out

    def _adjoint(self,x, **kwargs):
        res = x
        for j in range(self.exponent):
            res = self.op.adjoint(res)
        return res

    def _iadjoint(self,x, out = None, **kwargs):
        if out is x:
            for _ in range(self.exponent):
                out = self.op.adjoint(out, out = out, **kwargs)
            return out
        else:
            out *= 0
            out += x
            for _ in range(self.exponent):
                out = self.op.adjoint(out, out = out, **kwargs)
            return out
    
    @Operator.inverse.getter
    def inverse(self):
        if self._inverse is not None:
            return self._inverse
        try:
            return Pow(self.op.inverse,self.exponent)
        except NotImplementedError:
            raise NotImplementedError(util.Errors._compose_message(
                    "INVERSE NOT DEFINED",
                f"The inverse of the power of the operator {self.op} is not defined since the operator has not a well defined inverse."
                ))


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

    def _eval(self, x, copy = False):
        if self.copy or copy:
            return x.copy()
        else:
            return x

    def _adjoint(self, x, **kwargs):
        return self._eval(x, **kwargs)
        
    def _adjoint_eval(self, x, **kwargs):
        return self._eval(x, **kwargs)
    
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
            try:
                mask = np.broadcast_to(mask, domain.shape)
            except:
                raise ValueError(util.Errors._compose_message(
                    "BROADCAST ERROR",
                    f"The mask for a CoordinateProjection for NumPyVectorSpace instances has to be broadcastable to the shape of the domain {domain.shape}. "+"\n\t"+ f"mask = {mask}"
                ))
            if mask.dtype != bool:
                raise TypeError(util.Errors.not_instance(
                    mask,
                    bool,
                    f"To construct a CoordinateProjection for a NumPyVectorSpace the given mask has to be of boolean type. "
                ))
        else:
            try:
                x = domain.rand()
                _ = x[mask]
                x[mask] = domain.ones()[mask]
            except:
                raise ValueError(util.Errors._compose_message(
                    "MASKING ERROR",
                    f"The mask for a CoordinateProjection for {domain} has to be able to get and set items. "+"\n\t "+f"mask = {mask}"
                ))
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

    def _iadjoint_eval(self, x, out):
        if x is out:
            out[~self.mask] = 0
            return out
        else:
            out *= 0
            out[self.mask] = x[self.mask]
            return out

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
        if isinstance(domain,vecsps.NumPyVectorSpace):
            try:
                mask = np.broadcast_to(mask, domain.shape)
            except:
                raise ValueError(util.Errors._compose_message(
                    "BROADCAST ERROR",
                    f"The mask for a CoordinateProjection for NumPyVectorSpace instances has to be broadcastable to the shape of the domain {domain.shape}. "+"\n\t "+f"mask = {mask}"
                ))
            if mask.dtype != bool:
                raise TypeError(util.Errors.not_instance(
                    mask,
                    bool,
                    f"To construct a CoordinateProjection for a NumPyVectorSpace the given mask has to be of boolean type. "
                ))
        else:
            try:
                x = domain.rand()
                _ = x[mask]
                x[mask] = domain.ones()[mask]
            except:
                raise ValueError(util.Errors._compose_message(
                    "MASKING ERROR",
                    f"The mask for a CoordinateProjection for {domain} has to be able to get and set items. "+"\n\t "+f"mask = {mask}"
                ))
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

    def _ieval(self, x, out):
        if x is out:
            out[~self.mask] = 0
            return out
        else:
            out *= 0
            out[self.mask] = x[self.mask]
            return out

    def _adjoint(self, x):
        return self._eval(x)
    
    def _adjoint_eval(self, x):
        return self._eval(x)

    def _iadjoint(self, x, out):
        return self._ieval(x, out)
    
    def _iadjoint_eval(self, x, out):
        return self._eval(x, out)

    def __repr__(self):
        return util.make_repr(self, self.domain)


class PtwMultiplication(Operator):
    r"""A multiplication operator by a constant factor where each vector entry is multiplied 
    by the vector entry of `factor`. This works analogous to the pointwise multiplication in numpy. Note that the pointwise multiplication of the coefficients
    might not represent a pointwise multiplication of the underlying functions depending on the vector space.

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
        if not isinstance(domain,vecsps.VectorSpaceBase):
            raise ValueError(util.Errors.not_a_vecsp(
                domain,
                vecsps.VectorSpaceBase,
                add_info="The domain for a PtwMultiplication has to be a VectorSpaceBases instance."
            ))
        if isinstance(domain,vecsps.NumPyVectorSpace):
            try:
                factor = np.broadcast_to(factor, domain.shape)
            except:
                raise ValueError(util.Errors._compose_message(
                    "BROADCAST ERROR",
                    f"The factor for a PtwMultiplication for NumPyVectorSpace instances \n has to be broadcastable to the shape of the domain {domain.shape}. "+"\n\t "+f"factor = {factor}"
                ))
        elif np.isscalar(factor):
            factor = factor*domain.ones()
        elif factor not in domain:
            raise ValueError(util.Errors.not_in_vecsp(
                factor,
                domain,
                vec_name="factor",
                space_name="domain",
                add_info="For a PtwMultiplication the factor has to be in the domain."
            ))
        self.factor = factor.copy()
        super().__init__(domain, domain, linear=True)

    def _eval(self, x):
        return self.factor * x

    def _ieval(self, x, out):
        if out is not x:
            out *= 0
            out += x
        out *= self.factor
        return out

    def _adjoint(self, x):
        if self.domain.is_complex:
            return x * self.factor.conj()
        else:
            return x * self.factor

    def _iadjoint(self, x, out):
        if out is not x:
            out *= 0
            out += x
        if self.domain.is_complex:
            out *= self.factor.conj()
        else:
            out *= self.factor
        return out

    def _adjoint_eval(self, x):
        if self.domain.is_complex:
            return x * self.factor.conj()*self.factor
        else:
            return x * self.factor**2

    def _iadjoint_eval(self, x, out):
        if out is not x:
            out *= 0
            out += x
        if self.domain.is_complex:
            out *= self.factor.conj()*self.factor
        else:
            out *= self.factor**2
        return out

    @property
    def invertible(self):
        # TODO: check if multiplier has zeros!
        return True

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
        if not isinstance(op,Operator):
            raise ValueError(util.Errors.not_instance(
                op,
                Operator,
                add_info="Construction of an OuterShift is only possible for an Operator instance."
            ))
        if np.isscalar(offset):
            offset = op.codomain.ones()*offset
        elif offset not in op.codomain:
            raise ValueError(util.Errors.not_in_vecsp(
                offset,
                op.codomain,
                vec_name="offset",
                space_name="codomain",
                add_info="Construction of a OuterShift failed!"
            ))
        offset = offset
        super().__init__(op.domain, op.codomain)
        if isinstance(op, type(self)):
            offset = offset + op.offset
            op = op.op
        self.op = op
        self.offset = offset.copy()

    def _ieval(self, x, out, differentiate=False, **kwargs):
        if differentiate:
            out, self._deriv = self.op.linearize(x, out = out, **kwargs)
            out += self.offset
            return out
        else:
            out = self.op(x, out=out,**kwargs)
            out += self.offset
            return out

    def _iadjoint_eval(self, x, out, **kwargs):
        out, self._deriv = self.op.linearize(x, out = out, return_adjoint_eval= True, **kwargs)
        out += self._adjoint(self.offset)
        return out

    def _iderivative(self, x, out, **kwargs):
        return self._deriv(x, out = out, **kwargs)

    def _iadjoint(self, y, out, **kwargs):
        return self._deriv.adjoint(y, out = out, **kwargs)
    
    def _iadjoint_derivative(self, x, out, **kwargs):
        return self._deriv.adjoint_eval(x, out = out, **kwargs)


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
        if not isinstance(op,Operator):
            raise ValueError(util.Errors.not_instance(
                op,
                Operator,
                add_info="Construction of an InnerShift is only possible for an Operator instance."
            ))
        if np.isscalar(offset):
            offset = op.domain.ones()*offset
        elif offset not in op.domain:
            raise ValueError(util.Errors.not_in_vecsp(
                offset,
                op.domain,
                vec_name="offset",
                space_name="domain",
                add_info="Construction of a InnerShift failed!"
            ))
        offset = offset
        super().__init__(op.domain, op.codomain)
        if isinstance(op, type(self)):
            offset = offset + op.offset
            op = op.op
        self.op = op
        self.offset = offset.copy()

    def _ieval(self, x, out, differentiate=False, **kwargs):
        if differentiate:
            out, self._deriv = self.op.linearize(x-self.offset, out = out, **kwargs)
            return out
        else:
            return self.op(x - self.offset, out = out, **kwargs)
        
    def _iadjoint_eval(self, x, out, **kwargs):
        out, self._deriv = self.op.linearize(x-self.offset, out = out, return_adjoint_eval=True, **kwargs)
        return out 
    
    def _iderivative(self, h, out, **kwargs):
        return self._deriv(h, out = out, **kwargs)

    def _iadjoint(self, y, out, **kwargs):
        return self._deriv.adjoint(y, out = out, **kwargs)
    
    def _iadjoint_derivative(self, x, out, **kwargs):
        return self._deriv.adjoint_eval(x, out = out, **kwargs)

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
        if any(not isinstance(op,Operator) for op in ops):
            raise ValueError(util.Errors.not_instance(
                ops,
                tuple(Operator),
                add_info=f"Construction of a direct sum requires the variable number of arguments given to be Operator instances."
            ))
        self.ops = []
        r""" List of all operators :math:`(T_1,\dots,T_n)`"""
        for op in ops:
            if flatten and isinstance(op, type(self)):
                self.ops.extend(op.ops)
            else:
                self.ops.append(op)
        if isinstance(domain,vecsps.DirectSum):
            if any([d != op.domain for d,op in zip(domain.summands,self.ops)]):
                raise ValueError(util.Errors.not_equal(
                    domain,
                    [op.domain for op in self.ops],
                    add_info=f"Was given a DirectSum whos components do not match with the domain of the operators."
                    ))
            else:
                pass
        elif domain is None:
            domain = vecsps.DirectSum(*[op.domain for op in self.ops])
        elif callable(domain):
            domain = domain(*(op.domain for op in self.ops))
            if not isinstance(domain,vecsps.DirectSum):
                raise ValueError(util.Errors.not_a_vecsp(
                    domain,
                    vecsps.DirectSum,
                    add_info="The given callabel to construct the domain for a DirectSum Operator did not produce a DirectSum vector space."
                ))
            if any([d != op.domain for d,op in zip(domain.summands,self.ops)]):
                raise TypeError(util.Errors.not_equal(
                    domain,
                    [op.domain for op in self.ops],
                    add_info="The callable to construct the domain for  a DirectSum Operator created a DirectSum vectorspace whos components do not match with the domain of the operators. "
                    ))
        else:
            raise TypeError(util.Errors.not_a_vecsp(
                domain,
                vecsps.DirectSum,
                add_info='domain={} is neither a VectorSpaceBase nor callable'.format(domain)
                ))
        
        if isinstance(codomain,vecsps.DirectSum):
            if any([d != op.codomain for d,op in zip(codomain.summands,self.ops)]):
                raise ValueError(util.Errors.not_equal(
                    codomain,
                    [op.codomain for op in self.ops],
                    add_info="Was given a DirectSum whos components do not match with the domain of the operators."
                    ))
            else:
                pass
        elif codomain is None:
            codomain = vecsps.DirectSum(*[op.codomain for op in self.ops])
        elif callable(codomain):
            codomain = codomain(*(op.codomain for op in self.ops))
            if not isinstance(domain,vecsps.DirectSum):
                raise ValueError(util.Errors.not_a_vecsp(
                    codomain,
                    vecsps.DirectSum,
                    add_info="The given callabel to construct the codomain for a DirectSum Operator did not produce a DirectSum vector space."
                ))
            if any([d != op.codomain for d,op in zip(codomain.summands,self.ops)]):
                raise TypeError(util.Errors.not_equal(
                    codomain,
                    [op.codomain for op in self.ops],
                    add_info="The callable to construct the codomain for a DirectSum Operator created a DirectSum vectorspace whos components do not match with the codomain of the operators."
                    ))
        else:
            raise TypeError(util.Errors.not_a_vecsp(
                domain,
                vecsps.DirectSum,
                add_info='codomain={} is neither a VectorSpaceBase nor callable'.format(codomain)
                ))
        super().__init__(domain=domain, codomain=codomain, linear=all(op.linear for op in self.ops))

    def _ieval(self, x, out, differentiate=False, **kwargs):
        if differentiate:
            linearizations = [op.linearize(x_i,out = out_i) for op, x_i, out_i in zip(self.ops,x, out)]
            self._derivs = [l[1] for l in linearizations]
            return out
        else:
            for op, x_i, out_i in zip(self.ops, x, out):
                out_i = op(x_i, out = out_i)
            return out

    def _iadjoint_eval(self, x, out, **kwargs):
        if hasattr(self,"full_domain") and out not in self.full_domain:
            out = self._insert_constants(out)
        if self.linear:
            for op, x_i, out_i in zip(self.ops, x, out):
                out_i = op.adjoint_eval(x_i, out = out_i)
        else:
            self._derivs = []
            for op, x_i, out_i in zip(self.ops, x, out):
                out_i, deriv = op.linearize(x_i, out = out_i,return_adjoint_eval=True)
                self._derivs.append(deriv)
        return out

    def _iderivative(self, x, out, **kwargs):
        for deriv, x_i, out_i in zip(self._derivs, x, out):
            out_i = deriv(x_i, out = out_i)
        return out

    def _iadjoint(self, y, out, **kwargs):
        if hasattr(self,"full_domain") and out not in self.full_domain:
            out = self._insert_constants(out)
        if self.linear:
            ops = self.ops
        else:
            ops = self._derivs
        for op, y_i, out_i in zip(ops, y, out):
            out_i = op.adjoint(y_i, out = out_i)
        return out
    
    def _iadjoint_derivative(self, x, out, **kwargs):
        if hasattr(self,"full_domain") and out not in self.full_domain:
            out = self._insert_constants(out)
        for deriv, y_i, out_i in zip(self._derivs, x, out):
            out_i = deriv.adjoint_eval(y_i, out = out_i)
        return out

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
            raise NotImplementedError(util.Errors._compose_message(
                    "INVERSE NOT DEFINED",
                    "The inverse of the DirectSum {} is not known \n since one of the operators has not a well defined inverse.".format(self)
                ))

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
        if len(ops) == 0:
            raise ValueError(util.Errors._compose_message(
                    "",
                "The list of Operators for a VectorOfOperators cannot be empty!"
            ))
        if any(not isinstance(op,Operator) for op in ops):
            raise ValueError(util.Errors.not_instance(
                ops,
                tuple(Operator),
                add_info="Construction of a VectorOfOperators requires the a list/tuple of Operator instances."
            ))
        self.ops = ops
        r"""List of all Operators :math:`(T_1,\dots,T_n)`"""

        if domain is None:
            self.domain = self.ops[0].domain
        else:
            self.domain = domain
        if any(op.domain != self.domain for op in self.ops):
            raise TypeError(util.Errors.not_equal(
                self.domain,
                [op.domain for op in self.ops],
                second_type="list of all domains",
                add_info="The Operators in the VectorOfOperators have to have identical domains!"
            ))

        if codomain is None:
            codomain = vecsps.DirectSum(*tuple([op.codomain for op in ops]))
        if isinstance(codomain, vecsps.VectorSpaceBase):
            pass
        elif callable(codomain):
            codomain = codomain(*(op.codomain for op in self.ops))
        else:
            raise TypeError(util.Errors.not_instance(
                codomain,
                vecsps.VectorSpaceBase,
                'codomain for VectorOfOperators is neither a VectorSpaceBase nor callable'
            ))
        if not isinstance(codomain,vecsps.DirectSum):
            raise TypeError(util.Errors.not_a_vecsp(
                codomain,
                vecsps.DirectSum,
                add_info="The codomain of a VectorOfOperators must be a DiectSum"
            ))
        if any(op.codomain != c for op,c in zip(ops,codomain)):
            raise TypeError(util.Errors.not_equal(
                codomain,
                [op.codomain for op in ops],
                second_type="List(Codomains)",
                add_info="The codomains components given or constructed does match \nwith the codomains of the individual operators of the VecotrOfOperators."
            ))
        super().__init__(domain=self.domain, codomain=codomain, linear=all(op.linear for op in ops))

    def _ieval(self, x, out, differentiate=False, **kwargs):
        if differentiate:
            self._derivs = []
            for i, op in enumerate(self.ops):
                out[i], deriv = op.linearize(x, out = out[i],**kwargs)
                self._derivs.append(deriv)
            return out
        else:
            for op, out_i in zip(self.ops,out):
                out_i = op(x, out = out_i,**kwargs)
            return out
    
    def _adjoint_eval(self, x, **kwargs):
        out = self.domain.zeros()
        if self.linear:
            out += sum(op.adjoint_eval(x,**kwargs) for op in self.ops)
        else:
            self._derivs = []
            for op in self.ops:
                tmp, deriv = op.linearize(x,return_adjoint_eval=True,**kwargs) 
                out += tmp
                self._derivs.append(deriv)
        return out

    def _iderivative(self, x, out, **kwargs):
        for i,deriv in enumerate(self._derivs):
            out[i] = deriv(x,**kwargs)
        return out

    def _adjoint(self, y, **kwargs):
        if self.linear:
            ops = self.ops
        else:
            ops = self._derivs
        out = self.domain.zeros()
        for op, y_i in zip(ops, y):
            out += op.adjoint(y_i, **kwargs)
        return out
    
    def _adjoint_derivative(self, x, **kwargs):
        out = self.domain.zeros() 
        for deriv in self._derivs:
            out += deriv.adjoint_eval(x,**kwargs)
        return out

    def __repr__(self):
        vec_repr = "[" +", ".join([repr(op) for op in self.ops])+"]"
        return util.make_repr(self, vec_repr)

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
    *ops : tuple(tuple(Operator)) or list(list(Operator)) 
        Variable number of tuples of Operator instances to build the matrix. Each tuple has to have 
        the same length and a zero operators should be given by None.
    domain, codomain : vecsps.VectorSpaceBase or callable, optional
        Either the underlying vector space or a factory function that will be called with all
        summands' vector spaces passed as arguments and should return a vecsps.DirectSum instance.
        The resulting vector space should be iterable, yielding the individual summands.
        Default: vecsps.DirectSum.
    """

    def __init__(self, ops,  domain=None, codomain=None):
        self.ops = ops
        r""" Matrix of Operators :math:`(T_ij)`"""
        if all((not isinstance(op_col,list) or len(op_col) != len(ops[0]) for op_col in ops)):
            raise ValueError(util.Errors.not_instance(
                ops,
                list(Operator),
                add_info="Construction of a MatrixOfOperators requires to define a Matrix by lists/tuples. That is defining [[Operator, ...],[Operator, ...], ...]. Moreover, the internal lists have to be of identical lengths. "
            ))
        ops_flat = [op for op_col in ops for op in op_col]
        if any((not isinstance(op, Operator) and op!=None for op in ops_flat)):
            raise ValueError(util.Errors._compose_message(
                    "NOT CORRECT TYPE",
                "Construction of a MatrixOfOperators requires to define a Matrix by lists/tuples. That is defining [[Operator/None, ...],[Operator/None, ...], ...]. The given lists Contains other objects than Operators or None: "+"\n" +"[[" +"],\n[".join([", ".join([repr(op) if op else "0" for op in row]) for row in self.ops])+"]"
            ))

        domains = [None]*len(ops)
        for j in range(len(ops)):
            for i in range(len(ops[0])):
                if ops[j][i]:
                    if domains[j]:
                        if domains[j] != ops[j][i].domain:
                            raise ValueError(util.Errors.not_equal(
                                ops[j][i].domain,
                                domains[j],
                                add_info="The domains of one column in the MatrixOfOperators have to be identical!"
                            ))
                    else:    
                        domains[j] = ops[j][i].domain
        if None in domains:
            raise ValueError(util.Errors._compose_message(
                    "EMPTY DOMAIN",
                f"At least one column of the MatrixOfOperators is empty and contains no domain. Since the domains {domains} has a not specified entry. When construction from given operots:"+"\n"+"[[" +"],\n[".join([", ".join([repr(op) if op else "0" for op in row]) for row in self.ops])+"]"
            ))

        if domain is None:
            domain = vecsps.DirectSum(*tuple(domains))
        elif isinstance(domain, vecsps.DirectSum):
            pass
        elif callable(domain):
            domain = domain(*tuple(domains))
            if not isinstance(domain,vecsps.DirectSum):
                raise TypeError(util.Errors.not_a_vecsp(
                    domain,
                    vecsps.DirectSum,
                    add_info="The domain of a MatrixOfOperators constructed from a callable is not a DiectSum"
                ))
        else:
            raise TypeError(util.Errors.not_instance(
                domain,
                vecsps.DirectSum,
                add_info='domain is neither a DirectSum nor callable'
                ))


        codomains = [None]*len(ops[0])
        for i in range(len(ops[0])):
            for j in range(len(ops)):
                if ops[j][i]:
                    if codomains[i]:
                        if codomains[i] != ops[j][i].codomain:
                            raise ValueError(util.Errors.not_equal(
                                ops[j][i].codomain,
                                codomains[i],
                                add_info="The codomains of one row in the MatrixOfOperators have to be identical!"
                            ))
                    else:
                        codomains[i] = ops[j][i].codomain
        if None in codomains:
            raise ValueError(util.Errors._compose_message(
                    "EMPTY CODOMAIN",
                f"At least one row of the MatrixOfOperators is empty and contains no codomain. Since the codomains:{codomains} has an not specified entry. When construction from given operots:"+"\n"+"[[" +"],\n[".join([", ".join([repr(op) if op else "0" for op in row]) for row in self.ops])+"]"
            ))

        if codomain is None:
            codomain = vecsps.DirectSum
        if isinstance(codomain, vecsps.VectorSpaceBase):
            pass
        elif callable(codomain):
            codomain = codomain(*tuple(codomains))
            if not isinstance(domain,vecsps.DirectSum):
                raise TypeError(util.Errors.not_a_vecsp(
                    domain,
                    vecsps.DirectSum,
                    add_info="The construction from a callable is not a DiectSum"
                ))
        else:
            raise TypeError(util.Errors.not_instance(
                codomain,
                vecsps.DirectSum,
                'codomain is neither a DirectSum nor callable'
                ))
        
        super().__init__(domain=domain, codomain=codomain, linear=all(op==None or op.linear for op in ops_flat))

    def _eval(self, x, differentiate=False, **kwargs):
        out = self.codomain.zeros()
        self._derivs = []
        for T_j, x_j in zip(self.ops,x):
            self._derivs.append([])
            for i, (T_ij,out_i) in enumerate(zip(T_j,out)):
                if differentiate:
                    if T_ij:
                        out_deriv,deriv = T_ij.linearize(x_j)
                        out_i += out_deriv
                        Tprime_ij = deriv
                    else:
                        Tprime_ij = None
                    self._derivs[i].append(Tprime_ij)
                else:   
                    if T_ij:
                        out_i += T_ij(x_j)
        return out

    def _derivative(self, x, **kwargs):
        out = self.codomain.zeros()
        for Tprime_j, x_j in zip(self._derivs,x):
            for Tprime_ij,out_i in zip(Tprime_j,out):
                if Tprime_ij:
                    out_i += Tprime_ij(x_j)
        return out

    def _adjoint(self, y, **kwargs):
        if self.linear:
            ops = self.ops
        else:
            ops = self._derivs
        out = self.domain.zeros()
        for Tprime_j, out_j in zip(ops, out):
            for Tprime_ij, y_i in zip(Tprime_j,y):
                if Tprime_ij:
                    out_j += Tprime_ij.adjoint(y_i)
        return out

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
        if not isinstance(domain,vecsps.DirectSum):
            raise ValueError(util.Errors.not_a_vecsp(
                domain,
                vecsps.DirectSum,
                add_info="To construct a Sum (summation operater) the domain has to be a DirectSum!"
            ))
        if any(domain.summands[0].shape!=summand.shape for summand in domain.summands):
            raise ValueError(util.Errors.not_equal(
                domain,
                domain.summands[0],
                first_type="DirectSum containing identical domains.",
                second_type="the first domain",
                add_info="The domain has to be a DirectSum of identical domains."
            ))
        if codomain is None:
            if(domain.is_complex):
                codomain=domain.summands[0].complex_space()
            else:
                codomain=domain.summands[0]  
        super().__init__(domain, codomain, True)
        if self.domain.summands[0].shape != codomain.shape:
            raise ValueError(util.Errors.not_equal(
                self.domain.summands[0],
                self.codomain,
                add_info="The codomain has to be indentical to the summands of the domain."
            ))

    def _eval(self,x):
        return sum(self.domain.split(x))
    
    
    def _iadjoint(self,y, out):
        for i, summand in enumerate(self.domain):
            out[i] = y.real if not summand.is_complex else y
        return out
    

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
        if not isinstance(domain,vecsps.DirectSum):
            raise ValueError(util.Errors.not_a_vecsp(
                domain,
                vecsps.DirectSum,
                add_info="To construct a Product (product operator) the domain has to be a DirectSum!"
            ))
        if any(domain.summands[0].shape!=summand.shape for summand in domain.summands):
            raise ValueError(util.Errors.not_equal(
                domain,
                domain.summands[0],
                first_type="DirectSum containing identical domains.",
                second_type="the first domain",
                add_info="The domain has to be a DirectSum of identical domains."
            ))
        if codomain is None:
            if(domain.is_complex):
                codomain=domain.summands[0].complex_space()
            else:
                codomain=domain.summands[0]
        super().__init__(domain, codomain, False)
        if self.domain.summands[0].shape != codomain.shape:
            raise ValueError(util.Errors.not_equal(
                self.domain.summands[0],
                self.codomain,
                add_info="The codomain has to be indentical to the summands of the domain."
            ))

    def _eval(self,x,differentiate=False):
        out = self.codomain.ones()
        for x_i in x:
            out *= x_i
        if differentiate:
            self.deriv_data=[self.codomain.ones() for _ in x]
            for j,x_j in enumerate(x):
                for i in range(self.domain.ndim):
                    if(i!=j):
                        self.deriv_data[i]*=x_j
        return out
    
    def _derivative(self, x):
        out = self.deriv_data[0]*x[0]
        for i,x_i in enumerate(x[1:]):
            out+=self.deriv_data[i+1]*x_i
        return out

    def _iadjoint(self,y, out):
        for i, summand in enumerate(self.domain.summands):
            out[i] = (y*self.deriv_data[i].conj()).real if not summand.is_complex else y*self.deriv_data[i].conj()
        return out


class RealPart(Operator):
    r"""The pointwise real part operator.

    Parameters
    ----------
    domain : regpy.vecsps.VectorSpaceBase
        The underlying vector space. The codomain will be the corresponding
        `regpy.vecsps.VectorSpaceBase.real_space`.
    """

    def __init__(self, domain):
        if not isinstance(domain,vecsps.VectorSpaceBase):
            raise ValueError(util.Errors.not_a_vecsp(
                domain,
                vecsps.VectorSpaceBase
            ))
        codomain = domain.real_space()
        super().__init__(domain, codomain, linear=True)

    def _eval(self, x, out = None):
        return x.real.copy()

    def _adjoint(self, y):
        out = self.domain.zeros()
        out += y
        return out

    def _iadjoint(self, y, out):
        out *= 0
        out += y
        return out


class ImaginaryPart(Operator):
    r"""The pointwise imaginary part operator.

    Parameters
    ----------
    domain : regpy.vecsps.VectorSpaceBase
        The underlying vector space. The codomain will be the corresponding
        `regpy.vecsps.VectorSpaceBase.real_space`.
    """

    def __init__(self, domain):
        if not isinstance(domain,vecsps.VectorSpaceBase) or not domain.is_complex:
            raise ValueError(util.Errors.not_a_vecsp(
                domain,
                vecsps.VectorSpaceBase,
                add_info="To consider a ImaginaryPart operator the domain is required to be complex!"
            ))
        codomain = domain.real_space()
        super().__init__(domain, codomain, linear=True)

    def _eval(self, x):
        return x.imag

    def _adjoint(self, y):
        return 1j * y

class SplitRealImag(VectorOfOperators):
    """Splits a complex vector into its real and imaginary part. 
    Implemented as VectorOfOperators of RealPart and ImaginaryPart.

    Parameters
    ----------
    domain: regpy.vecsps.VectorSpaceBase
    """
    def __init__(self, domain):
        if not isinstance(domain,vecsps.VectorSpaceBase) or not domain.is_complex:
            raise ValueError(util.Errors.not_a_vecsp(
                domain,
                vecsps.VectorSpaceBase,
                add_info="To consider a ImaginaryPart operator the domain is required to be complex!"
            ))
        Re = RealPart(domain)
        Im = ImaginaryPart(domain)
        super().__init__((Re,Im))

    @Operator.inverse.getter
    def inverse(self):
        return self.adjoint

    @property
    def invertible(self):
        return True

class SquaredModulus(Operator):
    r"""The pointwise squared modulus operator.

    Parameters
    ----------
    domain : regpy.vecsps.VectorSpaceBase
        The underlying vector space. The codomain will be the corresponding
        `regpy.vecsps.VectorSpaceBase.real_space`.
    """

    def __init__(self, domain):
        if not isinstance(domain,vecsps.VectorSpaceBase):
            raise ValueError(util.Errors.not_a_vecsp(
                domain,
                vecsps.VectorSpaceBase
            ))
        codomain = domain.real_space()
        super().__init__(domain, codomain)

    def _eval(self, x, differentiate=False):
        if differentiate:
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

    def _ieval(self,x,out):
        out *= 0
        return out

    def _adjoint(self, x):
        return self.domain.zeros()

    def _iadjoint(self, x, out):
        out *= 0
        return out
    
    def _adjoint_eval(self, x):
        return self.domain.zeros()
    
    def _iadjoint_eval(self, x, out):
        out *= 0
        return out

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
        if not isinstance(func,Functional):
            raise ValueError(util.Errors.not_instance(
                func,
                Functional,
            ))
        if not hasattr(func,"gradient"):
            raise ValueError(util.Errors._compose_message(
                    "MISSING ATTRIBUTE",
                f"The given functional {func} does not have a defined gradient. Making it impossible to approximate a Hessian operator."
            ))
        self.gradx = func.gradient(x)
        """The gradient at `x`"""
        self.func = func
        self.x = x.copy()
        self.stepsize = stepsize
        # linear=True is a necessary lie
        super().__init__(func.domain, func.domain, linear=True)
        self.log.info('Using approximate Hessian of functional {}'.format(self.func))

    def _eval(self, h):
        grad = self.func.subgradient(self.x + self.stepsize * h)
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
        if not isinstance(op2,Operator):
            raise ValueError(util.Errors.not_instance(
                op2,
                Operator
            ))
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

