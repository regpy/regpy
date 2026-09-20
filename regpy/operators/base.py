r"""
Forward operators
=================

This module provides the basis for defining forward operators, and implements some simple
auxiliary operators. Actual forward problems are implemented in submodules.

The base class is `Operator`.
"""

from __future__ import annotations

from typing import TypeVar
from collections import defaultdict
from collections.abc import Callable, Iterator, Sequence
from copy import deepcopy
from numbers import Real, Integral
from functools import wraps
from types import MethodType
from typing import Any, TYPE_CHECKING

from math import sqrt,inf

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse.linalg import LinearOperator

from regpy import util, vecsps
from regpy.vecsps.base import Vector, Scalar

if TYPE_CHECKING:
    from regpy.functionals import Functional
    from regpy.hilbert import HilbertSpace

__all__ = ["ObsDataT",
           "Operator", 
           "Pow", 
           "Identity", 
           "CoordinateProjection", 
           "CoordinateMask", 
           "PtwMultiplication", 
           "Composition",
           "OuterShift", 
           "InnerShift", 
           "DirectSum", 
           "VectorOfOperators", 
           "MatrixOfOperators", 
           "Sum",
           "Product", 
           "RealPart", 
           "ImaginaryPart", 
           "SplitRealImag",
           "SquaredModulus", 
           "Zero", 
           "SciPyLinearOperator"
           ]


class _Revocable:
    def __init__(self, val: Operator) -> None:
        self.__val = val

    @classmethod
    def take(cls, other: _Revocable) -> _Revocable:
        return cls(other.revoke())

    def get(self) -> Operator:
        try:
            return self.__val
        except AttributeError:
            raise RuntimeError(util.Errors._compose_message("REVOKED",
                'Attempted to use revoked reference')) from None

    def revoke(self) -> Operator:
        val = self.get()
        del self.__val
        return val

    @property
    def valid(self) -> bool:
        try:
            self.__val
            return True
        except AttributeError:
            return False

ObsDataT = TypeVar("ObsDataT")
"""
ObsDataT is a placeholder for a data type. 
In many cases, data are a :class:`Vector` in the codomain of an Operator. 
However, data may be anything that is accepted by :meth:`Operator.adjoint_data` 
and :meth:`regpy.functionals.Functional.as_data_func`.
"""

_EVAL_METHOD_KEYS = {
    '_eval': 'eval', '_ieval': 'eval',
    '_derivative': 'deriv', '_iderivative': 'deriv',
    '_adjoint': 'adjoint', '_iadjoint': 'adjoint',
    '_adjoint_eval': 'adjoint_eval', '_iadjoint_eval': 'adjoint_eval',
    '_adjoint_derivative': 'adjoint_deriv', '_iadjoint_derivative': 'adjoint_deriv',
    '_adjoint_data': 'adjoint_data', '_iadjoint_data': 'adjoint_data',
    'adjoint_data': 'adjoint_data',
}


def _count_evaluation(key):
    """Count an outer operation once, including in-place fallback dispatch."""
    def decorate(method):
        if getattr(method, '_counts_evaluation', False):
            return method
        @wraps(method)
        def counted(self, *args, **kwargs):
            depth = getattr(self, '_eval_count_depth', 0)
            if depth == 0:
                self._eval_counts[key] += 1
            self._eval_count_depth = depth + 1
            try:
                return method(self, *args, **kwargs)
            finally:
                self._eval_count_depth = depth
        counted._counts_evaluation = True
        return counted
    return decorate


class Operator:
    r"""Base class for linear and nonlinear forward operators.

    Operator instances are callable; calling one with a vector evaluates it.
    Domain and codomain vectors are duck typed according to their respective
    :class:`regpy.vecsps.VectorSpaceBase` instances. 

    Subclasses implementing nonlinear operators should normally implement the
    following methods:

        _eval(self, x, differentiate=False)
        _derivative(self, x)
        _adjoint(self, y)

    The ``_adjoint`` implementation may be omitted when the nonlinear operator
    is constructed with ``self_adjoint=True``, which declares its derivatives
    to be self-adjoint.

    These methods are not intended for external use, but should be invoked indirectly via calling
    the operator or using the `Operator.linearize` method. They must not modify their argument, and
    should return vectors that can be freely modified by the caller, i.e. should not share data
    with anything. Usually, this means they should allocate a new vector for the return value.

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
    as constants and should not be deep-copied to `self._consts` (a `set`). By default, `domain` and
    `codomain` will not be copied, since `regpy.vecsps.VectorSpaceBase` instances should never
    change in-place.

    If no derivative at some point is needed, `_eval` will be called with `differentiate=False`,
    allowing it to save on precomputations. It does not need to ensure that data shared with some
    derivative remains intact; all derivative instances will be invalidated regardless.

    For each operation family, subclasses normally need to implement only one of the
    allocating and in-place variants: ``_eval`` or ``_ieval``, ``_derivative`` or
    ``_iderivative``, and ``_adjoint`` or ``_iadjoint``. The base class derives the missing
    variant from the implemented one. The same applies to ``_adjoint_data``,
    ``_adjoint_eval`` and ``_adjoint_derivative`` and their corresponding ``_i*`` methods.
    Implementing both variants is useful only when their optimal algorithms differ.

    In general, subclasses should prefer implementing the in-place ``_i*`` variant: it can
    write directly into caller-provided storage and the base class can still construct the
    allocating result by creating an output vector once. Implementations must completely
    overwrite the requested output and return that exact output object. They should also be
    safe when input and output are identical whenever domain and codomain are compatible.

    Linear operators should therefore normally implement

        _ieval(self, x, out)
        _iadjoint(self, y, out)

    or, when a genuinely better allocating algorithm exists, the corresponding ``_eval``
    and ``_adjoint`` methods. If ``self_adjoint=True`` is passed to the
    constructor, implementing ``_adjoint`` or ``_iadjoint`` is unnecessary:
    adjoint evaluations use ``_eval`` or ``_ieval``, respectively.

    A nonlinear operator itself is never self-adjoint. For such an operator,
    the constructor argument ``self_adjoint=True`` instead declares that its
    derivative is self-adjoint at every linearization point. Consequently,
    :attr:`self_adjoint` remains false on the nonlinear operator, while the
    :class:`Derivative` returned by :meth:`linearize` has
    ``self_adjoint=True`` and uses ``_derivative`` for its adjoint evaluation.
    A self-adjoint operator or derivative must map a vector space to itself;
    therefore, the constructor requires identical domain and codomain whenever
    ``self_adjoint=True`` is requested.

    A linear operator that was not recognized as self-adjoint during its
    construction can be marked afterwards with
    :meth:`declare_as_self_adjoint`. This is useful for operators assembled
    from other operators in ways not covered by RegPy's operator algebra.

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
        op * vector          # composition with pointwise multiplication in domain
        op + arr             # operator shifted in codomain
        op + scalar          # dto.

    Parameters
    ----------
    domain, codomain : regpy.vecsps.VectorSpaceBase
        The vector space on which the operator's arguments / values are defined.
    linear : bool [default: False]
        Whether the operator is linear.
    inverse: Operator or None [default: None]
        If an explicit expression for the inverse exists, it can be provided in the constructor 
        and will be available under op.inverse.
    self_adjoint : bool [default: False]
        For a linear operator, whether the operator equals its adjoint. For a
        nonlinear operator, whether all its derivatives are self-adjoint; the
        nonlinear operator's public :attr:`self_adjoint` property nevertheless
        remains false.
    """

    log = util.ClassLogger()

    @property
    def eval_count_enabled(self) -> bool:
        """Whether counting is enabled on this instance (never recursively)."""
        return '_eval_count_originals' in self.__dict__

    def enable_eval_count(self, reset: bool = True) -> None:
        """Instrument this operator only; components remain unchanged.

        Disabled operators use their original methods without a counting
        wrapper or counter dictionary. Derivative/adjoint wrappers dispatch
        back to the instrumented methods of this operator. Enabling twice
        does not stack wrappers. Counts are reset by default.

        Compositions built while counting is enabled retain this operator as
        a component instead of flattening it. Enable counting before building
        a containing composition; already flattened compositions cannot be
        retroactively instrumented through the original object.
        """
        if not isinstance(reset, bool):
            raise TypeError("reset must be a bool.")
        if not self.eval_count_enabled:
            originals = {}
            for name, key in _EVAL_METHOD_KEYS.items():
                method = getattr(self, name)
                if not isinstance(method, MethodType):
                    raise TypeError(f"Cannot instrument non-bound method {name}.")
                originals[name] = self.__dict__.get(name)
            self._eval_count_originals = originals
            self._eval_counts = dict.fromkeys(set(_EVAL_METHOD_KEYS.values()), 0)
            for name, key in _EVAL_METHOD_KEYS.items():
                method = getattr(self, name)
                setattr(self, name, MethodType(_count_evaluation(key)(method.__func__), self))
        if reset:
            self.reset_eval_counts()

    def disable_eval_count(self) -> None:
        """Restore uninstrumented methods and discard counters on this instance."""
        for name, original in self.__dict__.pop('_eval_count_originals', {}).items():
            if original is None:
                self.__dict__.pop(name, None)
            else:
                setattr(self, name, original)
        self.__dict__.pop('_eval_counts', None)
        self.__dict__.pop('_eval_count_depth', None)

    @property
    def eval_counts(self) -> dict[str, int]:
        """Snapshot of this operator's logical operation counts.

        Counting is opt-in via enable_eval_count(). Reading this property
        while disabled returns zeros without allocating persistent counters.
        Components are not enabled automatically. Counts include attempts
        through derivative/adjoint wrappers, but nested fallback calls on the
        same instance count only once. Self-adjoint aliases count as eval.
        Modifying the snapshot does not alter counters. Serial use only.
        """
        if not self.eval_count_enabled:
            return dict.fromkeys(('eval', 'deriv', 'adjoint', 'adjoint_eval',
                                  'adjoint_deriv', 'adjoint_data'), 0)
        return dict(self._eval_counts)

    def reset_eval_counts(self) -> None:
        """Zero enabled counters without enabling counting or touching children."""
        if self.eval_count_enabled:
            self._eval_counts = dict.fromkeys(
                ('eval', 'deriv', 'adjoint', 'adjoint_eval', 'adjoint_deriv', 'adjoint_data'), 0
            )

    def total_eval_count(self) -> int:
        """Return counts weighted by two for adjoint_eval and adjoint_deriv."""
        counts = self.eval_counts
        return sum(counts.values()) + counts['adjoint_eval'] + counts['adjoint_deriv']

    def __init__(
        self,
        domain: vecsps.VectorSpaceBase | None = None,
        codomain: vecsps.VectorSpaceBase | None = None,
        linear: bool = False,
        inverse: Operator | None = None,
        self_adjoint: bool = False,
    ) -> None:
        if not isinstance(domain,vecsps.VectorSpaceBase):
            raise ValueError(util.Errors.not_a_vecsp(domain,vecsps.VectorSpaceBase,"The domain of an operator has to be some derivative of vector spaces."))
        if not isinstance(codomain,vecsps.VectorSpaceBase):
            raise ValueError(util.Errors.not_a_vecsp(codomain,vecsps.VectorSpaceBase,"The codomain of an operator has to be some derivative of vector spaces."))
        if not isinstance(self_adjoint, bool):
            raise TypeError(util.Errors.type_error(
                "self_adjoint must be a boolean.", self_adjoint
            ))
        if self_adjoint and domain != codomain:
            raise ValueError(util.Errors.value_error(
                "self_adjoint=True requires identical domain and codomain."
            ))
        self.domain = domain
        r"""The vector space on which the operator is defined. Either a
        subclass of `regpy.vecsps.VectorSpaceBase` or `None`."""
        self.codomain = codomain
        r"""The vector space on which the operator values are defined. Either
        a subclass of `regpy.vecsps.VectorSpaceBase` or `None`."""
        self.linear = linear
        r"""Boolean indicating whether the operator is linear."""
        self._self_adjoint = bool(linear and self_adjoint)
        self._derivative_self_adjoint = self_adjoint
        self._gram_factor = 1.
        self._constants = {}
        """A dictionary containing constants set to inputs by the `set_constant`"""        
        self._consts = {'domain', 'codomain','lu'}
        r"""properties that are handled differently when copying the operator."""
        if inverse is not None and not isinstance(inverse, Operator):
            raise TypeError(util.Errors._compose_message("INVERSE NOT EXISTENT","The inverse has to be an Operator instance or None."))
        self._inverse = inverse

        self.reset_detect_loop()

    @property
    def self_adjoint(self) -> bool:
        r"""Whether this operator is linear and equal to its adjoint.

        This property is always false for nonlinear operators. If
        ``self_adjoint=True`` was passed when constructing a nonlinear
        operator, the flag applies to the derivatives returned by
        :meth:`linearize` instead.
        """
        return self._self_adjoint

    def declare_as_self_adjoint(self, check: bool = True) -> None:
        r"""Declare an existing linear operator to be self-adjoint.

        This method is intended for operators assembled from other operators
        in ways for which the operator algebra cannot infer
        self-adjointness. After a successful declaration,
        :attr:`self_adjoint` is true and :attr:`adjoint` returns this operator
        itself. Any separately implemented adjoint methods are consequently
        ignored by future access through the public :attr:`adjoint` property.

        If ``check`` is true, the declaration is made only if evaluation and
        adjoint evaluation agree, up to a dtype-dependent rounding tolerance,
        on one random test vector. This is a useful safeguard but not a proof
        of self-adjointness. 

        Parameters
        ----------
        check : bool, optional
            Whether to compare evaluation and adjoint evaluation on a random
            vector before changing the flag. Defaults to ``True``.

        Raises
        ------
        TypeError
            If ``check`` is not boolean.
        ValueError
            If the operator is nonlinear, domain and codomain differ, or the
            numerical check fails.
        NotImplementedError
            If ``check`` is true but adjoint evaluation is unavailable.
        """
        if not isinstance(check, bool):
            raise TypeError(util.Errors.type_error(
                "check must be a boolean.", check
            ))
        if not self.linear:
            raise ValueError(util.Errors.value_error(
                "Only a linear operator can be declared self-adjoint.",
                self,
                "declare_as_self_adjoint",
            ))
        if self.domain != self.codomain:
            raise ValueError(util.Errors.value_error(
                "A self-adjoint operator must have identical domain and "
                "codomain.",
                self,
                "declare_as_self_adjoint",
            ))

        if check:
            # Keep the old adjoint proxy: setting the flag first would make
            # the check compare the operator with itself and hence vacuous.
            old_adjoint = self.adjoint
            test_vector = self.domain.randn()
            value = self(test_vector)
            try:
                adjoint_value = old_adjoint(test_vector)
            except NotImplementedError as exc:
                raise NotImplementedError(util.Errors._compose_message(
                    "ADJOINT NOT AVAILABLE",
                    "The numerical self-adjointness check requires an "
                    "implemented adjoint. Use check=False only if "
                    "self-adjointness is known independently."
                )) from exc

            error = self.domain.norm(value - adjoint_value)
            scale = max(
                1.,
                self.domain.norm(value),
                self.domain.norm(adjoint_value),
            )
            try:
                dtype = np.empty(
                    (), dtype=getattr(self.domain, "dtype", float)
                ).real.dtype
                eps = np.finfo(dtype).eps
            except (TypeError, ValueError):
                eps = np.finfo(float).eps
            tolerance = 1000 * eps * scale

            if not np.isfinite(error) or error > tolerance:
                raise ValueError(util.Errors.value_error(
                    "The numerical self-adjointness check failed: "
                    f"||A x - A* x|| = {error}, allowed tolerance = "
                    f"{tolerance}.",
                    self,
                    "declare_as_self_adjoint",
                ))

        self._self_adjoint = True
        self._derivative_self_adjoint = True
        # ``adjoint`` is memoized. It was necessarily accessed by the check
        # and may also have been accessed earlier by user code. Deleting the
        # cached value makes the next access observe the new flag and return
        # ``self``.
        del self.adjoint

    def __deepcopy__(self, memo: dict[int, Any]) -> Operator:
        cls = type(self)
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k.startswith("__memoized_"):
                continue            
            if k in self._consts:
                setattr(result, k, v)
            else:
                setattr(result, k, deepcopy(v, memo))
        return result

    @property
    def attrs(self) -> set[str]:
        r"""The set of all instance attributes. Useful for updating the `_consts` attribute via

            self._consts.update(self.attrs)

        to declare every current attribute as constant for deep copies.
        """
        return set(self.__dict__)

    @property
    def gram_factor(self) -> float:
        r"""Scalar factor representing the codomain Gram operator.

        Codomain-free evaluations use this factor in place of an explicitly
        applied codomain Gram operator. The default is ``1.``.
        """
        return self._gram_factor

    @gram_factor.setter
    def gram_factor(self, value: float) -> None:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise TypeError(util.Errors.type_error(
                f"gram_factor must be a real scalar. Got {value!r}."
            ))
        if value < 0:
            raise ValueError(util.Errors.value_error(
                f"gram_factor must be non-negative. Got {value!r}."
            ))
        self._gram_factor = float(value)

    def set_gram_factor(self, gram: Operator | HilbertSpace) -> None:
        r"""Set :attr:`gram_factor` from a scalar Gram operator.

        Parameters
        ----------
        gram : Operator or HilbertSpace
            A Hilbert space, or its Gram operator. Supported Gram operators
            are :class:`Identity`, a :class:`LinearCombination` consisting of
            one multiple of an identity, and :class:`PtwMultiplication` with a
            constant multiplier.

        Raises
        ------
        TypeError
            If ``gram`` is neither an operator nor a Hilbert space, or if the
            extracted factor is not real.
        ValueError
            If the Gram operator does not act on this operator's codomain, is
            not one of the supported forms, or yields a negative factor.
        """
        # Importing HilbertSpace at module level would cause a circular import:
        # regpy.hilbert imports the operator classes used for Gram operators.
        from regpy.hilbert import HilbertSpace

        if isinstance(gram, HilbertSpace):
            gram = gram.gram
        if not isinstance(gram, Operator):
            raise TypeError(util.Errors.type_error(
                "gram must be a HilbertSpace or an Operator. "
                f"Got {type(gram).__name__}."
            ))
        if gram.domain != self.codomain or gram.codomain != self.codomain:
            raise ValueError(util.Errors.value_error(
                "The Gram operator's domain and codomain must both equal "
                f"the operator codomain {self.codomain}. Got domain "
                f"{gram.domain} and codomain {gram.codomain}."
            ))

        if isinstance(gram, Identity):
            factor = 1.
        elif isinstance(gram, LinearCombination):
            if len(gram.ops) != 1 or not isinstance(gram.ops[0], Identity):
                raise ValueError(util.Errors.value_error(
                    "A LinearCombination used as a Gram operator must contain "
                    "exactly one Identity operator."
                ))
            factor = gram.coeffs[0]
        elif isinstance(gram, PtwMultiplication):
            ones = gram.domain.ones()
            normalizer = gram.domain.vdot(ones, ones)
            if normalizer == 0:
                raise ValueError(util.Errors.value_error(
                    "A PtwMultiplication used as a Gram operator must have a "
                    "constant multiplier."
                ))
            factor = gram.domain.vdot(gram.factor, ones) / normalizer
            residual = gram.factor - factor * ones
            residual_norm_sq = gram.domain.vdot(residual, residual).real
            factor_norm_sq = gram.domain.vdot(
                gram.factor, gram.factor
            ).real
            tolerance = (
                100 * np.finfo(float).eps * max(1., abs(factor_norm_sq))
            )
            if residual_norm_sq > tolerance:
                raise ValueError(util.Errors.value_error(
                    "A PtwMultiplication used as a Gram operator must have a "
                    "constant multiplier."
                ))
        else:
            raise ValueError(util.Errors.value_error(
                "Unsupported Gram operator. Expected Identity, a "
                "LinearCombination containing one Identity, or "
                "PtwMultiplication with a constant multiplier. Got "
                f"{type(gram).__name__}."
            ))

        factor = np.real_if_close(factor)
        if np.ndim(factor) == 0:
            factor = factor.item()
        self.gram_factor = factor

    def _apply_gram_factor(self, vector: Vector) -> Vector:
        """Scale a codomain-free result by :attr:`gram_factor`."""
        if self.gram_factor != 1.:
            vector *= self.gram_factor
        return vector

    def reset_detect_loop(self) -> None:
        self._detect_loop = [False]*6  

    def __call__(self, x: Vector, out: Vector | None = None, **kwargs) -> Vector:
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
            self.reset_detect_loop()
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
            

    def linearize(
        self,
        x: Vector,
        out: Vector | list[Vector | Operator | None] | None = None,
        return_adjoint_eval: bool = False,
        **kwargs,
    ) -> tuple[Vector, Operator]:
        r"""Linearize the operator around some point.

        Parameters
        ----------
        x : Vector
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
          Vector, Derivative:
             The value and the derivative at `x`, the latter as an `Operator` instance.
        if return_adjoint_eval ==True: 
           Vector, Derivative
               Vector is :math:`F'[x]^\ast F(x)`, Derivative is as above. The adjoint derivative
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
                    kwargs.setdefault("differentiate", True)
                    y = self._eval(self._insert_constants(x), **kwargs)
                    if y not in self.codomain:
                        raise RuntimeError(util.Errors.not_in_vecsp(
                            y,
                            self.codomain
                            ))
                    deriv = Derivative(self.__get_handle())
                    self.reset_detect_loop()
                    return y, deriv
                else:
                    Fstar_y = self._adjoint_eval(
                        self._insert_constants(x), **kwargs
                    )
                    self._apply_gram_factor(Fstar_y)
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
                    self._apply_gram_factor(out[0])
                    out[1] = Derivative(self.__get_handle()) 
            self.reset_detect_loop()
            return out[0],out[1]

    @util.memoized_property
    def adjoint(self) -> Operator:
        r"""For linear operators, this is the adjoint as a linear `regpy.operators.Operator`
        instance. For a self-adjoint operator, this property returns the
        operator itself, so calling ``op.adjoint(x)`` is identical to calling
        ``op(x)``. Otherwise, an :class:`Adjoint` proxy is computed on demand
        and saved for subsequent invocations.

        Returns
        -------
        Operator
            The operator itself if :attr:`self_adjoint` is true; otherwise an
            :class:`Adjoint` instance.
        """
        if not self.linear:
            raise RuntimeError(util.Errors.not_linear_op(
                self,
                "To construct an adjoint the Operator has to be linear!"
                ))
        if self.self_adjoint:
            return self
        return Adjoint(self)
    
    @util.memoized_property
    def adjoint_eval(self) -> Operator:
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

    def __revoke(self) -> None:
        try:
            self.__handle = _Revocable.take(self.__handle)
        except AttributeError:
            pass

    def __get_handle(self) -> _Revocable:
        try:
            return self.__handle
        except AttributeError:
            self.__handle = _Revocable(self)
            return self.__handle

    def _eval(self, x: Vector, **kwargs) -> Vector:
        self.log.debug(f"eval: {self._detect_loop}")
        if self._detect_loop[0]:
            raise NotImplementedError(util.Errors._compose_message(
                "NOT DEFINED METHOD",
                "By default the method _eval or _ieval are not implemented for an Operator!\n You as a user have to define it!"
            ))
        self._detect_loop[0] = True
        out = self.codomain.zeros()
        return self._ieval(x, out, **kwargs)
    
    def _ieval(self, x: Vector, out: Vector, **kwargs) -> Vector:
        r"""In place/ out evaluation. A method that ideally provides an in place evaluation into the specified out
        value. The default implementation uses _eval. Note: If eval is not defined either, then it will raise an error.

        Parameters
        ----------
        x : Vector
            The point at which to evaluate in the domain.
        out : Vector
            The output vector in which to write the result.
        differentiate : bool, optional
            Determines if precomputation for derivate is needed, by default False
        """
        self.log.debug(f"eval: {self._detect_loop}")
        if self._detect_loop[0]:
            raise NotImplementedError(util.Errors._compose_message(
                "NOT DEFINED METHOD",
                "By default the method _eval or _ieval are not implemented for an Operator!\n You as a user have to define it!"
            ))
        self._detect_loop[0] = True
        if out is x:
            source = self.domain.empty()
            self.domain.copyto(source, x)
            x = source
        self.codomain.copyto(out, self._eval(x, **kwargs))
        return out


    def _derivative(self, x: Vector, **kwargs) -> Vector:
        if self._detect_loop[1]:
            raise NotImplementedError(util.Errors._compose_message(
                "NOT DEFINED METHOD",
                "By default, the method _derivative or _iderivative are not implemented for an Operator!\n You as a user have to define it!"
            ))
        self.log.debug(f"deriv: {self._detect_loop}")
        self._detect_loop[1] = True
        out = self.codomain.zeros()
        return self._iderivative(x, out, **kwargs)
    
    def _iderivative(self, x: Vector, out: Vector, **kwargs) -> Vector:
        r"""In place/ out evaluation of derivative. A method that ideally provides an in place evaluation into the 
        specified out value of the derivative. The default implementation uses _derivative. Note: If _derivative is 
        not defined either, then it will raise an error.

        Parameters
        ----------
        x : Vector
            The point at which to evaluate in the domain.
        out : Vector
            The output vector in which to write the result.
        differentiate : bool, optional
            Determines if precomputation for derivate is needed, by default False
        """
        if self._detect_loop[1]:
            raise NotImplementedError(util.Errors._compose_message(
                "NOT DEFINED METHOD",
                "By default, the method _derivative or _iderivative are not implemented for an Operator!\n You as a user have to define it!"
            ))
        self._detect_loop[1] = True
        if out is x:
            source = self.domain.empty()
            self.domain.copyto(source, x)
            x = source
        self.codomain.copyto(out, self._derivative(x, **kwargs))
        return out

    def _adjoint(self, y: Vector, **kwargs) -> Vector:
        if self.self_adjoint:
            self.reset_detect_loop()
            return self._eval(y, **kwargs)
        if self._detect_loop[2]:
            raise NotImplementedError(util.Errors._compose_message(
                "NOT DEFINED METHOD",
                "By default, the method _adjoint or _iadjoint are not implemented for an Operator!\n You as a user have to define it!"
            ))
        self._detect_loop[2] = True
        out = self.domain.zeros()
        return self._iadjoint(y, out, **kwargs)
    
    def _iadjoint(self, y: Vector, out: Vector, **kwargs) -> Vector:
        if self.self_adjoint:
            self.reset_detect_loop()
            return self._ieval(y, out, **kwargs)
        if self._detect_loop[2]:
            raise NotImplementedError(util.Errors._compose_message(
                "NOT DEFINED METHOD",
                "By default the method _adjoint or _iadjoint are not implemented for an Operator!\n You as a user have to define it!"
            ))
        self._detect_loop[2] = True
        if out is y:
            source = self.codomain.empty()
            self.codomain.copyto(source, y)
            y = source
        self.domain.copyto(out, self._adjoint(y, **kwargs))
        return out

    def _adjoint_data(self, data: ObsDataT, **kwargs) -> Vector:
        if self._detect_loop[3]:
            raise NotImplementedError(util.Errors._compose_message(
                "NOT DEFINED METHOD",
                "By default, the method _adjoint_data or _iadjoint_data are not implemented for an Operator!\n You as a user have to define it!"
            ))
        self.log.debug(f"adjdata: {self._detect_loop}")
        self._detect_loop[3] = True
        out = self.domain.zeros()
        return self._iadjoint_data(data, out, **kwargs)
    
    def _iadjoint_data(self, data: ObsDataT, out: Vector, **kwargs) -> Vector:
        if self._detect_loop[3]:
            raise NotImplementedError(util.Errors._compose_message(
                "NOT DEFINED METHOD",
                "By default, the method _adjoint_data or _iadjoint_data are not implemented for an Operator!\n You as a user have to define it!"
            ))
        self.log.debug(f"adjdata: {self._detect_loop}")
        self._detect_loop[3] = True
        if out is data:
            source = self.codomain.empty()
            self.codomain.copyto(source, data)
            data = source
        self.domain.copyto(out, self._adjoint_data(data, **kwargs))
        return out
        
    def adjoint_data(
        self, data: ObsDataT, out: Vector | None = None, **kwargs
    ) -> Vector:
        self.reset_detect_loop()
        try:
            if out is None:
                try:
                    result = self._adjoint_data(data, **kwargs)
                except NotImplementedError:
                    self.reset_detect_loop()
                    result = self._adjoint(data, **kwargs)
            else:
                if out not in self.domain:
                    raise ValueError(...)

                try:
                    result = self._iadjoint_data(data, out, **kwargs)
                except NotImplementedError:
                    self.reset_detect_loop()
                    result = self._iadjoint(data, out, **kwargs)
            return self._apply_gram_factor(result)
        finally:
            self.reset_detect_loop()
    
    def _adjoint_eval(self, x: Vector, **kwargs) -> Vector:
        if self._detect_loop[4]:
            if not hasattr(self, '_warning_adjoint_eval_issued') or self._warning_adjoint_eval_issued == False:
                self.log.warning("The default implementation of  _adjoint_eval is not optimized to exclude constructing elements in the codomain!")
                self._warning_adjoint_eval_issued = True
            if len(kwargs) != 0:
                self.log.warning(f"In the default implementation of _adjoint_eval it is unclear where to put keyword arguments. Maybe implement your own _adjoint_eval and process the kwargs. Thus ignoring yours: kwargs= {kwargs}")
            if self.linear:
                return self.adjoint(self(x))
            else:
                y,deriv = self.linearize(x)
                return deriv.adjoint(y)
        else:
            self._detect_loop[4] = True
            out = self.domain.zeros()
            return self._iadjoint_eval(x, out, **kwargs)
    
    def _iadjoint_eval(self, x: Vector, out: Vector, **kwargs) -> Vector:
        if self._detect_loop[4]:
            if not hasattr(self, '_warning_adjoint_eval_issued') or self._warning_adjoint_eval_issued == False:
                self.log.warning("The default implementation of _iadjoint_eval is not optimized to exclude constructing elements in the codomain!")
                self._warning_adjoint_eval_issued = True
            if len(kwargs) != 0:
                self.log.warning(f"In the default implementation of _adjoint_eval it is unclear where to put keyword arguments. Maybe implement your own _adjoint_eval and process the kwargs. Thus ignoring yours: kwargs= {kwargs}")
            if self.linear:
                return self.adjoint(self(x), out = out)
            else:
                y, deriv = self.linearize(x)
                return deriv.adjoint(y, out = out)
        else:
            self._detect_loop[4] = True
            if out is x:
                source = self.domain.empty()
                self.domain.copyto(source, x)
                x = source
            self.domain.copyto(out, self._adjoint_eval(x, **kwargs))        
            return out
            

    def _adjoint_derivative(self, x: Vector, **kwargs) -> Vector:
        if self._detect_loop[5]:
            if not hasattr(self, '_warning_adjoint_derivative_issued') or self._warning_adjoint_derivative_issued == False:
                self.log.warning(f"The default implementation of _adjoint_derivative is not optimised to exclude constructing elements in the codomain!")
                self._warning_adjoint_derivative_issued = True
            if len(kwargs) != 0:
                self.log.warning(f"In the default implementation of _adjoint_derivative it is unclear where to put keyword arguments.Maybe implement your own _adjoint_derivative and process the kwargs Thus ignoring yours: kwargs= {kwargs}")
            if self.linear:
                return self._adjoint(self._eval(x))
            elif self._derivative_self_adjoint:
                y = self._derivative(x)
                self.reset_detect_loop()
                return self._derivative(y)
            else:
                return self._adjoint(self._derivative(x))
        else:
            self._detect_loop[5] = True
            out = self.domain.zeros()
            return self._iadjoint_derivative(x, out, **kwargs)
        
    def _iadjoint_derivative(self, x: Vector, out: Vector, **kwargs) -> Vector:
        if self._detect_loop[5]:
            if not hasattr(self, '_warning_adjoint_derivative_issued') or self._warning_adjoint_derivative_issued == False:
                self.log.warning(f"The default implementation of _iadjoint_derivative is not optimised to exclude constructing elements in the codomain!")
                self._warning_adjoint_derivative_issued = True
            if len(kwargs) != 0:
                self.log.warning(f"In the default implementation of _iadjoint_derivative it is unclear where to put keyword arguments. Maybe implement your own _iadjoint_derivative and process the kwargs Thus ignoring yours: kwargs= {kwargs}")
            if self.linear:
                return self._iadjoint(self._eval(x), out)
            elif self._derivative_self_adjoint:
                y = self._derivative(x)
                self.reset_detect_loop()
                return self._iderivative(y, out)
            else:
                return self._iadjoint(self._derivative(x), out)
        else:
            self._detect_loop[5] = True
            if out is x:
                source = self.domain.empty()
                self.domain.copyto(source, x)
                x = source
            self.domain.copyto(out, self._adjoint_derivative(x, **kwargs))
            return out

    @property
    def inverse(self) -> Operator:
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
    def inverse(self, inv: Operator) -> None:
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
    def invertible(self) -> bool:
        return self._inverse is not None

    def as_linear_operator(self) -> SciPyLinearOperator:
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
        
    def norm(
        self,
        h_domain: HilbertSpace | None = None,
        h_codomain: HilbertSpace | None = None,
        method: str | None = None,
        codomain_free: bool = False,
        reltol: float = 1e-3,
        max_op_eval: int | None = None,
    ) -> float:
        r"""Approximate the norm of a linear operator between Hilbert spaces.

        This method always interprets the operator as a map from
        ``h_domain`` to ``h_codomain``. Its norm is computed from the largest
        eigenvalue of :math:`G_X^{-1}T^*G_YT`, followed by a square root. This also
        applies when the operator is labelled self-adjoint and the two Hilbert
        spaces are identical.

        If ``codomain_free`` is true, the same norm is computed without
        constructing codomain vectors: the self-adjoint normal operator
        :attr:`adjoint_eval` is passed to :meth:`norm_sa`, and the square root
        of the resulting value is returned.
 
        Parameters
        ----------
        h_domain: Hilbert space on the domain. Defaults to L2 if None.
        h_codomain: Hilbert space on the codomain. Defaults to L2 if None.
            Not used if ``codomain_free`` is true.
        method: string [default: None]
            'lanczos' uses the capped, fully reorthogonalized implementation;
            'eigs' uses SciPy/ARPACK with the existing breakdown fallback.
            Method by which an approximation of the operator norm is computed. If None uses self.default_norm_method or 'lanczos'
              if this is not set. Alternatives: "power" for power iteration; "eigs" for the SciPy implementation.
        codomain_free: bool [default: False]
            If true, the method avoids any use of vectors in the image space of the operator by using 
            :attr:`adjoint_eval` and :meth:`norm_sa`.

        reltol: float [default: 1e-3]
            Positive, finite relative residual tolerance (default 1e-3).
            The custom Lanczos method stops when its selected Ritz residual
            is small relative to its Ritz value. At the evaluation cap it
            returns its best current estimate even if this criterion fails.
            SciPy uses its eigenvalue tolerance; power uses a relative
            residual. None of these is a certified norm error bound.
        max_op_eval: int or None, optional
            Budget of forward/adjoint-equivalent evaluations. A normal
            application costs two; an ordinary self-adjoint application one.
            Gram operations are excluded. Unused odd budget units are ignored
            for normal operators. Defaults to 100 evaluations when omitted.
            Must allow at least one application. An explicit budget is not
            supported by method='eigs' and raises ValueError.

        Returns
        -------
        scalar
            Approximation of the norm from ``h_domain`` to ``h_codomain``.

        Raises
        ------
        NotImplementedError
            If the operator is nonlinear or the method is not implemented.
        """

        if isinstance(reltol, (bool, np.bool_)) or not np.isscalar(reltol) or not np.isrealobj(reltol):
            raise ValueError("reltol must be a positive, finite real scalar.")
        try:
            reltol = float(reltol)
        except (TypeError, ValueError) as error:
            raise ValueError("reltol must be a positive, finite real scalar.") from error
        if not np.isfinite(reltol) or reltol <= 0:
            raise ValueError("reltol must be a positive, finite real scalar.")

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
        method=getattr(self,'default_norm_method','lanczos') if method is None else method
        self._validate_norm_budget(method, max_op_eval)
        if codomain_free:
            return sqrt(self.adjoint_eval.norm_sa(
                h_domain=h_domain,
                method=method,
                reltol=reltol,
                **({} if max_op_eval is None else {"max_op_eval": max_op_eval}),
            ))
        if(h_codomain is None):
            h_codomain=L2(self.codomain)
        elif h_codomain.vecsp != self.codomain:
            raise ValueError(util.Errors.not_equal(
                h_codomain.vecsp,
                self.codomain,
                add_info=f"Trying to compute the norm of the operator {self} with a given Hilbert space {h_codomain} on codomain."))
        if method == "power":
            return self._power_method(h_domain, h_codomain, stopping_rule=reltol,
                                      max_iter=self._norm_budget_steps(max_op_eval, 2))
        elif method in ("lanczos", "eigs"):
            op = self.adjoint * h_codomain.gram * self
            if method == "lanczos":
                eigenvalue = op._capped_lanczos(
                    h_domain, reltol, self._norm_budget_steps(max_op_eval, 2), 'LA'
                )
            else:
                eigenvalue = op._unshifted_norm_eigenvalue(h_domain, reltol, 'LA')
            return sqrt(max(0., eigenvalue))
        else:
            raise NotImplementedError(util.Errors._compose_message(
                "NOT DEFINED METHOD",
                f"The method {method} is unknown to compute the norm of an Operator {self}!"
            ))

    def norm_sa(
        self,
        h_domain: HilbertSpace | None = None,
        method: str | None = None,
        reltol: float = 1e-3,
        max_op_eval: int | None = None,
    ) -> float:
        r"""Approximate the norm of a self-adjoint map into a dual space.

        The operator must be linear and explicitly labelled self-adjoint. It
        is interpreted as a map :math:`A:X\to X^*`, where ``h_domain`` is the
        Hilbert space :math:`X`. The method computes the largest absolute
        generalized eigenvalue of

        .. math::
            A x = \lambda G_X x,

        and hence the norm of :math:`G_X^{-1}A:X\to X`.

        This interpretation differs from :meth:`norm`, even if that method is
        called for a self-adjoint operator with identical domain and codomain
        Hilbert spaces. In that case :meth:`norm` treats the operator as a map
        :math:`X\to X`, whereas this method treats its values as elements of
        :math:`X^*`.

        Parameters
        ----------
        h_domain: Hilbert space on the domain. Defaults to L2 if None.
        method: string [default: None]
            'lanczos' uses the capped, fully reorthogonalized implementation;
            'eigs' uses SciPy/ARPACK with the existing breakdown fallback.
            Method by which an approximation of the operator norm is
            computed. If None, uses ``self.default_norm_method`` or
            ``'lanczos'`` if this is not set. Alternative: ``'power'`` for
            power iteration; ``'eigs'`` retains the SciPy implementation.

        reltol: float [default: 1e-3]
            Positive, finite relative residual tolerance (default 1e-3).
            The custom Lanczos method stops when its selected Ritz residual
            is small relative to its Ritz value. At the evaluation cap it
            returns its best current estimate even if this criterion fails.
            SciPy uses its eigenvalue tolerance; power uses a relative
            residual. None of these is a certified norm error bound.
        max_op_eval: int or None, optional
            Budget of forward/adjoint-equivalent evaluations. A normal
            application costs two; an ordinary self-adjoint application one.
            Gram operations are excluded. Unused odd budget units are ignored
            for normal operators. Defaults to 100 evaluations when omitted.
            Must allow at least one application. An explicit budget is not
            supported by method='eigs' and raises ValueError.

        Returns
        -------
        scalar
            Approximation of the norm of :math:`G_X^{-1}A`.

        Raises
        ------
        NotImplementedError
            If the operator is nonlinear or the method is not implemented.
        ValueError
            If the operator is not labelled self-adjoint or ``h_domain`` does
            not belong to its domain.
        """
        if isinstance(reltol, (bool, np.bool_)) or not np.isscalar(reltol) or not np.isrealobj(reltol):
            raise ValueError("reltol must be a positive, finite real scalar.")
        try:
            reltol = float(reltol)
        except (TypeError, ValueError) as error:
            raise ValueError("reltol must be a positive, finite real scalar.") from error
        if not np.isfinite(reltol) or reltol <= 0:
            raise ValueError("reltol must be a positive, finite real scalar.")

        if not self.linear:
            raise NotImplementedError(util.Errors.not_linear_op(
                self,
                f"To compute the norm of the Operator {self} it has to be linear!"
            ))
        if not self.self_adjoint:
            raise ValueError(util.Errors.value_error(
                "norm_sa requires an operator labelled as self-adjoint.",
                self,
                "norm_sa",
            ))

        from regpy.hilbert import L2
        if h_domain is None:
            h_domain = L2(self.domain)
        elif h_domain.vecsp != self.domain:
            raise ValueError(util.Errors.not_equal(
                h_domain.vecsp,
                self.domain,
                add_info=f"Trying to compute the self-adjoint norm of the operator {self} \n with a given Hilbert space {h_domain} on domain."
            ))

        method = getattr(self, 'default_norm_method', 'lanczos') if method is None else method
        self._validate_norm_budget(method, max_op_eval)
        if method == "power":
            return self._power_method_sa(h_domain, stopping_rule=reltol,
                                         max_iter=self._norm_budget_steps(max_op_eval,
                                             2 if isinstance(self, AdjointEval) else 1))
        elif method == "lanczos":
            cost = 2 if isinstance(self, AdjointEval) else 1
            return abs(self._capped_lanczos(
                h_domain, reltol, self._norm_budget_steps(max_op_eval, cost), 'LM'
            ))
        elif method == "eigs":
            return abs(self._unshifted_norm_eigenvalue(h_domain, reltol, 'LM'))
        else:
            raise NotImplementedError(util.Errors._compose_message(
                "NOT DEFINED METHOD",
                f"The method {method} is unknown to compute the self-adjoint norm of an Operator {self}!"
            ))

    @staticmethod
    def _validate_norm_budget(method, max_op_eval):
        if max_op_eval is not None:
            if isinstance(max_op_eval, (bool, np.bool_)) or not isinstance(max_op_eval, Integral):
                raise TypeError("max_op_eval must be a positive integer or None.")
            if max_op_eval <= 0:
                raise ValueError("max_op_eval must be positive.")
            if method == 'eigs':
                raise ValueError("method='eigs' does not support max_op_eval.")

    @staticmethod
    def _norm_budget_steps(max_op_eval, cost):
        budget = 100 if max_op_eval is None else int(max_op_eval)
        if budget < cost:
            raise ValueError(f"max_op_eval must be at least {cost} for this norm.")
        return budget // cost

    def _capped_lanczos(self, h_domain, reltol, max_steps, which):
        """Unshifted Hilbert-space Lanczos with full reorthogonalization.

        Each loop performs exactly one application of self. A null initial
        direction is retried within the same budget. Invariant-subspace
        breakdown terminates normally. As with other Krylov estimators, a
        starting vector can miss a dominant eigenspace; estimates are not
        certified upper bounds. Memory use grows with the Krylov dimension.
        """
        op = SciPyLinearOperator(self)
        gram = SciPyLinearOperator(h_domain.gram)
        gram_inv = SciPyLinearOperator(h_domain.gram_inv)
        n = op.shape[0]
        if n == 0:
            return 0.
        def inner(x, y):
            return np.vdot(x, gram @ y)
        def norm(x):
            return sqrt(max(0., float(inner(x, x).real)))
        def random_unit():
            x = self.domain.flatten(self.domain.randn()).copy()
            length = norm(x)
            if not np.isfinite(length) or length == 0:
                raise ValueError("Lanczos requires a finite nonzero starting vector.")
            return x / length
        q = random_unit()
        basis, diagonal, offdiagonal = [], [], []
        previous = None
        beta_old = 0.
        estimate = 0.
        eps = np.finfo(q.real.dtype).eps
        for step in range(max_steps):
            z = gram_inv @ (op @ q)
            if not np.all(np.isfinite(z)):
                raise ValueError("Lanczos produced a nonfinite operator image.")
            image_norm = norm(z)
            if image_norm == 0 and not basis:
                if step + 1 < min(2, max_steps):
                    q = random_unit()
                    continue
                return 0.
            alpha = float(inner(q, z).real)
            basis.append(q.copy())
            diagonal.append(alpha)
            z = z - alpha * q
            if previous is not None:
                z -= beta_old * previous
            # Two passes suppress loss of orthogonality and spurious Ritz values.
            for _ in range(2):
                for v in basis:
                    z -= inner(v, z) * v
            beta = norm(z)
            matrix = np.diag(diagonal)
            if offdiagonal:
                matrix += np.diag(offdiagonal, 1) + np.diag(offdiagonal, -1)
            values, vectors = np.linalg.eigh(matrix)
            index = len(values) - 1 if which == 'LA' else int(np.argmax(abs(values)))
            estimate = float(values[index])
            residual = beta * abs(vectors[-1, index])
            if (beta <= 100 * eps * image_norm or len(basis) == n
                    or (estimate != 0 and residual <= reltol * abs(estimate))):
                return estimate
            offdiagonal.append(beta)
            previous, q = q, z / beta
            beta_old = beta
        return estimate

    def _unshifted_norm_eigenvalue(self, h_domain, reltol, which):
        """Estimate an extremal generalized eigenvalue without a shift.

        Two zero probe images are treated as a zero operator. No absolute
        threshold is used, so small nonzero operators are not deliberately
        rounded to zero. ARPACK factorization breakdown (-9999), which can
        also occur for nonzero singular operators, falls back to power
        iteration on the squared Hilbert-space operator. Other failures
        propagate. Here 'LA' is used only for positive semidefinite operators;
        for 'LM' only the absolute value of the result is used by the caller.
        """
        from scipy.sparse.linalg import eigsh, ArpackError

        op = SciPyLinearOperator(self)
        gram = SciPyLinearOperator(h_domain.gram)
        # ARPACK requires k < n (and complex eigsh requires k < n - 1).
        if op.shape[0] <= 2:
            from scipy.linalg import eigvalsh
            basis = np.eye(op.shape[0], dtype=op.dtype)
            matrix = np.column_stack([op @ e for e in basis.T])
            metric = np.column_stack([gram @ e for e in basis.T])
            values = eigvalsh(matrix, metric)
            return float(values[-1] if which == 'LA'
                         else values[np.argmax(np.abs(values))])

        for _ in range(2):
            v = self.domain.randn()
            v_norm = h_domain.norm(v)
            if v_norm == 0:
                raise ValueError("Norm estimation requires a nonzero random probe.")
            v = v / v_norm
            v0 = self.domain.flatten(v)
            image = op @ v0
            if not np.all(np.isfinite(image)):
                raise ValueError("Norm estimation produced a nonfinite probe image.")
            if np.any(image != 0):
                break
        else:
            return 0.

        try:
            return float(eigsh(
                op, k=1, M=gram,
                Minv=SciPyLinearOperator(h_domain.gram_inv),
                which=which, v0=v0, tol=reltol,
                return_eigenvectors=False,
            )[0])
        except ArpackError as error:
            if not str(error).startswith("ARPACK error -9999:"):
                raise
            # A nonzero image does not prevent breakdown when the Krylov
            # space is exhausted, e.g. for projections. Do not classify
            # this as a zero operator or introduce a spectral shift.
            for _ in range(1000):
                y = h_domain.gram_inv(self(v))
                estimate = h_domain.norm(y)
                if not np.isfinite(estimate) or estimate == 0:
                    raise RuntimeError(
                        "Norm fallback encountered a zero or nonfinite image."
                    ) from error
                # Apply B twice, normalizing between applications to avoid
                # explicitly squaring its scale. Squaring also handles
                # eigenvalues +lambda and -lambda of equal magnitude.
                z = h_domain.gram_inv(self(y / estimate))
                residual = h_domain.norm(z - estimate * v) / estimate
                if np.isfinite(residual) and residual <= reltol:
                    return float(estimate)
                z_norm = h_domain.norm(z)
                if not np.isfinite(z_norm) or z_norm == 0:
                    raise RuntimeError(
                        "Norm fallback encountered a zero or nonfinite iterate."
                    ) from error
                v = z / z_norm
            raise RuntimeError(
                "Norm fallback did not converge within 1000 iterations."
            ) from error

    def _power_method(
        self,
        h_domain: HilbertSpace,
        h_codomain: HilbertSpace,
        max_iter: int = int(1e2),
        stopping_rule: float = 1e-3,
    ) -> float:
        r"""Approximate the operator norm by power iteration.

        The operator's normal operator is used and the final square root is
        taken. This method should be called through :meth:`norm`.

        Parameters
        ----------
        h_domain: Hilbert space on the domain.
        h_codomain: Hilbert space on the codomain.
        max_iter: int maximum number of iterations
        stopping_rule: float Iteration is stopped if relative residual is smaller than this value.
        """
        x = self.domain.rand()
        x_norm = h_domain.norm(x)
        if x_norm == 0:
            raise ValueError("Power iteration requires a nonzero initial vector.")
        x = x / x_norm
        relative_residual = inf
        op = self.adjoint * h_codomain.gram * self
        for _ in range(max_iter):
            if relative_residual < stopping_rule:
                break
            ystar = op(x)
            y = h_domain.gram_inv(ystar)
            lmb = sqrt(self.domain.vdot(y, ystar).real)
            if lmb == 0:
                return 0.
            relative_residual = h_domain.norm(y - lmb * x) / lmb
            x = y/lmb
        return sqrt(lmb)

    def _power_method_sa(
        self,
        h_domain: HilbertSpace,
        max_iter: int = int(1e2),
        stopping_rule: float = 1e-3,
    ) -> float:
        r"""Approximate the self-adjoint norm by power iteration.

        The operator is applied directly and yields the norm of
        :math:`G_X^{-1}A`. This method should be called through
        :meth:`norm_sa`.

        Parameters
        ----------
        h_domain: Hilbert space on the domain.
        max_iter: int maximum number of iterations
        stopping_rule: float Iteration is stopped if relative residual is smaller than this value.
        """
        x = self.domain.rand()
        x_norm = h_domain.norm(x)
        if x_norm == 0:
            raise ValueError("Power iteration requires a nonzero initial vector.")
        x = x / x_norm
        relative_residual = inf
        for _ in range(max_iter):
            if relative_residual < stopping_rule:
                break
            ystar = self(x)
            y = h_domain.gram_inv(ystar)
            lmb = sqrt(self.domain.vdot(y, ystar).real)
            if lmb == 0:
                return 0.
            rayleigh = self.domain.vdot(x, ystar).real
            signed_lmb = -lmb if rayleigh < 0 else lmb
            relative_residual = h_domain.norm(y - signed_lmb * x) / lmb
            x = y/lmb
        return lmb
    
    def set_constant(self, c: Vector | Scalar, index: int) -> None:
        """Assuming the operator you defined has a domain that is composed of multiple
        inputs and thus a direct sum, this method allows you to fix one of that inputs
        as a constant. 

        This method changes the domain to either a direct sum of the remaining
        components or just the remaining component.

        Moreover, it validates if the remaining operator is linear using the utility method
        and changes the linearity flag.

        Parameters
        ----------
        c : Vector or Scalar
            The constant vector or scalar to be set.
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
    
    def reset_constants(self) -> None:
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

    def get_constants(self) -> dict[int, Vector | Scalar]:
        """Returns the constants set by `set_constant` as a dictionary. The keys are the indices
        of the full domain and the values are the constants set.
        
        Returns
        -------
        dict
            The dictionary with the constants set by `set_constant`.
        """
        return self._constants

    def _insert_constants(self, x: Vector) -> Vector:
        """Inserts the constants into the vector.

        Parameters
        ----------
        x : Vector
            The vector in the reduced domain that to be completed with constants.

        Returns
        -------
        Vector
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
     
    def _reduce_to_domain(self, y: Vector) -> Vector:
        """Remove the constant coefficients in the vector where one has specified 
        constants in the original operator.

        Parameters
        ----------
        y : Vector
            The vector in the full codomain that to be reduced to the new domain.

        Returns
        -------
        Vector
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
    
    def __mul__(self, other: Scalar | Vector | Operator) -> Operator:
        if np.isscalar(other) and other == 1:
            return self
        elif isinstance(other, Operator):
            return Composition(self, other)
        elif np.isscalar(other) or other in self.domain:
            return self * PtwMultiplication(self.domain, other)
        else:
            return NotImplemented

    def __rmul__(self, other: Scalar | Vector | Operator) -> Operator:
        if np.isscalar(other):
            if other == 1:
                return self
            if isinstance(self, PtwMultiplication):
                return PtwMultiplication(self.domain, other * self.factor)
            return LinearCombination((other, self))         
        elif other in self.codomain:
            return PtwMultiplication(self.codomain, other) * self
        elif isinstance(other, Operator):
            return Composition(other, self) 
        else:
            return NotImplemented
    
    def __imul__(self,other: Scalar | Vector | Operator) -> Operator:
        return self*other

    def __add__(self, other: Scalar | Vector | Operator) -> Operator:
        if np.isscalar(other) and other == 0:
            return self
        elif np.isscalar(other) or other in self.codomain:
            return OuterShift(self, other)
        elif isinstance(other, Operator):
            return LinearCombination(self, other)
        else:
            return NotImplemented

    def __radd__(self, other: Scalar | Vector | Operator) -> Operator:
        return self + other
    
    def __iadd__(self, other: Scalar | Vector | Operator) -> Operator:
        return self + other

    def __sub__(self, other: Scalar | Vector | Operator) -> Operator:
        return self + (-other)

    def __rsub__(self, other: Scalar | Vector | Operator) -> Operator:
        return (-self) + other
    
    def __isub__(self, other: Scalar | Vector | Operator) -> Operator:
        return self - other

    def __neg__(self) -> Operator:
        return (-1) * self

    def __pos__(self) -> Operator:
        return self
    
    def __pow__(self, power : int) -> Operator:
        return Pow(self, power)
    
    def __getitem__(self, val: int | slice) -> PartOfOperator:
        if val is None:
            return self
        return PartOfOperator(self,val)


class Adjoint(Operator):
    r"""A proxy class wrapping a linear operator. Calling it will evaluate the operator's
    adjoint. This class should not be instantiated directly, but rather through the
    `Operator.adjoint` property of a linear operator.

    Parameters
    ----------
    op : Operator
        The base operator giving rise to this adjoint.
    """

    def __init__(self, op: Operator) -> None:
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
        super().__init__(
            op.codomain,
            op.domain,
            linear=True,
            self_adjoint=op.self_adjoint,
        )
        if hasattr(self.op,"full_domain"):
            self._constants = {}
            """The constant inputs of the op need to be set in adjoint evaluation
            """
            self.full_domain = op.codomain

    def _eval(self, x: Vector, **kwargs) -> Vector:
        self.op.reset_detect_loop()
        return self.op._reduce_to_domain(self.op._adjoint(x, **kwargs))
        
    def _ieval(self, x: Vector, out: Vector, **kwargs) -> Vector:
        self.op.reset_detect_loop()
        return self.op._reduce_to_domain(self.op._iadjoint(x, out, **kwargs))

    def _adjoint(self, x: Vector, **kwargs) -> Vector:
        self.op.reset_detect_loop()
        return self.op._eval(self._insert_constants(x), **kwargs)

    def _iadjoint(self, x: Vector, out: Vector, **kwargs) -> Vector:
        self.op.reset_detect_loop()
        return self.op._ieval(self._insert_constants(x), out, **kwargs)

    def _adjoint_eval(self, x: Vector, **kwargs) -> Vector:
        r"""Evaluate ``self.adjoint(self(x))``.

        An operator may provide ``_eval_adjoint`` to implement its opposite
        normal operator, ``op(op.adjoint(x))``, without materializing the
        intermediate vector. This hook is used, for example, by Fourier
        transforms. Operators without such a hook retain the generic
        implementation.
        """
        eval_adjoint = getattr(self.op, '_eval_adjoint', None)
        if eval_adjoint is None:
            return super()._adjoint_eval(x, **kwargs)
        self.op.reset_detect_loop()
        return eval_adjoint(x, **kwargs)

    def _iadjoint_eval(
        self, x: Vector, out: Vector, **kwargs
    ) -> Vector:
        ieval_adjoint = getattr(self.op, '_ieval_adjoint', None)
        if ieval_adjoint is None:
            return super()._iadjoint_eval(x, out, **kwargs)
        self.op.reset_detect_loop()
        return ieval_adjoint(x, out, **kwargs)

    @util.memoized_property
    def adjoint(self) -> Operator:
        return self.op

    @Operator.inverse.getter
    def inverse(self) -> Operator:
        if self._inverse is not None:
            return self._inverse
        try:
            return self.op.inverse.adjoint
        except NotImplementedError:
            raise NotImplementedError(util.Errors._compose_message(
                    "INVERSE NOT DEFINED",
                    "The inverse of the adjoint of operator {} is not known.".format(self.op)))

    def __repr__(self) -> str:
        return util.make_repr(self, self.op)


class Derivative(Operator):
    r"""A proxy class wrapping a nonlinear operator. Calling it will evaluate the operator's
    derivative. This class should not be instantiated directly, but rather through the
    `Operator.linearize` method of a nonlinear operator.

    If the wrapped nonlinear operator was constructed with
    ``self_adjoint=True``, this derivative exposes :attr:`self_adjoint` as true
    and its adjoint evaluation delegates to the wrapped operator's
    ``_derivative`` or ``_iderivative`` implementation. The nonlinear operator
    therefore does not need to implement ``_adjoint`` or ``_iadjoint`` in this
    case.

    Parameters
    ----------
    op : Operator
        The base operator giving rise to this derivative.
    """

    def __init__(self, op: Operator | _Revocable) -> None:
        if not isinstance(op, _Revocable):
            # Wrap plain operators in a _Revocable that will never be revoked to
            # avoid case distinctions below.
            op = _Revocable(op)
        self.op = op
        r"""The underlying operator."""
        _op = op.get()
        """The underlying operator."""
        super().__init__(
            _op.domain,
            _op.codomain,
            linear=True,
            self_adjoint=_op._derivative_self_adjoint,
        )
        self.gram_factor = _op.gram_factor
        # Setting the corresponding constants of op to zero
        if hasattr(_op,"full_domain"):
            self.full_domain = _op.full_domain
            self._constants = {index : self.full_domain[index].zeros() for index in _op._constants}

    def _eval(self, x:Vector, **kwargs) -> Vector:
        self.op.get().reset_detect_loop()
        return self.op.get()._derivative(x, **kwargs)

    def _ieval(self, x: Vector, out: Vector, **kwargs) -> Vector:
        self.op.get().reset_detect_loop()
        return self.op.get()._iderivative(x, out, **kwargs)

    def _adjoint(self, x: Vector, **kwargs) -> Vector:
        self.op.get().reset_detect_loop()
        if self.self_adjoint:
            return self.op.get()._derivative(x, **kwargs)
        return self._reduce_to_domain(self.op.get()._adjoint(x, **kwargs))

    def _iadjoint(self, x: Vector, out: Vector, **kwargs) -> Vector:
        self.op.get().reset_detect_loop()
        if self.self_adjoint:
            return self.op.get()._iderivative(x, out, **kwargs)
        return self._reduce_to_domain(self.op.get()._iadjoint(x, out, **kwargs))
    
    def adjoint_data(self, x: Vector, out: Vector | None  = None, **kwargs) -> Vector:
        if self.self_adjoint:
            return self._apply_gram_factor(self(x, out=out, **kwargs))
        if out is None:
            return self._reduce_to_domain(self.op.get().adjoint_data(x, **kwargs))
        else:
            return self._reduce_to_domain(self.op.get().adjoint_data(x, out = out, **kwargs))
    
    def _adjoint_eval(self, x: Vector, **kwargs) -> Vector:
        self.op.get().reset_detect_loop()
        return self._reduce_to_domain(self.op.get()._adjoint_derivative(x, **kwargs))

    def _iadjoint_eval(self, x: Vector, out: Vector, **kwargs) -> Vector:
        self.op.get().reset_detect_loop()
        return self._reduce_to_domain(self.op.get()._iadjoint_derivative(x, out, **kwargs))

    def __repr__(self) -> str:
        return util.make_repr(self, self.op.get())


class AdjointEval(Operator):
    r"""A proxy class wrapping a linear operator :math:`F`. Calling it will evaluate the
     composition of the operator's adjoint with its evaluation :math:`F^\ast\circ F`. This
     class should not be instantiated directly, but rather through the `Operator.
     adjoint_eval` method of a linear operator.
    The `_eval` and `_adjoint` require the implementation of `_adjoint_eval` note that only 
    one implementation is needed as it is a self-adjoint operator. Its
    :attr:`self_adjoint` property is therefore always true.

    Parameters
    ----------
    op : Operator
        The base operator giving rise to this combination of adjoint and derivative.
    """

    def __init__(self, op: Operator) -> None:
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
        super().__init__(
            op.domain,
            op.domain,
            linear=True,
            self_adjoint=True,
        )
        # Setting the corresponding constants of op to zero
        if hasattr(self.op,"full_domain"):
            self.full_domain = self.op.full_domain
            self._constants = {index : self.full_domain[index].zeros() for index in self.op._constants}

    def _eval(self, x: Vector, **kwargs) -> Vector:
        self.op.reset_detect_loop()
        result = self._reduce_to_domain(
            self.op._adjoint_eval(x, **kwargs)
        )
        return self.op._apply_gram_factor(result)

    def _ieval(self, x: Vector, out: Vector, **kwargs) -> Vector:
        self.op.reset_detect_loop()
        result = self._reduce_to_domain(
            self.op._iadjoint_eval(x, out, **kwargs)
        )
        return self.op._apply_gram_factor(result)

    def _adjoint(self, x: Vector, **kwargs) -> Vector:
        return self._eval(x, **kwargs)

    def _iadjoint(self, x: Vector, out: Vector, **kwargs) -> Vector:
        return self._ieval(x, out, **kwargs)


    def adjoint_data(self, x: Vector, out: Vector | None = None, **kwargs) -> Vector:
        if out is None:
            return self._reduce_to_domain(self.op.adjoint_data(x, **kwargs))
        else:
            return self._reduce_to_domain(self.op.adjoint_data(x, out = out, **kwargs))

    def __repr__(self) -> str:
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

    A linear combination is marked as self-adjoint if every constituent
    operator is self-adjoint and every combined coefficient is real.
    """

    def __init__(self, *args: Operator | tuple[Scalar, Operator]) -> None:
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

        self_adjoint = (
            all(op.self_adjoint for op in self.ops)
            and all(bool(np.isreal(coeff)) for coeff in self.coeffs)
        )
        super().__init__(
            domain,
            codomain,
            linear=all(op.linear for op in self.ops),
            self_adjoint=self_adjoint,
        )

    def _eval(
        self, x: Vector, differentiate: bool = False, **kwargs
    ) -> Vector:
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
    
    def _ieval(
        self, x: Vector, out: Vector, differentiate: bool = False, **kwargs
    ) -> Vector:
        if len(self.ops) == 1:
            coeff = self.coeffs[0]
            op = self.ops[0]
            if differentiate:
                out, deriv = op.linearize(x, out=out, **kwargs)
                self._derivs = [deriv]
            else:
                out = op(x, out=out, **kwargs)
            out *= coeff
            return out
        return super()._ieval(x, out, differentiate=differentiate, **kwargs)

    def _iadjoint_eval(self, x: Vector, out: Vector, **kwargs) -> Vector:
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
    
    def _iadjoint_derivative(self, x: Vector, out: Vector, **kwargs) -> Vector:
        if len(self.ops) == 1:
            out = self._derivs[0].adjoint_eval(x, out = out, **kwargs)
            out *= np.abs(self.coeffs[0])**2
            return out
        self.log.warning("A default fast implementation for an adjoint derivative of a linear combination is only available for a single scalar multiplication! Consider defining your own operator with a faster method!")
        return super()._iadjoint_derivative(x, out,**kwargs)

    def _iderivative(self, x: Vector, out: Vector, **kwargs) -> Vector:
        self.codomain.fill(0.)
        for coeff, deriv in zip(self.coeffs, self._derivs):
            out += coeff * deriv(x,**kwargs)
        return out

    def _iadjoint(self, y: Vector, out: Vector, **kwargs) -> Vector:
        if y is out:
            source = self.codomain.empty()
            self.codomain.copyto(source, y)
            y = source
        if self.linear:
            ops = self.ops
        else:
            ops = self._derivs
        self.domain.fill(out,0)
        for coeff, op in zip(self.coeffs, ops):
            out += np.conjugate(coeff) * op.adjoint(y,**kwargs)
        return out
       
    def _iadjoint_data(self, y: Vector, out: Vector, **kwargs) -> Vector:
        if y is out:
            source = self.codomain.empty()
            self.codomain.copyto(source, y)
            y = source
        self.domain.fill(out,0.)
        for coeff, op in zip(self.coeffs, self.ops):
            out +=  op.adjoint_data(np.conjugate(coeff) * y, **kwargs)
        return out

    @Operator.inverse.getter
    def inverse(self) -> Operator:
        if self._inverse is not None:
            return self._inverse
        if len(self.ops) > 1:
            raise NotImplementedError(util.Errors._compose_message(
                    "INVERSE NOT DEFINED",
                f"The inverse of the linear combination {self} is not defined. Since it was not explicitly defined and automatically computing it with more than one operator is ambiguous. You may specify an explicit inverse by setting self.inverse for this operator."
                ))
        return (1 / self.coeffs[0]) * self.ops[0].inverse

    def __repr__(self) -> str:
        return util.make_repr(self, *zip(self.coeffs, self.ops))

    def __str__(self) -> str:
        reprs = []
        for coeff, op in zip(self.coeffs, self.ops):
            if coeff == 1:
                reprs.append(repr(op))
            else:
                reprs.append('{} * {}'.format(coeff, op))
        return ' + '.join(reprs)


class Composition(Operator):
    r"""A composition of operators. This class should normally not be instantiated directly,
    but rather through multiplying :class:`Operator` instances.

    .. code-block:: python

        op_composed = op_n * ... * op_2 * op_1

    If desired, you can then activate the use of buffers by

        op_composed.use_buffers = True

    Then intermediate values will be written into reusable
    buffers.  There is at most one buffer for every interface between two
    operators.  Adjacent interfaces share a buffer whenever the operator between
    them has compatible domain and codomain, because that operator can then be
    evaluated in place.

    Parameters
    ----------
    *ops : tuple[Operator, ...]
        Operators to be composed. The domain of each operator must be compatible
        with the codomain of the following operator.
    inverse : Operator, optional
        Explicit inverse of the composition.

    Notes
    -----
    A linear composition that explicitly represents a singular-value
    decomposition can be marked with :meth:`label_as_svd`. The public
    properties :attr:`is_svd` and :attr:`svd_data` then expose the verified
    decomposition to spectral algorithms.
    """

    def __init__(
        self,
        *ops: Operator,
        use_buffers: bool = True,
        inverse: Operator | None = None,
    ) -> None:
        if not ops or not isinstance(ops[0], Operator):
            raise ValueError(util.Errors.not_instance(
                ops[0] if ops else None,
                Operator,
                add_info="The first argument of the list of operators is not an Operator."
            ))
        for i, (f, g) in enumerate(zip(ops, ops[1:])):
            if not isinstance(g, Operator):
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
        """The flattened list of composed operators."""
        self.d_eq_cd = []
        """Whether each operator has compatible domain and codomain."""
        for op in ops:
            # A counted composition is an observable evaluation boundary.
            # Flattening it would bypass its instrumented methods (e.g. the
            # blur inside an SSNewton inactive-set operator T * projection).
            if isinstance(op, Composition) and not op.eval_count_enabled:
                self.ops.extend(op.ops)
                self.d_eq_cd.extend(op.d_eq_cd)
            else:
                self.ops.append(op)
                self.d_eq_cd.append(op.domain.compatible(op.codomain))

        self._use_buffers = False
        self._interface_buffers = []
        # One entry for every interface. Entries may reference the same buffer.
        self._buffer_status = {}
        # Status is stored per unique buffer to guard against recursion or
        # concurrent use of one Composition instance.

        super().__init__(
            self.ops[-1].domain,
            self.ops[0].codomain,
            linear=all(op.linear for op in self.ops),
            inverse=inverse,
        )
        self._is_svd = False
        self._svd_data = {}
        if use_buffers:
            self.use_buffers = True

    @property
    def is_svd(self) -> bool:
        r"""Whether this composition has been verified and labelled as an SVD."""
        return self._is_svd

    @property
    def svd_data(self) -> dict[str, Any]:
        r"""Verified singular-value-decomposition data.

        The dictionary is empty until :meth:`label_as_svd` succeeds. It then
        contains ``l``, ``U``, ``V``, ``M``, ``G_X``, ``G_Y`` and
        ``is_eig``. The index ``l`` follows the one-based mathematical
        convention used in the method documentation.
        """
        return self._svd_data

    @staticmethod
    def _svd_gram_operator(
        gram: Operator | Real,
        space: vecsps.VectorSpaceBase,
        name: str,
    ) -> Operator:
        """Normalize a Gram argument to a self-adjoint operator."""
        if isinstance(gram, (bool, np.bool_)):
            raise TypeError(util.Errors.type_error(
                f"{name} must be a positive real number or an Operator."
            ))
        if isinstance(gram, Real):
            if gram <= 0:
                raise ValueError(util.Errors.value_error(
                    f"{name} must be positive. Got {gram!r}."
                ))
            if gram == 1:
                return space.identity
            return PtwMultiplication(space, float(gram))
        if not isinstance(gram, Operator):
            raise TypeError(util.Errors.type_error(
                f"{name} must be a positive real number or an Operator. "
                f"Got {type(gram).__name__}."
            ))
        if gram.domain != space or gram.codomain != space:
            raise ValueError(util.Errors.value_error(
                f"{name} must map {space} to itself. Got domain "
                f"{gram.domain} and codomain {gram.codomain}."
            ))
        if not gram.self_adjoint:
            raise ValueError(util.Errors.value_error(
                f"{name} must be self-adjoint."
            ))
        return gram

    @staticmethod
    def _svd_operator_error(
        first: Operator,
        second: Operator,
        test_vector: Vector,
    ) -> tuple[float, float]:
        first_value = first(test_vector)
        second_value = second(test_vector)
        space = first.codomain
        error = space.norm(first_value - second_value)
        scale = max(
            1., space.norm(first_value), space.norm(second_value)
        )
        return error, scale

    @staticmethod
    def _svd_tolerance(space: vecsps.VectorSpaceBase, scale: float) -> float:
        try:
            dtype = np.empty(
                (), dtype=getattr(space, "dtype", float)
            ).real.dtype
            eps = np.finfo(dtype).eps
        except (TypeError, ValueError):
            eps = np.finfo(float).eps
        return 5000 * eps * max(1., scale)

    def label_as_svd(
        self,
        l: int,
        G_X: Operator | Real = 1.,
        G_Y: Operator | Real = 1.,
        is_eig: bool = False,
    ) -> None:
        r"""Verify and label this composition as a singular-value decomposition.

        If ``self`` represents

        .. math::
            T=A_1A_2\cdots A_k,

        ``l`` identifies the positive pointwise multiplication
        :math:`M=A_l`. The remaining factors define

        .. math::
            V=A_1\cdots A_{l-1},\qquad
            U=A_k^*\cdots A_{l+1}^*,

        so that :math:`T=VMU^*`. On one random reduced-space vector the
        method verifies

        .. math::
            V^*G_YV=I,\qquad U^*G_XU=I.

        This randomized test detects inconsistent declarations but is not a
        proof of the identities.

        Parameters
        ----------
        l : int
            One-based position of the singular-value multiplier. It must lie
            in ``{2, ..., len(self.ops) - 1}``.
        G_X, G_Y : Operator or positive real number [default: 1]
            Domain and codomain Gram operators. A scalar is interpreted as
            that multiple of the corresponding identity operator.
        is_eig : bool [default: False]
            Whether the decomposition is also an eigenvalue decomposition.
            In this case ``U == V`` and ``G_X == G_Y`` are checked as well.

        Raises
        ------
        TypeError
            If an argument has an invalid type.
        ValueError
            If the factorization does not satisfy the structural or numerical
            SVD conditions.
        """
        if isinstance(l, (bool, np.bool_)) or not isinstance(
                l, (int, np.integer)):
            raise TypeError(util.Errors.type_error(
                "l must be an integer."
            ))
        l = int(l)
        if not 2 <= l <= len(self.ops) - 1:
            raise ValueError(util.Errors.value_error(
                "l must satisfy 2 <= l <= len(self.ops) - 1. "
                f"Got l={l} for {len(self.ops)} factors."
            ))
        if not isinstance(is_eig, (bool, np.bool_)):
            raise TypeError(util.Errors.type_error(
                "is_eig must be boolean."
            ))
        is_eig = bool(is_eig)
        if not all(op.linear for op in self.ops):
            raise ValueError(util.Errors.value_error(
                "Every factor of an SVD composition must be linear."
            ))

        middle_index = l - 1
        M = self.ops[middle_index]
        if not isinstance(M, PtwMultiplication):
            raise ValueError(util.Errors.value_error(
                f"The factor at position l={l} must be a "
                "PtwMultiplication operator."
            ))
        multiplier = np.asarray(M.factor)
        if not np.all(np.isreal(multiplier)) or not np.all(
                multiplier.real > 0):
            raise ValueError(util.Errors.value_error(
                "The singular-value multiplier must contain only positive "
                "real entries."
            ))

        all_spaces = [self.domain]
        all_spaces.extend(op.codomain for op in self.ops[::-1])
        all_complex = all(space.is_complex for space in all_spaces)
        distinguished_real = (
            not self.domain.is_complex
            and not self.codomain.is_complex
            and not M.domain.is_complex
        )
        if not (all_complex or distinguished_real):
            raise ValueError(util.Errors.value_error(
                "An SVD composition must either use complex vector spaces "
                "throughout, or have real domain, codomain, and singular-"
                "coordinate space."
            ))

        left_factors = self.ops[:middle_index]
        V = (
            left_factors[0]
            if len(left_factors) == 1
            else Composition(*left_factors)
        )
        right_adjoint_factors = [
            op.adjoint for op in self.ops[middle_index + 1:][::-1]
        ]
        U = (
            right_adjoint_factors[0]
            if len(right_adjoint_factors) == 1
            else Composition(*right_adjoint_factors)
        )
        if M.domain != M.codomain:
            raise ValueError(util.Errors.value_error(
                "The singular-value multiplier must map the reduced vector "
                "space to itself."
            ))
        if V.domain != M.codomain or U.domain != M.domain:
            raise ValueError(util.Errors.value_error(
                "The U, M, and V factors use incompatible reduced spaces."
            ))

        G_X = self._svd_gram_operator(G_X, self.domain, "G_X")
        G_Y = self._svd_gram_operator(G_Y, self.codomain, "G_Y")
        reduced_space = M.domain
        test_vector = reduced_space.randn()
        identity = reduced_space.identity

        V_gram_V = Composition(V.adjoint, G_Y, V)
        error, scale = self._svd_operator_error(
            V_gram_V, identity, test_vector
        )
        tolerance = self._svd_tolerance(reduced_space, scale)
        if not np.isfinite(error) or error > tolerance:
            raise ValueError(util.Errors.value_error(
                "V is not isometric with respect to G_Y: "
                f"||V* G_Y V z - z||={error}, tolerance={tolerance}."
            ))

        U_gram_U = Composition(U.adjoint, G_X, U)
        error, scale = self._svd_operator_error(
            U_gram_U, identity, test_vector
        )
        tolerance = self._svd_tolerance(reduced_space, scale)
        if not np.isfinite(error) or error > tolerance:
            raise ValueError(util.Errors.value_error(
                "U is not isometric with respect to G_X: "
                f"||U* G_X U z - z||={error}, tolerance={tolerance}."
            ))

        if is_eig:
            if self.domain != self.codomain:
                raise ValueError(util.Errors.value_error(
                    "An eigenvalue decomposition requires identical domain "
                    "and codomain."
                ))
            error, scale = self._svd_operator_error(U, V, test_vector)
            tolerance = self._svd_tolerance(self.domain, scale)
            if not np.isfinite(error) or error > tolerance:
                raise ValueError(util.Errors.value_error(
                    "is_eig=True requires U and V to agree. "
                    f"||U z - V z||={error}, tolerance={tolerance}."
                ))
            gram_test_vector = self.domain.randn()
            error, scale = self._svd_operator_error(
                G_X, G_Y, gram_test_vector
            )
            tolerance = self._svd_tolerance(self.domain, scale)
            if not np.isfinite(error) or error > tolerance:
                raise ValueError(util.Errors.value_error(
                    "is_eig=True requires G_X and G_Y to agree. "
                    f"||G_X z - G_Y z||={error}, tolerance={tolerance}."
                ))

        self._svd_data = {
            'l': l,
            'U': U,
            'V': V,
            'M': M,
            'G_X': G_X,
            'G_Y': G_Y,
            'is_eig': is_eig,
        }
        self._is_svd = True

    @property
    def use_buffers(self) -> bool:
        """Whether reusable intermediate buffers are enabled."""
        return self._use_buffers

    @use_buffers.setter
    def use_buffers(self, value: bool) -> None:
        if not isinstance(value, (bool, np.bool_)):
            raise TypeError(util.Errors.type_error(
                'use_buffers must be boolean.', value
            ))
        value = bool(value)
        if value == self._use_buffers:
            return
        if any(self._buffer_status.values()):
            raise RuntimeError(util.Errors.runtime_error(
                "Cannot change use_buffers while composition buffers are in use."
            ))
        self._use_buffers = value
        if value:
            self._initialize_buffers()
        else:
            self._interface_buffers = []
            self._buffer_status = {}

    @property
    def use_buffer(self) -> bool:
        """Backward-compatible singular alias for :attr:`use_buffers`."""
        return self.use_buffers

    @use_buffer.setter
    def use_buffer(self, value: bool) -> None:
        self.use_buffers = value

    def _initialize_buffers(self) -> None:
        """Allocate and map reusable buffers to composition interfaces."""
        self._interface_buffers = []
        self._buffer_status = {}
        if len(self.ops) < 2:
            return

        previous = None
        for interface_index in range(len(self.ops) - 1):
            # Interface i is ops[i].domain == ops[i+1].codomain.
            # It may share the preceding interface buffer exactly when ops[i]
            # maps a compatible space to itself.
            if interface_index > 0 and self.d_eq_cd[interface_index]:
                buffer = previous
            else:
                buffer = self.ops[interface_index].domain.empty()
                self._buffer_status[id(buffer)] = False
            self._interface_buffers.append(buffer)
            previous = buffer

    def _acquire_buffers(self) -> list[Vector] | None:
        if not self._use_buffers:
            return None
        if len(self._interface_buffers) != max(0, len(self.ops) - 1):
            self._initialize_buffers()
        busy = [key for key, in_use in self._buffer_status.items() if in_use]
        if busy:
            raise RuntimeError(util.Errors.runtime_error(
                "Composition buffers are already in use. This usually indicates "
                "recursive evaluation or concurrent use of the same Composition "
                "instance. Disable use_buffers or use a separate Composition instance."
            ))
        for key in self._buffer_status:
            self._buffer_status[key] = True
        return self._interface_buffers

    def _release_buffers(self) -> None:
        for key in self._buffer_status:
            self._buffer_status[key] = False

    def _ieval(
        self, x: Vector, out: Vector, differentiate: bool = False, **kwargs
    ) -> Vector:
        if not self._use_buffers or len(self.ops) < 2:
            return self._ieval_without_buffers(x, out, differentiate=differentiate, **kwargs)

        buffers = self._acquire_buffers()
        try:
            if differentiate:
                self._derivs = [None] * len(self.ops)
                current, deriv = self.ops[-1].linearize(
                    x, out=buffers[-1], **kwargs
                )
                self._derivs[-1] = deriv
                for i in range(len(self.ops) - 2, 0, -1):
                    current, deriv = self.ops[i].linearize(
                        current, out=buffers[i - 1], **kwargs
                    )
                    self._derivs[i] = deriv
                out, deriv = self.ops[0].linearize(current, out=out, **kwargs)
                self._derivs[0] = deriv
                return out

            current = self.ops[-1](x, out=buffers[-1], **kwargs)
            for i in range(len(self.ops) - 2, 0, -1):
                current = self.ops[i](current, out=buffers[i - 1], **kwargs)
            return self.ops[0](current, out=out, **kwargs)
        finally:
            self._release_buffers()

    def _ieval_without_buffers(
        self, x: Vector, out: Vector, differentiate: bool = False, **kwargs
    ) -> Vector:
        if differentiate:
            self._derivs = []
            x, deriv = self.ops[-1].linearize(x, **kwargs)
            self._derivs.append(deriv)
            for d_eq_cd, op in zip(self.d_eq_cd[-2:0:-1], self.ops[-2:0:-1]):
                if d_eq_cd:
                    x, deriv = op.linearize(x, out=x, **kwargs)
                else:
                    x, deriv = op.linearize(x, **kwargs)
                self._derivs.insert(0, deriv)
            out, deriv = self.ops[0].linearize(x, out=out, **kwargs)
            self._derivs.insert(0, deriv)
            return out

        x = self.ops[-1](x, **kwargs)
        for d_eq_cd, op in zip(self.d_eq_cd[-2:0:-1], self.ops[-2:0:-1]):
            if d_eq_cd:
                x = op(x, out=x, **kwargs)
            else:
                x = op(x, **kwargs)
        return self.ops[0](x, out=out, **kwargs)

    def _iderivative(self, x: Vector, out: Vector, **kwargs) -> Vector:
        if not self._use_buffers or len(self._derivs) < 2:
            x = self._derivs[-1](x, **kwargs)
            for d_eq_cd, deriv in zip(
                    self.d_eq_cd[-2:0:-1], self._derivs[-2:0:-1]):
                if d_eq_cd:
                    x = deriv(x, out=x, **kwargs)
                else:
                    x = deriv(x, **kwargs)
            return self._derivs[0](x, out=out, **kwargs)

        buffers = self._acquire_buffers()
        try:
            current = self._derivs[-1](x, out=buffers[-1], **kwargs)
            for i in range(len(self._derivs) - 2, 0, -1):
                current = self._derivs[i](
                    current, out=buffers[i - 1], **kwargs
                )
            return self._derivs[0](current, out=out, **kwargs)
        finally:
            self._release_buffers()

    def _iadjoint(self, y: Vector, out: Vector, **kwargs) -> Vector:
        ops = self.ops if self.linear else self._derivs
        if not self._use_buffers or len(ops) < 2:
            y = ops[0].adjoint(y, **kwargs)
            for d_eq_cd, op in zip(self.d_eq_cd[1:-1], ops[1:-1]):
                if d_eq_cd:
                    y = op.adjoint(y, out=y, **kwargs)
                else:
                    y = op.adjoint(y, **kwargs)
            return ops[-1].adjoint(y, out=out, **kwargs)

        buffers = self._acquire_buffers()
        try:
            current = ops[0].adjoint(y, out=buffers[0], **kwargs)
            for i in range(1, len(ops) - 1):
                current = ops[i].adjoint(
                    current, out=buffers[i], **kwargs
                )
            return ops[-1].adjoint(current, out=out, **kwargs)
        finally:
            self._release_buffers()

    def _iadjoint_eval(self, x: Vector, out: Vector, **kwargs) -> Vector:
        if kwargs:
            self.log.warning(
                "In the default implementation of _adjoint_eval it is unclear "
                "where to put keyword arguments. Ignoring kwargs=%s", kwargs
            )
        if not self._use_buffers or len(self.ops) < 2:
            return self._iadjoint_eval_without_buffers(x, out)

        buffers = self._acquire_buffers()
        try:
            if self.linear:
                current = self.ops[-1](x, out=buffers[-1])
                for i in range(len(self.ops) - 2, 0, -1):
                    current = self.ops[i](current, out=buffers[i - 1])
                current = self.ops[0].adjoint_eval(
                    current, out=buffers[0]
                )
                for i in range(1, len(self.ops) - 1):
                    current = self.ops[i].adjoint(
                        current, out=buffers[i]
                    )
                return self.ops[-1].adjoint(current, out=out)

            self._derivs = [None] * len(self.ops)
            current, deriv = self.ops[-1].linearize(
                x, out=buffers[-1]
            )
            self._derivs[-1] = deriv
            for i in range(len(self.ops) - 2, 0, -1):
                current, deriv = self.ops[i].linearize(
                    current, out=buffers[i - 1]
                )
                self._derivs[i] = deriv
            current, deriv = self.ops[0].linearize(
                current, out=buffers[0], return_adjoint_eval=True
            )
            self._derivs[0] = deriv
            for i in range(1, len(self._derivs) - 1):
                current = self._derivs[i].adjoint(
                    current, out=buffers[i]
                )
            return self._derivs[-1].adjoint(current, out=out)
        finally:
            self._release_buffers()

    def _iadjoint_eval_without_buffers(self, x: Vector, out: Vector) -> Vector:
        if self.linear:
            x = self.ops[-1](x)
            for d_eq_cd, op in zip(
                    self.d_eq_cd[-2:0:-1], self.ops[-2:0:-1]):
                if d_eq_cd:
                    x = op(x, out=x)
                else:
                    x = op(x)
            x = self.ops[0].adjoint_eval(x, out=x)
            for d_eq_cd, op in zip(self.d_eq_cd[1:-1], self.ops[1:-1]):
                if d_eq_cd:
                    x = op.adjoint(x, out=x)
                else:
                    x = op.adjoint(x)
            return self.ops[-1].adjoint(x, out=out)

        self._derivs = []
        x, deriv = self.ops[-1].linearize(x)
        self._derivs.append(deriv)
        for d_eq_cd, op in zip(
                self.d_eq_cd[-2:0:-1], self.ops[-2:0:-1]):
            if d_eq_cd:
                x, deriv = op.linearize(x, out=x)
            else:
                x, deriv = op.linearize(x)
            self._derivs.insert(0, deriv)
        x, deriv = self.ops[0].linearize(
            x, out=x, return_adjoint_eval=True
        )
        self._derivs.insert(0, deriv)
        for d_eq_cd, deriv in zip(
                self.d_eq_cd[1:-1], self._derivs[1:-1]):
            if d_eq_cd:
                x = deriv.adjoint(x, out=x)
            else:
                x = deriv.adjoint(x)
        return self._derivs[-1].adjoint(x, out=out)
    
    def _iadjoint_data(self, data: ObsDataT, out: Vector, **kwargs) -> Vector:
        ops = self.ops if self.linear else self._derivs
        if not self._use_buffers or len(ops) < 2:
            data = ops[0].adjoint_data(data, **kwargs)
            for d_eq_cd, op in zip(self.d_eq_cd[1:-1], ops[1:-1]):
                if d_eq_cd:
                    data = op.adjoint(data, out=data, **kwargs)
                else:
                    data = op.adjoint(data, **kwargs)
            return ops[-1].adjoint(data, out=out, **kwargs)

        buffers = self._acquire_buffers()
        try:
            current = ops[0].adjoint_data(
                data, out=buffers[0], **kwargs
            )
            for i in range(1, len(ops) - 1):
                current = ops[i].adjoint(
                    current, out=buffers[i], **kwargs
                )
            return ops[-1].adjoint(current, out=out, **kwargs)
        finally:
            self._release_buffers()

    def _iadjoint_derivative(self, x: Vector, out: Vector, **kwargs) -> Vector:
        if kwargs:
            self.log.warning(
                "In the default implementation of _adjoint_derivative it is "
                "unclear where to put keyword arguments. Ignoring kwargs=%s", kwargs
            )
        if not self._use_buffers or len(self._derivs) < 2:
            x = self._derivs[-1](x)
            for d_eq_cd, deriv in zip(
                    self.d_eq_cd[-2:0:-1], self._derivs[-2:0:-1]):
                if d_eq_cd:
                    x = deriv(x, out=x)
                else:
                    x = deriv(x)
            x = self._derivs[0].adjoint_eval(x, out=x)
            for d_eq_cd, deriv in zip(
                    self.d_eq_cd[1:-1], self._derivs[1:-1]):
                if d_eq_cd:
                    x = deriv.adjoint(x, out=x)
                else:
                    x = deriv.adjoint(x)
            return self._derivs[-1].adjoint(x, out=out)

        buffers = self._acquire_buffers()
        try:
            current = self._derivs[-1](x, out=buffers[-1])
            for i in range(len(self._derivs) - 2, 0, -1):
                current = self._derivs[i](
                    current, out=buffers[i - 1]
                )
            current = self._derivs[0].adjoint_eval(
                current, out=buffers[0]
            )
            for i in range(1, len(self._derivs) - 1):
                current = self._derivs[i].adjoint(
                    current, out=buffers[i]
                )
            return self._derivs[-1].adjoint(current, out=out)
        finally:
            self._release_buffers()

    @Operator.inverse.getter
    def inverse(self) -> Operator:
        if self._inverse is not None:
            return self._inverse
        try:
            return Composition(*(op.inverse for op in self.ops[::-1]))
        except NotImplementedError:
            raise NotImplementedError(util.Errors._compose_message(
                "INVERSE NOT DEFINED",
                f"The inverse of the composition {self} is not well-defined since one or more of the operators does not have an inverse."
            ))

    def __repr__(self) -> str:
        if hasattr(self, "ops"):
            return util.make_repr(self, *[type(op).__qualname__ for op in self.ops])
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
    def __init__(
        self,
        base_op: Operator,
        index: int | slice | Sequence[int],
    ) -> None:
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
                    f"The given tuple/list of indices has to be a list of integers in the range of {-n_codim},...,{n_codim}."))
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

    def _get_codomain_part(self, y: Vector) -> Vector:
        if(isinstance(self.index,int)):
            return y[self.index]
        else:
            return self.codomain.join(*[y[i] for i in self.index])
        
    def _extend_to_full(self, out: Vector) -> Vector:
        if isinstance(self.index,int):
            return self.base_op.codomain.join(*[cd.zeros() if i!=self.index else out for i,cd in enumerate(self.base_op.codomain)])
        else:
            return self.base_op.codomain.join(*[cd.zeros() if i not in self.index else out[self.index.index(i)] for i,cd in enumerate(self.base_op.codomain)])

    def _ieval(
        self, x: Vector, out: Vector, differentiate: bool = False, **kwargs
    ) -> Vector:
        full_out = self._extend_to_full(out)
        if self.base_op.linear:
            self.base_op._ieval(x, full_out, **kwargs)
        else:
            self.base_op._ieval(
                x, full_out,
                differentiate=differentiate,
                **kwargs
            )
        return out


    def _iderivative(self, x: Vector, out: Vector, **kwargs) -> Vector:
        self.base_op._iderivative(
            x, self._extend_to_full(out), **kwargs
        )
        return out

    def _iadjoint(self, y: Vector, out: Vector, **kwargs) -> Vector:
        out = self.base_op._iadjoint(self._extend_to_full(y), out, **kwargs)
        return out
    
    def __getitem__(
        self, val: int | slice | tuple[int, ...]
    ) -> PartOfOperator:
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
    def __init__(self, op: Operator, exponent: int) -> None:
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
    
    def _eval(self, x: Vector, **kwargs) -> Vector:
        res = x
        for _ in range(self.exponent):
            res = self.op(res, **kwargs)
        return res

    def _ieval(self, x: Vector, out: Vector, **kwargs) -> Vector:
        if out is x:
            for _ in range(self.exponent):
                out = self.op(out, out = out, **kwargs)
            return out
        else:
            self.codomain.copyto(out, x)
            for _ in range(self.exponent):
                out = self.op(out, out = out, **kwargs)
            return out

    def _adjoint(self, x: Vector, **kwargs) -> Vector:
        res = x
        for j in range(self.exponent):
            res = self.op.adjoint(res)
        return res

    def _iadjoint(
        self, x: Vector, out: Vector | None = None, **kwargs
    ) -> Vector:
        if out is x:
            for _ in range(self.exponent):
                out = self.op.adjoint(out, out = out, **kwargs)
            return out
        else:
            self.domain.copyto(out, x)
            for _ in range(self.exponent):
                out = self.op.adjoint(out, out = out, **kwargs)
            return out
    
    @Operator.inverse.getter
    def inverse(self) -> Operator:
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

    The allocating methods return an independent vector. In-place calls copy into the
    supplied output vector. A configurable no-copy mode is intentionally not provided,
    because allocating operator results must be freely modifiable by the caller.
    The identity operator is always self-adjoint.

    Parameters
    ----------
    domain : regpy.vecsps.VectorSpaceBase
        The underlying vector space.
    """

    def __init__(self, domain: vecsps.VectorSpaceBase) -> None:
        super().__init__(
            domain,
            domain,
            linear=True,
            self_adjoint=True,
        )

    def _ieval(self, x: Vector, out: Vector, **kwargs) -> Vector:
        if out is not x:
            self.domain.copyto(out, x)
        return out

    def _iadjoint(self, x: Vector, out: Vector, **kwargs) -> Vector:
        return self._ieval(x, out, **kwargs)

    def _iadjoint_eval(self, x: Vector, out: Vector, **kwargs) -> Vector:
        return self._ieval(x, out, **kwargs)

    @Operator.inverse.getter
    def inverse(self) -> Operator:
        return self

    def __repr__(self) -> str:
        return util.make_repr(self, self.domain)

class CoordinateProjection(Operator):
    """Restriction to selected coordinates of a MaskableVectorSpace.

    The codomain is the corresponding MaskedVectorSpace. The adjoint is
    extension by zero in algebraic dual coordinates. Masks must match the
    domain's mask protocol and are copied at construction.
    """

    def __init__(self, domain: vecsps.VectorSpaceBase, mask) -> None:
        from regpy.vecsps.maskable import MaskableVectorSpace
        if not isinstance(domain, MaskableVectorSpace):
            raise TypeError("CoordinateProjection requires a MaskableVectorSpace.")
        restricted = domain.get_masked_space(mask)
        self.mask = restricted.mask
        super().__init__(domain=domain, codomain=restricted, linear=True)

    def _eval(self, x: Vector) -> Vector:
        return self.codomain.restrict(x)

    def _adjoint(self, x: Vector) -> Vector:
        return self.codomain.extend(x)

    def _adjoint_eval(self, x: Vector) -> Vector:
        result = self.domain.zeros()
        self.domain.copyto(result, x, where=self.mask)
        return result

    def _iadjoint_eval(self, x: Vector, out: Vector) -> Vector:
        # Copy first to preserve overlapping views and shared tuple components.
        value = self._adjoint_eval(x)
        self.domain.copyto(out, value)
        return out

    def __repr__(self) -> str:
        return util.make_repr(self, self.domain, self.mask)


class CoordinateMask(Operator):
    """Full-space coordinate projection using the maskable-space protocol.

    Selected coefficients are preserved; all others are zeroed. Supports
    NumPy, NGSolve and maskable direct sums without indexing vector objects.
    """

    def __init__(self, domain: vecsps.VectorSpaceBase, mask) -> None:
        from regpy.vecsps.maskable import MaskableVectorSpace
        if not isinstance(domain, MaskableVectorSpace):
            raise TypeError("CoordinateMask requires a MaskableVectorSpace.")
        self.mask = domain.get_masked_space(mask).mask
        super().__init__(domain=domain, codomain=domain, linear=True)

    def _eval(self, x: Vector) -> Vector:
        result = self.domain.zeros()
        self.domain.copyto(result, x, where=self.mask)
        return result

    def _ieval(self, x: Vector, out: Vector) -> Vector:
        if x is out:
            self.domain.fill(out, 0, where=self.domain.logical_not(self.mask))
        else:
            # A distinct object can still share storage with x.
            value = self._eval(x)
            self.domain.copyto(out, value)
        return out

    def _adjoint(self, x: Vector) -> Vector:
        return self._eval(x)

    def _adjoint_eval(self, x: Vector) -> Vector:
        return self._eval(x)

    def _iadjoint(self, x: Vector, out: Vector) -> Vector:
        return self._ieval(x, out)

    def _iadjoint_eval(self, x: Vector, out: Vector) -> Vector:
        return self._ieval(x, out)

    def __repr__(self) -> str:
        return util.make_repr(self, self.domain, self.mask)


class PtwMultiplication(Operator):
    r"""A multiplication operator by a constant factor where each vector entry is multiplied 
    by the vector entry of `factor`. This works analogous to the pointwise multiplication in numpy. Note that the pointwise multiplication of the coefficients
    might not represent a pointwise multiplication of the underlying functions depending on the vector space.

    Parameters
    ----------
    domain : regpy.vecsps.VectorSpaceBase
        The underlying vector space
    factor : Vector or ArrayLike
        The factor by which to multiply. In case of domain being NumPyVectorSpace it can be anything that can be broadcast to `domain.shape`.

    The operator is marked as self-adjoint exactly when the stored multiplier
    is real-valued. Native pointwise vector operations are used when
    available. Otherwise the operator falls back to
    :meth:`~regpy.vecsps.VectorSpaceBase.iter_coefficients` and
    :meth:`~regpy.vecsps.VectorSpaceBase.set_coefficients`.
    """
    def __init__(
        self,
        domain: vecsps.VectorSpaceBase,
        factor: Vector | ArrayLike,
    ) -> None:
        # Check that factor can broadcast against domain elements without
        # increasing their size.
        if not isinstance(domain,vecsps.VectorSpaceBase):
            raise ValueError(util.Errors.not_a_vecsp(
                domain,
                vecsps.VectorSpaceBase,
                add_info="The domain for a PtwMultiplication has to be a vector spaces instance."
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
        self.factor = domain.empty()
        domain.copyto(self.factor, factor)
        if not domain.is_complex:
            factor_is_real = True
        else:
            try:
                factor_is_real = domain.norm(self.factor.imag) == 0
            except (AttributeError, TypeError, NotImplementedError):
                factor_is_real = all(
                    np.isreal(coefficient)
                    for coefficient in domain.iter_coefficients(self.factor)
                )
        super().__init__(
            domain,
            domain,
            linear=True,
            self_adjoint=factor_is_real,
        )

    def _eval(self, x: Vector) -> Vector:
        try:
            return self.factor * x
        except (AttributeError, TypeError, ValueError, NotImplementedError):
            out = self.codomain.empty()
            return self._icoefficientwise(x, out)

    def _ieval(self, x: Vector, out: Vector) -> Vector:
        try:
            if out is not x:
                self.codomain.copyto(out, x)
            out *= self.factor
            return out
        except (AttributeError, TypeError, ValueError, NotImplementedError):
            return self._icoefficientwise(x, out)

    def _adjoint(self, x: Vector) -> Vector:
        try:
            if self.domain.is_complex:
                return x * np.conjugate(self.factor)
            return x * self.factor
        except (AttributeError, TypeError, ValueError, NotImplementedError):
            out = self.domain.empty()
            return self._icoefficientwise(
                x, out, conjugate_factor=self.domain.is_complex
            )

    def _iadjoint(self, x: Vector, out: Vector) -> Vector:
        try:
            if out is not x:
                self.domain.copyto(out, x)
            if self.domain.is_complex:
                out *= np.conjugate(self.factor)
            else:
                out *= self.factor
            return out
        except (AttributeError, TypeError, ValueError, NotImplementedError):
            return self._icoefficientwise(
                x, out, conjugate_factor=self.domain.is_complex
            )

    def _adjoint_eval(self, x: Vector) -> Vector:
        try:
            if self.domain.is_complex:
                return x * self.factor.conj()*self.factor
            return x * self.factor**2
        except (AttributeError, TypeError, ValueError, NotImplementedError):
            out = self.domain.empty()
            return self._icoefficientwise(x, out, squared_modulus=True)

    def _iadjoint_eval(self, x: Vector, out: Vector) -> Vector:
        try:
            if out is not x:
                self.domain.copyto(out, x)
            if self.domain.is_complex:
                out *= np.conjugate(self.factor)*self.factor
            else:
                out *= self.factor**2
            return out
        except (AttributeError, TypeError, ValueError, NotImplementedError):
            return self._icoefficientwise(x, out, squared_modulus=True)

    def _icoefficientwise(
        self,
        x: Vector,
        out: Vector,
        *,
        conjugate_factor: bool = False,
        squared_modulus: bool = False,
    ) -> Vector:
        """Protocol-conforming coefficientwise fallback implementation."""
        x_coefficients = self.domain.iter_coefficients(x)
        factor_coefficients = self.domain.iter_coefficients(self.factor)
        if squared_modulus:
            result = (
                x_coefficient
                * np.conjugate(factor_coefficient)
                * factor_coefficient
                for x_coefficient, factor_coefficient in zip(
                    x_coefficients, factor_coefficients
                )
            )
        elif conjugate_factor:
            result = (
                x_coefficient * np.conjugate(factor_coefficient)
                for x_coefficient, factor_coefficient in zip(
                    x_coefficients, factor_coefficients
                )
            )
        else:
            result = (
                x_coefficient * factor_coefficient
                for x_coefficient, factor_coefficient in zip(
                    x_coefficients, factor_coefficients
                )
            )
        self.domain.set_coefficients(out, result)
        return out

    @property
    def invertible(self) -> bool:
        # TODO: check if multiplier has zeros!
        return True

    @Operator.inverse.getter
    def inverse(self) -> Operator:
        if self._inverse is not None:
            return self._inverse
        sav = np.seterr(divide='raise')
        try:
            try:
                inverse_factor = 1 / self.factor
            except (AttributeError, TypeError, ValueError, NotImplementedError):
                inverse_factor = self.domain.empty()
                self.domain.set_coefficients(
                    inverse_factor,
                    (
                        1 / coefficient
                        for coefficient in self.domain.iter_coefficients(
                            self.factor
                        )
                    ),
                )
            return PtwMultiplication(self.domain, inverse_factor)
        finally:
            np.seterr(**sav)

    def __repr__(self) -> str:
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
    def __init__(self, op: Operator, offset: Vector | Scalar) -> None:
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
        self.offset = op.codomain.empty()
        op.codomain.copyto(self.offset, offset)

    def _ieval(
        self, x: Vector, out: Vector, differentiate: bool = False, **kwargs
    ) -> Vector:
        if differentiate:
            out, self._deriv = self.op.linearize(x, out = out, **kwargs)
            out += self.offset
            return out
        else:
            out = self.op(x, out=out,**kwargs)
            out += self.offset
            return out

    def _iadjoint_eval(self, x: Vector, out: Vector, **kwargs) -> Vector:
        out, self._deriv = self.op.linearize(x, out = out, return_adjoint_eval= True, **kwargs)
        out += self._adjoint(self.offset)
        return out

    def _iderivative(self, x: Vector, out: Vector, **kwargs) -> Vector:
        return self._deriv(x, out = out, **kwargs)

    def _iadjoint(self, y: Vector, out: Vector, **kwargs) -> Vector:
        return self._deriv.adjoint(y, out = out, **kwargs)
    
    def _iadjoint_derivative(self, x: Vector, out: Vector, **kwargs) -> Vector:
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
    def __init__(self, op: Operator, offset: Vector | Scalar) -> None:
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
        self.offset = op.domain.empty()
        op.domain.copyto(self.offset, offset)

    def _ieval(
        self, x: Vector, out: Vector, differentiate: bool = False, **kwargs
    ) -> Vector:
        if differentiate:
            out, self._deriv = self.op.linearize(x-self.offset, out = out, **kwargs)
            return out
        else:
            return self.op(x - self.offset, out = out, **kwargs)
        
    def _iadjoint_eval(self, x: Vector, out: Vector, **kwargs) -> Vector:
        out, self._deriv = self.op.linearize(x-self.offset, out = out, return_adjoint_eval=True, **kwargs)
        return out 
    
    def _iderivative(self, h: Vector, out: Vector, **kwargs) -> Vector:
        return self._deriv(h, out = out, **kwargs)

    def _iadjoint(self, y: Vector, out: Vector, **kwargs) -> Vector:
        return self._deriv.adjoint(y, out = out, **kwargs)
    
    def _iadjoint_derivative(self, x: Vector, out: Vector, **kwargs) -> Vector:
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
    *ops : tuple[Operator, ...]
        Variable number of Operator instances to be composed to a direct sum.
    flatten : bool, optional
        If True, summands that are themselves direct sums will be merged with
        this one. Default: False.
    domain, codomain : vecsps.VectorSpaceBase or callable, optional
        Either the underlying vector space or a factory function that will be called with all
        summands' vector spaces passed as arguments and should return a vecsps.DirectSum instance.
        The resulting vector space should be iterable, yielding the individual summands.
        Default: vecsps.DirectSum.

    The direct sum is marked as self-adjoint if all constituent operators are
    self-adjoint and the resulting domain and codomain coincide.
    """

    def __init__(
        self,
        *ops: Operator,
        flatten: bool = False,
        domain: vecsps.DirectSum | Callable[..., vecsps.DirectSum] | None = None,
        codomain: vecsps.DirectSum | Callable[..., vecsps.DirectSum] | None = None,
    ) -> None:
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
        super().__init__(
            domain=domain,
            codomain=codomain,
            linear=all(op.linear for op in self.ops),
            self_adjoint=(
                domain == codomain
                and all(op.self_adjoint for op in self.ops)
            ),
        )

    def _ieval(
        self, x: Vector, out: Vector, differentiate: bool = False, **kwargs
    ) -> Vector:
        if differentiate:
            linearizations = [op.linearize(x_i,out = out_i) for op, x_i, out_i in zip(self.ops,x, out)]
            self._derivs = [l[1] for l in linearizations]
            return out
        else:
            for op, x_i, out_i in zip(self.ops, x, out):
                out_i = op(x_i, out = out_i)
            return out

    def _iadjoint_eval(self, x: Vector, out: Vector, **kwargs) -> Vector:
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

    def _iderivative(self, x: Vector, out: Vector, **kwargs) -> Vector:
        for deriv, x_i, out_i in zip(self._derivs, x, out):
            out_i = deriv(x_i, out = out_i)
        return out

    def _iadjoint(self, y: Vector, out: Vector, **kwargs) -> Vector:
        if hasattr(self,"full_domain") and out not in self.full_domain:
            out = self._insert_constants(out)
        if self.linear:
            ops = self.ops
        else:
            ops = self._derivs
        for op, y_i, out_i in zip(ops, y, out):
            out_i = op.adjoint(y_i, out = out_i)
        return out
    
    def _iadjoint_derivative(self, x: Vector, out: Vector, **kwargs) -> Vector:
        if hasattr(self,"full_domain") and out not in self.full_domain:
            out = self._insert_constants(out)
        for deriv, y_i, out_i in zip(self._derivs, x, out):
            out_i = deriv.adjoint_eval(y_i, out = out_i)
        return out

    @Operator.inverse.getter
    def inverse(self) -> Operator:
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

    def __repr__(self) -> str:
        return util.make_repr(self, *self.ops)

    def __getitem__(self, item: int | slice) -> Operator | Sequence[Operator]:
        if item is None:
            return self
        return self.ops[item]

    def __iter__(self) -> Iterator[Operator]:
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
    ops : Sequence[Operator]
        Operator instances to be put together into a vector. Each operator
        is required to have the same domain.
    codomain : vecsps.VectorSpaceBase or callable, optional
        Either the underlying vector space or a factory function that will be called with all
        summands' vector spaces passed as arguments and should return a vecsps.DirectSum instance.
        The resulting vector space should be iterable, yielding the individual summands.
        Default: vecsps.DirectSum.
    """

    def __init__(
        self,
        ops: Sequence[Operator],
        domain: vecsps.VectorSpaceBase | None = None,
        codomain: vecsps.DirectSum | Callable[..., vecsps.DirectSum] | None = None,
    ) -> None:
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

    @staticmethod
    def _shares_memory(x: Vector, y: Vector) -> bool:
        """Whether two vectors can be modified through the same storage.

        Object identity covers generic vector types. NumPy arrays additionally
        need a memory-overlap check because direct-sum components may be views.
        TupleVectors are handled recursively for nested direct sums.
        """
        if x is y:
            return True
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            return np.shares_memory(x, y)
        if isinstance(x, vecsps.TupleVector):
            return any(VectorOfOperators._shares_memory(x_i, y) for x_i in x)
        if isinstance(y, vecsps.TupleVector):
            return any(VectorOfOperators._shares_memory(x, y_i) for y_i in y)
        return False

    def _preserve_input_if_needed(self, x: Vector, out: Vector) -> Vector:
        """Copy ``x`` only if an early output component aliases it.

        The final output component may alias ``x`` safely: after it is written,
        no later component evaluation needs the original input anymore.
        """
        for i, out_i in enumerate(out):
            if i == len(self.ops) - 1:
                break
            if self._shares_memory(x, out_i):
                source = self.domain.empty()
                self.domain.copyto(source, x)
                return source
        return x

    def _ieval(
        self, x: Vector, out: Vector, differentiate: bool = False, **kwargs
    ) -> Vector:
        x = self._preserve_input_if_needed(x, out)
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
    
    def _adjoint_eval(self, x: Vector, **kwargs) -> Vector:
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

    def _iderivative(self, x: Vector, out: Vector, **kwargs) -> Vector:
        x = self._preserve_input_if_needed(x, out)
        for deriv, out_i in zip(self._derivs, out):
            deriv(x, out=out_i, **kwargs)
        return out

    def _iadjoint(self, y: Vector, out: Vector, **kwargs) -> Vector:
        ops = self.ops if self.linear else self._derivs

        # Writing to ``out`` may destroy a component of ``y`` that is still
        # needed later. Preserve the complete input only in that case.
        if any(self._shares_memory(out, y_i) for y_i in y):
            y_copy = self.codomain.empty()
            self.codomain.copyto(y_copy, y)
            y = y_copy

        first = True
        for op, y_i in zip(ops, y):
            if first:
                op.adjoint(y_i, out=out, **kwargs)
                first = False
            else:
                out += op.adjoint(y_i, **kwargs)

        return out

    def _adjoint(self, y: Vector, **kwargs) -> Vector:
        if self.linear:
            ops = self.ops
        else:
            ops = self._derivs
        out = self.domain.zeros()
        for op, y_i in zip(ops, y):
            out += op.adjoint(y_i, **kwargs)
        return out
    
    def _adjoint_derivative(self, x: Vector, **kwargs) -> Vector:
        out = self.domain.zeros() 
        for deriv in self._derivs:
            out += deriv.adjoint_eval(x,**kwargs)
        return out

    def __repr__(self) -> str:
        vec_repr = "[" +", ".join([repr(op) for op in self.ops])+"]"
        return util.make_repr(self, vec_repr)

    def __getitem__(self, item: int | slice) -> Operator | Sequence[Operator]:
        return self.ops[item]

    def __iter__(self) -> Iterator[Operator]:
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
    ops : Sequence[Sequence[Operator | None]]
        Rows or columns of operator instances used to build the matrix. Each inner
        sequence has to have the same length; zero operators are represented by None.
    domain, codomain : vecsps.VectorSpaceBase or callable, optional
        Either the underlying vector space or a factory function that will be called with all
        summands' vector spaces passed as arguments and should return a vecsps.DirectSum instance.
        The resulting vector space should be iterable, yielding the individual summands.
        Default: vecsps.DirectSum.
    """

    def __init__(
        self,
        ops: Sequence[Sequence[Operator | None]],
        domain: vecsps.DirectSum | Callable[..., vecsps.DirectSum] | None = None,
        codomain: vecsps.DirectSum | Callable[..., vecsps.DirectSum] | None = None,
    ) -> None:
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

    def _eval(
        self, x: Vector, differentiate: bool = False, **kwargs
    ) -> Vector:
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

    def _derivative(self, x: Vector, **kwargs) -> Vector:
        out = self.codomain.zeros()
        for Tprime_j, x_j in zip(self._derivs,x):
            for Tprime_ij,out_i in zip(Tprime_j,out):
                if Tprime_ij:
                    out_i += Tprime_ij(x_j)
        return out

    def _adjoint(self, y: Vector, **kwargs) -> Vector:
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

    def __repr__(self) -> str:
        mat_repr = "[[" +"],\n[".join([", ".join([repr(op) if op else "0" for op in row]) for row in self.ops])+"]"
        return util.make_repr(self, mat_repr)

    def __getitem__(
        self, item: int | slice
    ) -> Sequence[Operator | None] | Sequence[Sequence[Operator | None]]:
        return self.ops[item]

    def __iter__(self) -> Iterator[Sequence[Operator | None]]:
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

    def __init__(
        self,
        domain: vecsps.DirectSum,
        codomain: vecsps.VectorSpaceBase | None = None,
    ) -> None:
        if not isinstance(domain,vecsps.DirectSum):
            raise ValueError(util.Errors.not_a_vecsp(
                domain,
                vecsps.DirectSum,
                add_info="To construct a Sum (summation operator) the domain has to be a DirectSum!"
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
                add_info="The codomain has to be identical to the summands of the domain."
            ))

    def _ieval(self, x: Vector, out: Vector) -> Vector:
        components = iter(x)
        self.codomain.copyto(out, next(components))
        for component in components:
            out += component
        return out

    def _iadjoint(self, y: Vector, out: Vector) -> Vector:
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

    def __init__(
        self,
        domain: vecsps.DirectSum,
        codomain: vecsps.VectorSpaceBase | None = None,
    ) -> None:
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
                add_info="The codomain has to be identical to the summands of the domain."
            ))

    def _ieval(
        self, x: Vector, out: Vector, differentiate: bool = False
    ) -> Vector:
        components = list(x)
        self.codomain.copyto(out, components[0])
        for x_i in components[1:]:
            out *= x_i
        if differentiate:
            self.deriv_data = [self.codomain.ones() for _ in components]
            for j, x_j in enumerate(components):
                for i in range(len(components)):
                    if i != j:
                        self.deriv_data[i] *= x_j
        return out

    def _iderivative(self, x: Vector, out: Vector) -> Vector:
        self.codomain.copyto(out, self.deriv_data[0])
        out *= x[0]
        for factor, x_i in zip(self.deriv_data[1:], x[1:]):
            out += factor * x_i
        return out

    def _iadjoint(self, y: Vector, out: Vector) -> Vector:
        for i, summand in enumerate(self.domain.summands):
            out[i] = (y*np.conjugate(self.deriv_data[i])).real if not summand.is_complex else y*np.conjugate(self.deriv_data[i])
        return out


class RealPart(Operator):
    r"""The pointwise real part operator.

    The vectors' native ``real`` attribute is used when available. Otherwise
    the real parts of its native coefficients are written through the vector
    space's coefficient interface.

    Parameters
    ----------
    domain : regpy.vecsps.VectorSpaceBase
        The underlying vector space. The codomain will be the corresponding
        `regpy.vecsps.VectorSpaceBase.real_space`.
    """

    def __init__(self, domain: vecsps.VectorSpaceBase) -> None:
        if not isinstance(domain,vecsps.VectorSpaceBase):
            raise ValueError(util.Errors.not_a_vecsp(
                domain,
                vecsps.VectorSpaceBase
            ))
        codomain = domain.real_space()
        super().__init__(domain, codomain, linear=True)

    def _ieval(self, x: Vector, out: Vector) -> Vector:
        try:
            real_part = x.real
        except (AttributeError, TypeError, NotImplementedError):
            self.codomain.set_coefficients(
                out,
                (
                    np.real(coefficient)
                    for coefficient in self.domain.iter_coefficients(x)
                ),
            )
            return out
        self.codomain.copyto(out, real_part)
        return out

    def _iadjoint(self, y: Vector, out: Vector) -> Vector:
        self.domain.copyto(out, y)
        return out


class ImaginaryPart(Operator):
    r"""The pointwise imaginary part operator.

    The vector's native ``imag`` attribute is used when available. Otherwise
    the imaginary parts of its native coefficients are written through the
    vector space's coefficient interface.

    Parameters
    ----------
    domain : regpy.vecsps.VectorSpaceBase
        The underlying vector space. The codomain will be the corresponding
        `regpy.vecsps.VectorSpaceBase.real_space`.
    """

    def __init__(self, domain: vecsps.VectorSpaceBase) -> None:
        if not isinstance(domain,vecsps.VectorSpaceBase) or not domain.is_complex:
            raise ValueError(util.Errors.not_a_vecsp(
                domain,
                vecsps.VectorSpaceBase,
                add_info="To consider a ImaginaryPart operator the domain is required to be complex!"
            ))
        codomain = domain.real_space()
        super().__init__(domain, codomain, linear=True)

    def _ieval(self, x: Vector, out: Vector) -> Vector:
        try:
            imaginary_part = x.imag
        except (AttributeError, TypeError, NotImplementedError):
            self.codomain.set_coefficients(
                out,
                (
                    np.imag(coefficient)
                    for coefficient in self.domain.iter_coefficients(x)
                ),
            )
            return out
        self.codomain.copyto(out, imaginary_part)
        return out

    def _iadjoint(self, y: Vector, out: Vector) -> Vector:
        self.domain.copyto(out, 1j * y)
        return out

class SplitRealImag(VectorOfOperators):
    """Splits a complex vector into its real and imaginary part. 
    Implemented as VectorOfOperators of RealPart and ImaginaryPart.

    Parameters
    ----------
    domain: regpy.vecsps.VectorSpaceBase
    """
    def __init__(self, domain: vecsps.VectorSpaceBase) -> None:
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
    def inverse(self) -> Operator:
        return self.adjoint

    @property
    def invertible(self) -> bool:
        return True

class SquaredModulus(Operator):
    r"""The pointwise squared modulus operator.

    Native NumPy-style operations are used when the vector representation
    provides them. Otherwise evaluation, derivative and adjoint derivative
    fall back to the coefficient interface of the vector spaces.

    Parameters
    ----------
    domain : regpy.vecsps.VectorSpaceBase
        The underlying vector space. The codomain will be the corresponding
        `regpy.vecsps.VectorSpaceBase.real_space`.
    """

    def __init__(self, domain: vecsps.VectorSpaceBase) -> None:
        if not isinstance(domain,vecsps.VectorSpaceBase):
            raise ValueError(util.Errors.not_a_vecsp(
                domain,
                vecsps.VectorSpaceBase
            ))
        codomain = domain.real_space()
        super().__init__(domain, codomain)

    def _ieval(
        self, x: Vector, out: Vector, differentiate: bool = False
    ) -> Vector:
        if differentiate:
            self._factor = 2 * x
        try:
            np.square(x.real, out=out)
            out += np.square(x.imag)
        except (AttributeError, TypeError, ValueError, NotImplementedError):
            self.codomain.set_coefficients(
                out,
                (
                    np.real(coefficient * np.conjugate(coefficient))
                    for coefficient in self.domain.iter_coefficients(x)
                ),
            )
        return out

    def _iderivative(self, h: Vector, out: Vector) -> Vector:
        try:
            np.multiply(self._factor.real, h.real, out=out)
            if self.domain.is_complex:
                out += self._factor.imag * h.imag
        except (AttributeError, TypeError, ValueError, NotImplementedError):
            self.codomain.set_coefficients(
                out,
                (
                    np.real(
                        np.conjugate(factor_coefficient) * h_coefficient
                    )
                    for factor_coefficient, h_coefficient in zip(
                        self.domain.iter_coefficients(self._factor),
                        self.domain.iter_coefficients(h),
                    )
                ),
            )
        return out

    def _iadjoint(self, y: Vector, out: Vector) -> Vector:
        try:
            np.multiply(self._factor, y, out=out)
        except (AttributeError, TypeError, ValueError, NotImplementedError):
            self.domain.set_coefficients(
                out,
                (
                    factor_coefficient * y_coefficient
                    for factor_coefficient, y_coefficient in zip(
                        self.domain.iter_coefficients(self._factor),
                        self.codomain.iter_coefficients(y),
                    )
                ),
            )
        return out


class Zero(Operator):
    r"""The constant zero operator.

    Parameters
    ----------
    domain : regpy.vecsps.VectorSpaceBase
        The underlying vector space.
    codomain : regpy.vecsps.VectorSpaceBase, optional
        The vector space of the codomain. Defaults to `domain`.

    The zero operator is self-adjoint when domain and codomain coincide.
    """
    def __init__(
        self,
        domain: vecsps.VectorSpaceBase,
        codomain: vecsps.VectorSpaceBase | None = None,
    ) -> None:
        if codomain is None:
            codomain = domain
        super().__init__(
            domain,
            codomain,
            linear=True,
            self_adjoint=domain == codomain,
        )

    def _eval(self, x: Vector) -> Vector:
        return self.codomain.zeros()

    def _ieval(self, x: Vector, out: Vector) -> Vector:
        self.codomain.fill(out, 0)
        return out

    def _adjoint(self, x: Vector) -> Vector:
        return self.domain.zeros()

    def _iadjoint(self, x: Vector, out: Vector) -> Vector:
        self.domain.fill(out, 0)
        return out
    
    def _adjoint_eval(self, x: Vector) -> Vector:
        return self.domain.zeros()
    
    def _iadjoint_eval(self, x: Vector, out: Vector) -> Vector:
        self.domain.fill(out, 0)
        return out

class SciPyLinearOperator(LinearOperator):
    r"""A class wrapping a linear operator \(F\) into a scipy.sparse.linalg.LinearOperator so that it can be used conveniently in scipy methods.
    The domain and codomain are flattened.

    Parameters
    ----------
    op2 : Operator
        The operator to be put into a scipy.linalg.LinearOperator. 
    """
    def __init__(self, op2: Operator) -> None:
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

    
    def _matvec(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
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
    
    def _rmatvec(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
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
