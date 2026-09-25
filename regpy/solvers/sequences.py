"""Lazy arithmetic on numeric sequences, without NumPy or plotting dependencies.

Examples
--------
>>> from itertools import islice
>>> list(islice(0.1 * (0.5 ** Naturals()), 4))
[0.1, 0.05, 0.025, 0.0125]
>>> list(Naturals(first=2, last=4))
[2, 3, 4]
>>> list(islice(Naturals(last=3) + 4 * Naturals(first=1), 8))
[0, 1, 2, 3, 4, 8, 12, 16]
>>> list(MappedSequence(Naturals(last=3), lambda x: x / (1 + x)))
[0.0, 0.5, 0.6666666666666666, 0.75]
"""
from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Any

from collections.abc import Callable, Iterable, Iterator
from itertools import chain, count
from numbers import Integral, Real
import operator
from types import NotImplementedType

__all__ = ['Sequence', 'Naturals', 'MappedSequence']
Number = int | float


def _is_number(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


class Sequence(Iterator[Number]):
    r"""A single-pass, lazy sequence of real numbers.
    
    Construct from any numeric iterable, including a finite list or an infinite
    generator. Integers and floats share one interface; values are not coerced
    to float, so integer powers retain Python's arbitrary precision.
    
    Multiplication, division, floor division, modulo, subtraction and powers
    operate elementwise, with a scalar or another Sequence. Unary +, - and abs
    are supported. Adding two Sequences concatenates them; adding a scalar
    shifts every value. Use MappedSequence for arbitrary transformations.
    
    Binary elementwise operations stop at the shorter input. As with zip,
    discovering exhaustion can consume one unused value of the other input.
    An operation with exactly the same Sequence object on both sides consumes
    it only once per result. Distinct expressions sharing an underlying iterator
    still share its state: construct fresh sequences for independent uses.
    
    Iteration resumes where it stopped; iter(sequence) does not restart it.
    Operations construct new wrappers and do not consume inputs until advanced.
    Concatenation reaches the second input only when the first is exhausted.
    Arithmetic errors and non-real results are reported when advancing, not
    when constructing an expression. This includes complex fractional powers.
    
    Parameters
    ----------
    values: Iterable[Number]
        Values.
    """

    def __init__(self, values: Iterable[Number]) -> None:
        self._iterator = iter(values)

    def __iter__(self) -> Sequence:
        return self

    def __next__(self) -> Number:
        value = next(self._iterator)
        if not _is_number(value):
            raise TypeError(f'Sequence values must be real numbers, got {value!r}.')
        return value

    def map(self, f: Callable[[Number], Number]) -> MappedSequence:
        r"""Apply f lazily to each remaining value.
        
        Parameters
        ----------
        f: Callable[[Number], Number]
            F.
        
        
        Returns
        -------
        MappedSequence
            Result described above.
        """
        return MappedSequence(self, f)

    def _binary(self, other: Number | Sequence, operation: Callable[[Any, Any], Any], reflected: bool=False) -> Sequence | NotImplementedType:
        if isinstance(other, Sequence):
            if other is self:
                return self.map(lambda x: operation(x, x))
            left, right = (other, self) if reflected else (self, other)
            return Sequence(operation(x, y) for x, y in zip(left, right))
        if _is_number(other):
            return self.map(lambda x: operation(other, x) if reflected
                            else operation(x, other))
        return NotImplemented

    def __add__(self, other: Number | Sequence) -> Sequence | NotImplementedType:
        if isinstance(other, Sequence):
            return Sequence(chain(self, other))
        return self._binary(other, operator.add)

    def __radd__(self, other: Number | Sequence) -> Sequence | NotImplementedType:
        if isinstance(other, Sequence):
            return Sequence(chain(other, self))
        return self._binary(other, operator.add, True)

    def __sub__(self, other: Number | Sequence) -> Sequence | NotImplementedType:
        return self._binary(other, operator.sub)

    def __rsub__(self, other: Number | Sequence) -> Sequence | NotImplementedType:
        return self._binary(other, operator.sub, True)

    def __mul__(self, other: Number | Sequence) -> Sequence | NotImplementedType:
        return self._binary(other, operator.mul)

    def __rmul__(self, other: Number | Sequence) -> Sequence | NotImplementedType:
        return self._binary(other, operator.mul, True)

    def __truediv__(self, other: Number | Sequence) -> Sequence | NotImplementedType:
        return self._binary(other, operator.truediv)

    def __rtruediv__(self, other: Number | Sequence) -> Sequence | NotImplementedType:
        return self._binary(other, operator.truediv, True)

    def __floordiv__(self, other: Number | Sequence) -> Sequence | NotImplementedType:
        return self._binary(other, operator.floordiv)

    def __rfloordiv__(self, other: Number | Sequence) -> Sequence | NotImplementedType:
        return self._binary(other, operator.floordiv, True)

    def __mod__(self, other: Number | Sequence) -> Sequence | NotImplementedType:
        return self._binary(other, operator.mod)

    def __rmod__(self, other: Number | Sequence) -> Sequence | NotImplementedType:
        return self._binary(other, operator.mod, True)

    def __pow__(self, other: Number | Sequence) -> Sequence | NotImplementedType:
        return self._binary(other, operator.pow)

    def __rpow__(self, other: Number | Sequence) -> Sequence | NotImplementedType:
        return self._binary(other, operator.pow, True)

    def __neg__(self) -> Sequence:
        return self.map(operator.neg)

    def __pos__(self) -> Sequence:
        return self.map(operator.pos)

    def __abs__(self) -> Sequence:
        return self.map(operator.abs)


class Naturals(Sequence):
    r"""Yield first, first+1, ... through last inclusively.
    
    first defaults to 0; last=None gives an infinite sequence. Both bounds
    must be nonnegative integers. last < first produces an empty sequence.
    
    Parameters
    ----------
    first: int [default: 0]
        First.
    last: int | None [default: None]
        Last.
    """

    def __init__(self, first: int = 0, last: int | None = None) -> None:
        for name, value in (('first', first), ('last', last)):
            if name == 'last' and value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f'{name} must be a nonnegative integer.')
            if value < 0:
                raise ValueError(f'{name} must be nonnegative.')
        super().__init__(count(int(first)) if last is None
                         else range(int(first), int(last) + 1))


class MappedSequence(Sequence):
    r"""Yield f(x_n) from a supplied Sequence, consuming it lazily.
    
    f is called exactly once per requested output value. Its result must be
    real; integer and float values may be freely mixed.
    
    Parameters
    ----------
    sequence: Sequence
        Sequence.
    f: Callable[[Number], Number]
        F.
    """

    def __init__(self, sequence: Sequence,
                 f: Callable[[Number], Number]) -> None:
        if not isinstance(sequence, Sequence):
            raise TypeError('sequence must be a Sequence.')
        if not callable(f):
            raise TypeError('f must be callable.')
        super().__init__(map(f, sequence))


class EffectiveAlpha(Sequence):
    r"""Effective parameters of iterated Tikhonov, consumed lazily.
    
    Given positive finite parameters alpha_j, yield alpha_eff_k satisfying
    1/alpha_eff_k = sum(1/alpha_j, j=1,...,k). Integer and float inputs use
    the same interface. Accept a Sequence or any other numeric iterable.
    
    Iterated Tikhonov is implicit time stepping for Showalter's equation,
    with step sizes 1/alpha_j. Thus 1/alpha_eff_k is the accumulated time.
    This matches time scales, not the discrete and continuous filters.
    
    A stable recurrence avoids explicitly summing reciprocals. Finite input
    terminates after the same number of outputs; empty input stays empty.
    Validation occurs when values are consumed. Nonpositive/nonfinite values
    raise ValueError, non-real values TypeError, and underflow to zero raises
    FloatingPointError. Like every Sequence, this is a single-pass iterator.
    
    Parameters
    ----------
    alphas: Iterable[Number]
        Alphas.
    """

    def __init__(self, alphas: Iterable[Number]) -> None:
        from math import isfinite
        source = iter(alphas)

        def values() -> Iterator[Any]:
            r"""Values.
            
            
            Yields
            ------
            Any
                Result described above.
            """
            effective = None
            for alpha in source:
                if not _is_number(alpha):
                    raise TypeError('alphas must contain positive finite real numbers.')
                alpha = float(alpha)
                if not isfinite(alpha) or alpha <= 0:
                    raise ValueError('alphas must contain positive finite real numbers.')
                if effective is None:
                    effective = alpha
                else:
                    smaller, larger = min(effective, alpha), max(effective, alpha)
                    effective = smaller / (1.0 + smaller / larger)
                    if effective == 0.0:
                        raise FloatingPointError('The effective parameter underflowed to zero.')
                yield effective

        super().__init__(values())


def effective_alpha(alphas: Iterable[Number]) -> EffectiveAlpha:
    r"""Return EffectiveAlpha(alphas), supporting all Sequence arithmetic.
    
    Parameters
    ----------
    alphas: Iterable[Number]
        Alphas.
    
    
    Returns
    -------
    EffectiveAlpha
        Result described above.
    """
    return EffectiveAlpha(alphas)


__all__ += ['EffectiveAlpha', 'effective_alpha']
