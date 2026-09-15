"""Regularization parameter sequences without plotting dependencies."""
from __future__ import annotations
from collections.abc import Iterable, Iterator
from numbers import Integral, Real
from math import isfinite
import numpy as np

__all__ = ['GeometricSequence', 'LinearSequence', 'powers_of_two', 'powers_of_b', 'natural_numbers', 'squares', 'effective_alpha']

def powers_of_two() -> Iterator[int]:
    """Yield ``1, 2, 4, 8, ...`` indefinitely."""
    n = 0
    while True:
        yield 2**n
        n += 1

def powers_of_b(b: int) -> Iterator[int]:
    """Yield integer powers b**n for n=0,1,2,... indefinitely.

    b must be a positive integer; b=1 produces only ones. Validation occurs
    on first advancement. Invalid types (including bool) raise TypeError;
    nonpositive bases raise ValueError. Python integers avoid overflow.
    """
    if isinstance(b, (bool, np.bool_)) or not isinstance(b, Integral):
        raise TypeError("b must be a positive integer.")
    if b <= 0:
        raise ValueError("b must be positive.")
    base = int(b)
    value = 1
    while True:
        yield value
        value *= base


def effective_alpha(alphas: Iterable[float]) -> Iterator[float]:
    """Yield effective regularization parameters for iterated Tikhonov.

    Parameters
    ----------
    alphas : iterable of float
        Positive finite parameters alpha_j, consumed lazily. Finite inputs
        produce one output per input; infinite generators are supported.

    Yields
    ------
    float
        alpha_eff_k defined by 1/alpha_eff_k = sum(1/alpha_j, j=1,...,k).

    Notes
    -----
    Iterated Tikhonov with parameters alpha_j can be interpreted as implicit
    time stepping for Showalter's equation with step sizes 1/alpha_j.
    The accumulated time is therefore 1/alpha_eff_k. This identifies the
    time scale, not equality of the discrete and continuous filters.

    A stable recurrence avoids explicitly forming reciprocals, whose sum
    can overflow: alpha_eff is the smaller of the previous effective value
    and alpha_j divided by one plus their smaller-to-larger ratio.

    Raises
    ------
    TypeError
        For non-real inputs, including booleans.
    ValueError
        For nonpositive or nonfinite inputs.
    FloatingPointError
        If the effective parameter underflows to zero.
    """
    effective = None
    for alpha in alphas:
        if isinstance(alpha, (bool, np.bool_)) or not isinstance(alpha, Real):
            raise TypeError("alphas must contain positive finite real numbers.")
        alpha = float(alpha)
        if not isfinite(alpha) or alpha <= 0:
            raise ValueError("alphas must contain positive finite real numbers.")
        if effective is None:
            effective = alpha
        else:
            smaller, larger = min(effective, alpha), max(effective, alpha)
            effective = smaller / (1.0 + smaller / larger)
            if effective == 0.0:
                raise FloatingPointError("The effective parameter underflowed to zero.")
        yield effective


def natural_numbers() -> Iterator[int]:
    """Yield ``1, 2, 3, ...`` indefinitely."""
    n = 1
    while True:
        yield n
        n += 1

def squares() -> Iterator[int]:
    """Yield ``0, 1, 4, 9, ...`` indefinitely."""
    n = 0
    while True:
        yield n**2
        n += 1

class GeometricSequence:
    r"""Iterator generating a geometric sequence
    
    Parameters
    ----------
    alpha0 : float
        :math:`\alpha_0` the initial regularization parameter 
    q : float
        Rate of the geometric sequence

    Notes
    ----- 
    Sequence defined recursively by
    
    .. math::
        \alpha_0 &= \alpha_0 \\
        \alpha_{n+1} &= q*\alpha_n
    """    
    def __init__(self, alpha0:float=1.0,q:float=0.5):
        self.alpha, self.alpha0, self.q = alpha0, alpha0, q

    def __iter__(self):
        return self

    def __next__(self)->float:
        result = self.alpha
        self.alpha *= self.q
        return result



class LinearSequence(Iterator[int]):
    r"""Iterator generating a linear sequence ``c, c+a, c+2a, ...``.
    
    Parameters
    ----------
    a : int, optional
        Difference between neighboring sequence elements. Defaults to ``1``.
    start : int, optional
        First element of the sequence. Defaults to ``0``.
    """    
    def __init__(self, a: int = 1, start: int = 0) -> None:
        self.a: int = a
        self.x: int = start

    def __iter__(self) -> LinearSequence:
        return self

    def __next__(self)->int:
        result = self.x
        self.x += self.a
        return result
    
