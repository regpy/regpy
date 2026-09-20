"""Coordinate masks, restrictions and maskable direct sums.

Masks are reusable boolean containers. Default masks are flat boolean arrays
in iter_coefficients order. NumPy spaces use the vector shape; direct sums use
TupleMask with one mask per summand. No ordering is imposed on complex spaces.
Restricted vectors are flat arrays in the selected coefficient order.
"""
from __future__ import annotations

import numpy as np
from .base import VectorSpaceBase, DirectSum, TupleVector

__all__ = ["MaskableVectorSpace", "MaskedVectorSpace", "TupleMask",
           "MaskableDirectSum"]


class MaskableVectorSpace(VectorSpaceBase):
    """Vector space supporting coordinate selection through iterable masks.

    Backends provide iter_coefficients and set_coefficients. Defaults use
    flat masks and snapshot sources before writes, including aliased copies.
    """

    @property
    def is_ordered(self) -> bool:
        return not self.is_complex

    def is_mask(self, mask) -> bool:
        if isinstance(mask, np.ndarray):
            return mask.dtype == bool and mask.shape == (self._native_dimension,)
        # Reject consumable iterators: validation must not consume a mask.
        try:
            if iter(mask) is mask:
                return False
            return len(mask) == self._native_dimension and all(
                isinstance(v, (bool, np.bool_)) for v in mask)
        except TypeError:
            return False

    def iter_mask(self, mask):
        if not self.is_mask(mask):
            raise ValueError("Mask must contain one boolean per coordinate.")
        return iter(mask)

    def mask_from_iterable(self, values):
        values = list(values)
        if len(values) != self._native_dimension or any(
                not isinstance(v, (bool, np.bool_)) for v in values):
            raise ValueError("Expected one boolean per coordinate.")
        return np.asarray(values, dtype=bool)

    def nrTrues(self, mask) -> int:
        return sum(bool(v) for v in self.iter_mask(mask))

    def _logical(self, x, y, operation):
        return self.mask_from_iterable(
            operation(a, b) for a, b in zip(self.iter_mask(x), self.iter_mask(y),
                                          strict=True))

    def logical_and(self, x, y):
        return self._logical(x, y, lambda a, b: a and b)

    def logical_or(self, x, y):
        return self._logical(x, y, lambda a, b: a or b)

    def logical_xor(self, x, y):
        return self._logical(x, y, lambda a, b: bool(a) != bool(b))

    def logical_not(self, x):
        return self.mask_from_iterable(not a for a in self.iter_mask(x))

    def _comparison(self, x, predicate, *, ordered=False):
        if ordered and not self.is_ordered:
            raise TypeError("Coordinate ordering is unavailable on this space.")
        return self.mask_from_iterable(predicate(v) for v in self._iter_native_coefficients(x))

    def IfPos(self, x):
        """Mask of strictly positive coefficients."""
        return self._comparison(x, lambda v: v > 0, ordered=True)

    def _compare_pair(self, x, y, predicate, *, ordered=False):
        if ordered and not self.is_ordered:
            raise TypeError("Coordinate ordering is unavailable on this space.")
        return self.mask_from_iterable(predicate(a, b) for a, b in zip(
            self._iter_native_coefficients(x), self._iter_native_coefficients(y), strict=True))

    def eq_mask(self, x, y):
        """Return the mask x == y in native coefficient coordinates."""
        return self._compare_pair(x, y, lambda a, b: a == b)

    def geq_mask(self, x, y):
        """Return the mask x >= y; requires an ordered space."""
        return self._compare_pair(x, y, lambda a, b: a >= b, ordered=True)

    def copyto(self, x, y, where=None) -> None:
        if where is None:
            return super().copyto(x, y)
        selection = list(self.iter_mask(where))
        source = list(self._iter_native_coefficients(y))
        target = list(self._iter_native_coefficients(x))
        self._set_native_coefficients(x, (b if m else a for a, b, m in
                                 zip(target, source, selection, strict=True)))

    def fill(self, x, value, where=None) -> None:
        if where is None:
            return super().fill(x, value)
        if not np.isscalar(value):
            raise TypeError("fill requires a scalar.")
        target = list(self._iter_native_coefficients(x))
        self._set_native_coefficients(x, (value if m else a for a, m in
                                 zip(target, self.iter_mask(where), strict=True)))

    def restrict(self, x, mask):
        """Copy selected native coefficients to a one-dimensional vector."""
        return np.asarray([v for v, m in zip(self._iter_native_coefficients(x),
                          self.iter_mask(mask), strict=True) if m],
                          dtype=complex if self.is_complex else float)

    def extend(self, x, mask):
        """Copy selected coefficients into the full space, zero elsewhere."""
        space = self.get_masked_space(mask)
        if x not in space:
            raise ValueError("Vector does not belong to the masked space.")
        selected = iter(x)
        result = self.zeros()
        self._set_native_coefficients(result, (next(selected) if m else 0
                                      for m in self.iter_mask(mask)))
        return result

    def get_masked_space(self, mask):
        return MaskedVectorSpace(self, mask)

    def masked_space(self, mask):
        """Compatibility spelling of get_masked_space."""
        return self.get_masked_space(mask)


class MaskedVectorSpace(MaskableVectorSpace):
    """Flat coefficient space retaining its full space and an immutable mask.

    Construct through full_space.get_masked_space(mask). Restriction copies;
    extension fills unselected coefficients with zero. The selected space's
    Euclidean pairing makes restriction and extension algebraic adjoints.
    """

    def __init__(self, full_space: MaskableVectorSpace, mask):
        if not isinstance(full_space, MaskableVectorSpace):
            raise TypeError("full_space must be maskable.")
        # Rebuild to detach the mask from caller-owned storage.
        self.full_space = full_space
        self.mask = full_space.mask_from_iterable(full_space.iter_mask(mask))
        if isinstance(self.mask, np.ndarray):
            self.mask.flags.writeable = False
        self.shape = (full_space.nrTrues(self.mask),)
        super().__init__(np.ndarray, self.shape[0],
                         complex=full_space.is_complex, type=np,
                         as_real_space=full_space.as_real_space)
        self.dtype = np.dtype(complex if self.is_complex else float)

    def get_full_space(self):
        return self.full_space

    def restrict(self, x, mask=None):
        """Restrict from the parent space, or select coordinates of this space.

        Without mask, x belongs to get_full_space() and the stored mask is
        used. With mask, x belongs to this space and mask selects a further
        subset of its coordinates (the MaskableVectorSpace interface).
        """
        if mask is not None:
            return super().restrict(x, mask)
        return self.full_space.restrict(x, self.mask)

    def extend(self, x, mask=None):
        """Zero-extend into the parent space, or into this space using mask.

        Without mask, x belongs to this space and the result belongs to
        get_full_space(). With mask, x belongs to get_masked_space(mask)
        and the result belongs to this space.
        """
        if mask is not None:
            return super().extend(x, mask)
        return self.full_space.extend(x, self.mask)

    def __contains__(self, x):
        return (isinstance(x, np.ndarray) and x.shape == self.shape
                and np.can_cast(x.dtype, self.dtype, casting="safe"))

    def empty(self):
        return np.empty(self.shape, dtype=self.dtype)

    def zeros(self, read_only=False):
        x = np.zeros(self.shape, dtype=self.dtype)
        return self.read_only(x) if read_only else x

    def ones(self, read_only=False):
        x = np.ones(self.shape, dtype=self.dtype)
        return self.read_only(x) if read_only else x

    def read_only(self, x):
        if x not in self:
            raise ValueError("Vector outside masked space.")
        result = x.view()
        result.flags.writeable = False
        return result

    def _iter_native_coefficients(self, x):
        if x not in self:
            raise ValueError("Vector outside masked space.")
        return iter(x)

    def _set_native_coefficients(self, x, coefficients):
        if x not in self:
            raise ValueError("Vector outside masked space.")
        values = np.asarray(list(coefficients))
        if values.shape != self.shape:
            raise ValueError("Wrong number of coefficients.")
        np.copyto(x, values, casting="same_kind")

    @property
    def ndim(self):
        return len(self.shape)

    def compatible(self, other):
        return (isinstance(other, MaskedVectorSpace) and self.shape == other.shape
                and self.is_complex == other.is_complex
                and self.as_real_space == other.as_real_space)

    def _coordinate_space(self):
        # Delayed import: NumPyVectorSpace itself derives from MaskableVectorSpace.
        from .numpy import NumPyVectorSpace
        return NumPyVectorSpace(self.shape, dtype=self.dtype, as_real_space=self.as_real_space)

    def flatten(self, x):
        return self._coordinate_space().flatten(x)

    def fromflat(self, x):
        return self._coordinate_space().fromflat(x)



    def iter_basis(self):
        return self._coordinate_space().iter_basis()

    def real_space(self):
        return self.full_space.real_space().get_masked_space(self.mask)

    def complex_space(self):
        return self.full_space.complex_space().get_masked_space(self.mask)

    def __eq__(self, other):
        return (isinstance(other, MaskedVectorSpace)
                and self.as_real_space == other.as_real_space
                and self.full_space == other.full_space
                and tuple(self.full_space.iter_mask(self.mask))
                == tuple(other.full_space.iter_mask(other.mask)))


class TupleMask(tuple):
    """Immutable tuple of component masks, with componentwise logic."""

    def __new__(cls, masks):
        parts = []
        for mask in masks:
            if isinstance(mask, TupleMask):
                parts.append(mask)
            else:
                part = np.array(mask, copy=True)
                if part.dtype != bool:
                    raise TypeError("TupleMask components must be boolean.")
                part.flags.writeable = False
                parts.append(part)
        return super().__new__(cls, parts)

    def _binary(self, other, op):
        if not isinstance(other, TupleMask) or len(self) != len(other):
            raise ValueError("Incompatible tuple masks.")
        if any(getattr(a, "shape", None) != getattr(b, "shape", None)
               for a, b in zip(self, other)):
            raise ValueError("Incompatible component mask shapes.")
        return TupleMask(op(a, b) for a, b in zip(self, other, strict=True))

    def __and__(self, other):
        return self._binary(other, lambda a, b: a & b)

    def __or__(self, other):
        return self._binary(other, lambda a, b: a | b)

    def __xor__(self, other):
        return self._binary(other, lambda a, b: a ^ b)

    def __invert__(self):
        return TupleMask(~a for a in self)

    def __bool__(self):
        raise ValueError("TupleMask truth is ambiguous; use all() or any().")

    def all(self):
        return all(bool(a.all()) for a in self)

    def any(self):
        return any(bool(a.any()) for a in self)

    def sum(self):
        return sum(int(a.sum()) for a in self)


class MaskableDirectSum(DirectSum, MaskableVectorSpace):
    """Direct sum with TupleMask masks and TupleVector full vectors."""

    def __init__(self, *summands, flatten=False):
        if not all(isinstance(s, MaskableVectorSpace) for s in summands):
            raise TypeError("All summands must be maskable.")
        super().__init__(*summands, flatten=flatten)

    @property
    def is_ordered(self):
        return all(s.is_ordered for s in self.summands)

    def is_mask(self, mask):
        return (isinstance(mask, TupleMask) and len(mask) == len(self.summands)
                and all(s.is_mask(m) for s, m in zip(self.summands, mask)))

    def nrTrues(self, mask):
        if not self.is_mask(mask):
            raise ValueError("Invalid direct-sum mask.")
        return sum(s.nrTrues(m) for s, m in zip(self.summands, mask))

    def iter_mask(self, mask):
        if not self.is_mask(mask):
            raise ValueError("Invalid direct-sum mask.")
        return (v for s, m in zip(self.summands, mask) for v in s.iter_mask(m))

    def mask_from_iterable(self, values):
        from itertools import islice
        iterator = iter(values)
        result = TupleMask(s.mask_from_iterable(islice(iterator, s._native_dimension))
                           for s in self.summands)
        if next(iterator, None) is not None:
            raise ValueError("Too many mask entries.")
        return result

    def _compare_pair(self, x, y, predicate, *, ordered=False):
        if x not in self or y not in self:
            raise ValueError("Vectors outside the direct sum.")
        if ordered and not self.is_ordered:
            raise TypeError("Coordinate ordering is unavailable on this space.")
        return TupleMask(s._compare_pair(a, b, predicate, ordered=ordered)
                         for s, a, b in zip(self.summands, x, y, strict=True))

    def copyto(self, x, y, where=None):
        if where is None:
            return DirectSum.copyto(self, x, y)
        if x not in self or y not in self or not self.is_mask(where):
            raise ValueError("Incompatible vectors or mask.")
        x._require_writable("copyto")
        # Snapshot the entire source to handle aliases across components.
        source = self.empty()
        DirectSum.copyto(self, source, y)
        for s, a, b, m in zip(self.summands, x, source, where, strict=True):
            s.copyto(a, b, where=m)

    def fill(self, x, value, where=None):
        if where is None:
            return DirectSum.fill(self, x, value)
        if x not in self or not self.is_mask(where):
            raise ValueError("Incompatible vector or mask.")
        x._require_writable("fill")
        for s, a, m in zip(self.summands, x, where, strict=True):
            s.fill(a, value, where=m)
