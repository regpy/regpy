"""Shared structural types used only for annotations.

Vector mutability is represented by ``Vector`` and ``ReadOnlyVector`` from
``regpy.vecsps.base``. NumPy arrays and other non-vector arguments retain
ordinary backend types; no general immutability scheme is imposed on them.
"""
from __future__ import annotations

from typing import Any, Iterator, Protocol


class Mask(Protocol):
    """Reusable backend-specific coordinate mask.

    A mask supports length and iteration. Entries can be booleans, array rows,
    or component masks (for a direct sum). The corresponding vector space's
    ``is_mask`` method validates shape and boolean contents; the structural
    type deliberately does not impose a flat representation.
    """

    def __len__(self) -> int:
        """Return the number of top-level entries."""
        ...

    def __iter__(self) -> Iterator[Any]:
        """Iterate over the backend's mask entries."""
        ...
