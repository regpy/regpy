"""Shared structural types used only for annotations.

Vector mutability is represented by ``Vector`` and ``ReadOnlyVector`` from
``regpy.vecsps.base``. NumPy arrays and other non-vector arguments retain
ordinary backend types; no general immutability scheme is imposed on them.
"""
from __future__ import annotations

from typing import Any, Iterator, Protocol

