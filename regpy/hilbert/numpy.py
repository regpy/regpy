from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from types import NotImplementedType
    from numpy.typing import DTypeLike
    from regpy.vecsps.numpy import MeasureSpaceFcts, UniformGridFcts
    from regpy._typing import Mask
import numpy as np
from scipy.sparse import csc_matrix

from regpy import vecsps
from regpy.vecsps.maskable import MaskedVectorSpace
from regpy.util import memoized_property,Errors
from regpy.operators import Operator,PtwMultiplication,Pow,MatrixMultiplication,FourierTransform,CoordinateProjection

from .base import HilbertSpace

__all__ = ["L2MeasureSpaceFcts","L2UniformGridFcts","SobolevUniformGridFcts","HmDomain"]

class L2MeasureSpaceFcts(HilbertSpace):
    r"""`L2` implementation on a `regpy.vecsps.MeasureSpaceFcts`.
    
    Parameters
    ----------
    vecsp: vecsps.MeasureSpaceFcts
        Underlying discretization
    weights: np.ndarray | None [default: None]
        Weight in the norm.
    """

    def __init__(self, vecsp:vecsps.MeasureSpaceFcts, 
                 weights:np.ndarray|None=None) -> None:
        if not isinstance(vecsp,vecsps.MeasureSpaceFcts):
            raise TypeError(Errors.not_instance(vecsp,vecsps.MeasureSpaceFcts,"To define an L2MeasureSpaceFcts the vector space needs to be at least a MeasureSpaceFcts or a derivative of it."))
        super().__init__(vecsp)
        self.weights = weights

    @memoized_property
    def _gram_weights(self) -> np.ndarray | None:
        if self.weights is None:
            if np.all(self.vecsp.measure == 1):
                return None
            return self.vecsp.measure
        return np.multiply(self.weights, self.vecsp.measure)

    @memoized_property
    def gram(self)->PtwMultiplication:
        r"""Return the Gram operator representing the Hilbert inner product.
        
        
        Returns
        -------
        PtwMultiplication
            Result described above.
        """
        weights = self._gram_weights
        if weights is None:
            return self.vecsp.identity
        return PtwMultiplication(self.vecsp, weights)

    @memoized_property
    def cholesky(self) -> Operator:
        r"""Cholesky factor of the Gram operator.
        
        
        Returns
        -------
        Operator
            Constructed operator.
        """
        weights = self._gram_weights
        if weights is None:
            return self.vecsp.identity
        sqrt_weights = np.empty_like(
            weights, dtype=np.result_type(weights.dtype, float)
        )
        np.sqrt(weights, out=sqrt_weights)
        return PtwMultiplication(self.vecsp, sqrt_weights)

class L2UniformGridFcts(HilbertSpace):
    r"""`L2` implementation on a `regpy.vecsps.UniformGridFcts`, taking into account the volume
    element.
    
    Parameters
    ----------
    vecsp: vecsps.UniformGridFcts
        Underlying vector space.
    weights: np.ndarray | None [default: None]
        Weights.
    """

    def __init__(self, vecsp:vecsps.UniformGridFcts, weights:np.ndarray|None=None) -> None:
        if not isinstance(vecsp,vecsps.UniformGridFcts):
            raise TypeError(Errors.not_instance(vecsp,vecsps.UniformGridFcts,"To define an L2UniformGridFcts the vector space needs to be at a UniformGridFcts or a derivative of it."))
        super().__init__(vecsp)
        self.weights = weights

    @memoized_property
    def gram(self)-> PtwMultiplication:
        r"""Return the Gram operator representing the Hilbert inner product.
        
        
        Returns
        -------
        PtwMultiplication
            Result described above.
        """
        if self.weights is None:
            return self.vecsp.volume_elem * self.vecsp.identity
        else:
            return self.vecsp.volume_elem * PtwMultiplication(self.vecsp, self.weights)

    @memoized_property
    def cholesky(self) -> Operator:
        r"""Cholesky factor of the Gram operator. 
        
        
        Returns
        -------
        Operator
            Constructed operator.
        """
        if self.weights is None:
            return np.sqrt(self.vecsp.volume_elem) * self.vecsp.identity
        else:
            return np.sqrt(self.vecsp.volume_elem) * PtwMultiplication(self.vecsp, np.sqrt(self.weights))

class SobolevUniformGridFcts(HilbertSpace):
    r"""`Sobolev` implementation on a `regpy.vecsps.UniformGridFcts`.
    
    Parameters
    ----------
    vecsp: L2UniformGridFcts
        Grid on which to define the Sobolev space.
    index: float [default: 1.0]
        Sobolev index, Defaults: 1
    axes: list[int] | None [default: None]
        List of axes for which to compute in default all axes, Defaults: None
    """
    def __init__(self, 
                 vecsp:L2UniformGridFcts, 
                 index:float=1., 
                 axes:list[int]|None=None) -> None:
        if not isinstance(vecsp,vecsps.UniformGridFcts):
            raise TypeError(Errors.not_instance(vecsp,vecsps.UniformGridFcts,"To define an SobolevUniformGridFcts the vector space needs to be at a UniformGridFcts or a derivative of it."))
        super().__init__(vecsp)
        self.index = index
        if axes is None:
            axes = range(len(vecsp.shape_domain))
        self.axes = list(axes)

    def __eq__(self, other: object) -> bool | NotImplementedType:
        if isinstance(other, type(self)):
            return (
                self.vecsp == other.vecsp and
                self.index == other.index
            )
        else:
            return NotImplemented

    @memoized_property
    def gram(self)-> Operator:
        r"""Return the Gram operator representing the Hilbert inner product.
        
        
        Returns
        -------
        Operator
            Constructed operator.
        """
        from regpy.operators.convolution import BesselPotential        
        return BesselPotential(self.vecsp,2*self.index,convolution_axes=self.axes)

    @memoized_property
    def cholesky(self) -> Operator:
        r"""Cholesky factor of the Gram operators.
        
        
        Returns
        -------
        Operator
            Constructed operator.
        """
        from regpy.operators.convolution import BesselPotential        
        return BesselPotential(self.vecsp,self.index,convolution_axes=self.axes)

class HmDomain(HilbertSpace):
    r"""Implementation of a Sobolev space :math:`H^m(D)` for a subset :math:`D` of a `regpy.vecsps.UniformGridFcts` grid.
    Also supports a ``MaskedVectorSpace`` whose full space is a
    ``UniformGridFcts``. In that case the Gram and its inverse act on the
    *reduced masked coordinates*, rather than on the full grid.
    :math:`D` is characterized by a binary or integer-valued mask: `D={mask==1}`.
    `{mask==0}` are Dirichlet boundaries, and `{mask==-1}` Neumann boundaries.
    `mask` may also be boolean, in this case there are only Dirichlet boundaries.
    Boundary condition at the exterior boundaries are specified by `ext_bd_cond`, default is Neumann ('Neum')
    
    `m=index` is a non-negative integer, the order or index of the Sobolev space.
    The gram matrix is given by :math:`(\alpha I - \Delta)^{-m}`.
    
    By default it is assumed that the lengths in grid are given in physical dimensions,
    and a non-dimensionalization is carried out such that the largest side length (extent) of the grid is 1.
    
    If `weight` is specified, the Gram matrix will approximate :math:`(\alpha I-{weight}\Delta)^{-m}`. `weight` should be slowly varying.
    
    Parameters
    ----------
    vecsp: vecsps.UniformGridFcts | MaskedVectorSpace
        Underlying grid functions, or a masked space of such functions.
    mask: Mask | None [default: None]
        Mask to capture that subset :math:`D` on which the Sobolev space is defined. Can only contain 
        values `{-1,0,1}` or is a boolean. Shape has to match the shape of `vecsp`. In this case it is set to all True.
    h: str [default: 'normalized']
        The extent of the domain either given as a tuple or computed. Option key strings "physical" or 
        "normalized".
    index: int [default: 1]
        The Sobolev index :math:`m`.
    weight: float | None [default: None]
        Weights to be applied to Laplacian in the gram matrix definition.
    ext_bd_cond: str [default: 'Neum']
        Exterior boundary conditions to be applied. If not "Neum" takes Dirichlet boundary conditions.
    alpha: float [default: 1.0]
        Parameter when computing the gram matrix as :math:`(\alpha I - \Delta)^{-m}`.
    dtype: DTypeLike [default: float]
        Type of underlying grid.
    """

    def __init__(self,
                vecsp:vecsps.UniformGridFcts | MaskedVectorSpace,
                mask: Mask | None = None,
                h:str='normalized',
                index:int=1,
                weight: float | None=None,
                ext_bd_cond:str = 'Neum',
                alpha:float = 1.,
                dtype: DTypeLike = float) -> None:
        self._masked_domain = isinstance(vecsp, MaskedVectorSpace)
        full_vecsp = vecsp.get_full_space() if self._masked_domain else vecsp
        if not isinstance(full_vecsp, vecsps.UniformGridFcts):
            raise TypeError(Errors.not_instance(
                full_vecsp, vecsps.UniformGridFcts,
                "HmDomain requires a UniformGridFcts or a masked space of one."))

        if self._masked_domain:
            # The coefficient vector has no spatial shape; its support mask
            # belongs to the underlying full UniformGridFcts.
            support = np.asarray(vecsp.mask, dtype=bool)
            if support.shape != full_vecsp.shape:
                raise ValueError('The masked space has an incompatible support shape.')
            if mask is None:
                mask = support
            elif not np.array_equal(np.asarray(mask) == 1, support):
                raise ValueError(
                    'A supplied Hm0 mask must select exactly the masked '
                    'vector space support.')
        elif mask is None:
            mask = np.ones(full_vecsp.shape, dtype=bool)

        mask = np.asarray(mask)
        if full_vecsp.shape != mask.shape:
            raise ValueError(Errors.value_error("mask has to have the same shape as the vector space"),self)
        if not isinstance(index,int):
            raise TypeError(Errors.not_instance(index,int,add_info="index has to be a non-negative integer!"))
        elif index<0:
            raise ValueError(Errors.value_error("The index in an HmDomain has to be a non-negative integer!"))
        super().__init__(vecsp)


        self.ndim = mask.ndim
        """Dimension of vector space
        """
        if type(h) == tuple:
            self.h_val = h
        elif full_vecsp is None:
            self.h_val=1./np.max(mask.shape)*np.ones((self.ndim,))
        elif h=='physical':
            self.h_val = full_vecsp.extents/(np.array(full_vecsp.shape)-1)
        elif h=='normalized':
            self.h_val = (2.*np.pi/np.max(full_vecsp.extents))* (full_vecsp.extents/(np.array(full_vecsp.shape)-1))
        else:
            raise NotImplemented

        self.index = index
        """Sobolev index.
        """
        self.alpha = alpha
        """Regularizer for Gram matrix.
        """
        self.mask = np.equal(mask, 1)
        """Mask to determine the subspace D.
        """
        self.dtype = full_vecsp.dtype
        """Type of the underlying grid. 
        """

        self.proj = CoordinateProjection(
                full_vecsp,
                self.mask
            )

        mask_values = np.asarray(mask, dtype=np.int8)
        mask_padded = np.pad(
            mask_values, 1, mode="constant",
            constant_values=-1 if ext_bd_cond == "Neum" else 0,
        )

        self.G = np.zeros(mask_padded.shape, dtype=np.intp)
        interior_ind = mask_padded == 1
        self.G[interior_ind] = 1 + np.arange(
            np.count_nonzero(interior_ind), dtype=np.intp
        )
        self.G[mask_padded == -1] = -1

        if weight is None:
            self.weight = None
        elif weight.shape == mask.shape:
            self.weight = np.pad(weight,1,'edge')
        else:
            raise ValueError(Errors.value_error("weight has to have the same shape as the vector space or be None",self))

    @memoized_property
    def _I_minus_Delta_matrix(self)->csc_matrix:
        r"""Assemble ``alpha * I - Delta`` on the masked domain.
        
        Index and value chunks are collected separately and concatenated once.
        This avoids the quadratic copying caused by repeatedly extending NumPy
        arrays inside the neighbor loop.
        
        
        Returns
        -------
        csc_matrix
            Result described above.
        """
        weights = None if self.weight is None else self.weight.ravel()
        labels = self.G.ravel()
        points = np.flatnonzero(labels > 0)
        size = points.size

        row_chunks = []
        column_chunks = []
        value_chunks = []
        diagonal = np.full(size, self.alpha, dtype=float)

        strides = [1]
        for axis in range(self.ndim - 1, 0, -1):
            strides.append(strides[-1] * self.G.shape[axis])

        for direction, spacing in enumerate(self.h_val):
            inv_spacing_sq = 1.0 / spacing**2
            for offset in (-strides[direction], strides[direction]):
                neighbors = labels[points + offset]

                interior = np.flatnonzero(neighbors > 0)
                if interior.size:
                    point_indices = points[interior]
                    rows = labels[point_indices] - 1
                    columns = neighbors[interior] - 1
                    if weights is None:
                        entries = np.full(
                            interior.size, inv_spacing_sq, dtype=float
                        )
                    else:
                        entries = np.asarray(
                            weights[point_indices], dtype=float
                        ).copy()
                        entries *= weights[point_indices + offset]
                        np.sqrt(entries, out=entries)
                        entries *= inv_spacing_sq

                    row_chunks.append(rows)
                    column_chunks.append(columns)
                    value_chunks.append(-entries)
                    diagonal[rows] += entries

                dirichlet = np.flatnonzero(neighbors == 0)
                if dirichlet.size:
                    point_indices = points[dirichlet]
                    rows = labels[point_indices] - 1
                    if weights is None:
                        entries = np.full(
                            dirichlet.size, inv_spacing_sq, dtype=float
                        )
                    else:
                        entries = np.asarray(
                            weights[point_indices], dtype=float
                        ).copy()
                        entries *= weights[point_indices + offset]
                        np.sqrt(entries, out=entries)
                        entries *= inv_spacing_sq
                    diagonal[rows] += entries

        diagonal_indices = labels[points] - 1
        row_chunks.append(diagonal_indices)
        column_chunks.append(diagonal_indices)
        value_chunks.append(diagonal)

        rows = np.concatenate(row_chunks)
        columns = np.concatenate(column_chunks)
        values = np.concatenate(value_chunks)
        return csc_matrix((values, (rows, columns)), shape=(size, size))

    def I_minus_Delta(self)->csc_matrix:
        r"""Return the cached sparse matrix ``alpha * I - Delta``.
        
        
        Returns
        -------
        csc_matrix
            Result described above.
        """
        return self._I_minus_Delta_matrix

    @memoized_property
    def _I_minus_Delta_operator(self)->Operator:
        # Hm0 is defined on the reduced masked coefficients themselves. On a
        # full grid HmDomain retains the original projection/sandwich model.
        reduced_domain = self.vecsp if self._masked_domain else self.proj.codomain
        return MatrixMultiplication(
            # Keep native grid entries here. MatrixMultiplication handles
            # interleaved real coordinates when the complex space is realified.
            self.I_minus_Delta().astype(self.dtype, copy=False),
            inverse="superLU",
            domain=reduced_domain,
            codomain=reduced_domain,
            dtype=self.dtype,
        )

    @memoized_property
    def gram(self)->Operator:
        r"""Return the Gram operator representing the Hilbert inner product.
        
        
        Returns
        -------
        Operator
            Constructed operator.
        """
        mat = Pow(self._I_minus_Delta_operator, self.index)
        if self._masked_domain:
            return mat
        gram = self.proj.adjoint * mat * self.proj
        gram.inverse = self.proj.adjoint * mat.inverse * self.proj
        return gram

    @memoized_property
    def cholesky(self) -> Operator:
        r"""Cholesky.
        
        
        Returns
        -------
        Operator
            Constructed operator.
        """
        if self.index % 2 == 0:
            mat = Pow(self._I_minus_Delta_operator, self.index // 2)
            return mat if self._masked_domain else self.proj.adjoint * mat * self.proj
        raise NotImplementedError(
            "Cholesky factorization is only implemented for even indices so far."
        )
