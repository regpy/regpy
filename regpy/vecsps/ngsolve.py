# Revision: 20260920-ngs-vector-size-fix
r"""Finite element vector spaces using NGSolve

This module implements a `regpy.vecsps.VectorSpaceBase` instance for NGSolve spaces. This gives the basic
interface to use FES spaces defined in `ngsolve` to be used as `regpy.vecsps.VectorSpaceBase`/s in `regpy`.  
Operators are using such spaces are implemented in the `regpy.operators.ngsolve` module. Hilbert spaces 
and Functionals defined on such spaces can be found in `regpy.hilbert.ngsolve` and `regpy.functionals.ngsolve`
respectively. 
"""
from __future__ import annotations

from typing import Literal, overload

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Any
    from typing_extensions import Self
    from numpy.typing import ArrayLike
    from regpy.vecsps.base import Vector
    from regpy.functionals.base import Functional
    from regpy._typing import Mask

__all__ = ['NgsBaseVector','NgsVectorSpace',"NgsVectorSpaceWithInnerProduct"]

from copy import copy,deepcopy
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Optional
from warnings import warn

import ngsolve as ngs
import numpy as np
from scipy.sparse.linalg import splu
from scipy.sparse.linalg import spsolve
from scipy.sparse import csc_matrix, diags
from pyngcore.pyngcore import BitArray

from regpy.util import Errors, ClassLogger

from .base import ReadOnlyVector, Scalar, VectorSpaceBase
from .maskable import MaskableVectorSpace


@dataclass 
class NgsBaseVector:
    vec: ngs.la.BaseVector | ngs.la.DynamicVectorExpression
    make_copy: Optional[bool] = field(default=False)
    read_only: bool = field(default=False)

    __array_ufunc__ = None

    log = ClassLogger()

    def copy(self) -> Self:
        r"""Copy.
        
        
        Returns
        -------
        Self
            Result described above.
        """
        return copy(self)
    
    @property
    def is_complex_dtype(self) -> bool:
        r"""Is complex dtype.
        
        
        Returns
        -------
        bool
            Whether the stated condition is satisfied.
        """
        return self.vec.is_complex

    def __post_init__(self) -> None:
        if not isinstance(self.read_only, (bool, np.bool_)):
            raise TypeError(
                f"read_only must be boolean. Got {type(self.read_only)}."
            )
        self.read_only = bool(self.read_only)
        self.is_dynamic = False
        if isinstance(self.vec,ngs.la.BaseVector):
            if self.make_copy:
                self.vec = self.vec.Copy()
            self.size = self.vec.size
            self.is_complex = self.vec.is_complex
            pass
        elif isinstance(self.vec,ngs.la.DynamicVectorExpression):
            if self.make_copy:
                self.log.debug("To make a Copy of a Dynamic VectorExpression it has to be evaluated")
                self.vec = self.vec.Evaluate()
                self.size = self.vec.size
                self.is_complex = self.vec.is_complex
            else:
                self.is_dynamic = True
                # self.vec = self.vec.Evaluate()
                vec = self.vec.CreateVector()
                self.size = vec.size
                self.is_complex = vec.is_complex
            pass
        else:
            raise TypeError(Errors.type_error("Could not treat {} type only ngs.la.BaseVector or ngs.la.DynamicVectorExpression".format(type(self.vec))))

    def _require_writable(self, operation: str) -> None:
        if self.read_only:
            raise ValueError(
                f"Cannot perform {operation} on a read-only NgsBaseVector."
            )

    def conj(self) -> Functional:
        r"""Conj.
        
        
        Returns
        -------
        Functional
            Constructed functional.
        """
        if self.is_dynamic:
            self.log.debug("To make a conjugate of a Dynamic VectorExpression it has to be evaluated")
            self.vec = self.vec.Evaluate()
            self.is_dynamic = False
        if self.is_complex_dtype:
            z = self.vec.CreateVector()
            z.FV().NumPy()[:] = self.vec.FV().NumPy().conj()
            return NgsBaseVector(z,)
        else:
            return self.copy()
    
    @property
    def real(self, convert2real_vec: bool = True) -> NgsBaseVector | Vector:
        r"""Real part.
        
        Parameters
        ----------
        convert2real_vec: bool [default: True]
            Convert2real vec.
        
        
        Returns
        -------
        NgsBaseVector | Vector
            Result described above.
        """
        if self.is_dynamic:
            self.log.debug("To make a real of a Dynamic VectorExpression it has to be evaluated")
            self.vec = self.vec.Evaluate()
            self.is_dynamic = False
        if self.is_complex_dtype:
            if convert2real_vec:
                z = ngs.la.BaseVector(size = self.size)
                z.FV().NumPy()[:] = self.vec.FV().NumPy().real 
                return NgsBaseVector(z)
            else:
                z = self.vec.CreateVector()
                z.FV().NumPy()[:] = self.vec.FV().NumPy().real 
                return NgsBaseVector(z) 
        return self.copy()
    
    @property
    def imag(self, convert2real_vec: bool = True) -> NgsBaseVector:
        r"""Imagqinary part.
        
        Parameters
        ----------
        convert2real_vec: bool [default: True]
            Convert2real vec.
        
        
        Returns
        -------
        NgsBaseVector
            Result described above.
        """
        if self.is_dynamic:
            self.log.debug("To make a imag of a Dynamic VectorExpression it has to be evaluated")
            self.vec = self.vec.Evaluate()
            self.is_dynamic = False
        if self.is_complex_dtype:
            if convert2real_vec:
                z = ngs.la.BaseVector(size = self.size)
                z.FV().NumPy()[:] = 0
                z.FV().NumPy()[:] = self.vec.FV().NumPy().imag 
                return NgsBaseVector(z)
            else:
                z = self.vec.CreateVector()
                z.FV().NumPy()[:] = self.vec.FV().NumPy().imag 
                return NgsBaseVector(z)
        z = self.vec.CreateVector()
        z.FV().NumPy()[:] = 0
        return NgsBaseVector(z)
    
    def to_imag(self) -> Self | NgsBaseVector:
        r"""To imag.
        
        
        Returns
        -------
        Self | NgsBaseVector
            Result described above.
        """
        if self.is_complex:
            return self
        else:
            if self.is_dynamic:
                self.log.debug("For to_imag of a Dynamic VectorExpression it has to be evaluated")
                self.vec = self.vec.Evaluate()
                self.is_dynamic = False
            z = ngs.la.BaseVector(size = self.size, complex = True)
            z.FV().NumPy()[:] = self.vec.FV().NumPy() + 1j*0
            return NgsBaseVector(z)

    def __iadd__(self,other: ReadOnlyVector | Scalar) -> Self:
        self._require_writable("in-place addition")
        if not isinstance(other,NgsBaseVector) or other.size != self.size:
            raise ValueError(Errors.value_error("Adding NgsBaseVector only supported for NgsBaseVector of identical size!"))
        if self.is_complex and not other.is_complex:
            self.vec += other.to_imag().vec
        elif not self.is_complex and other.is_complex:
            warn("Adding complex vector to real vector, converting real to complex.")
            self.vec += other.real
        else:
            self.vec += other.vec
        return self

    def __isub__(self,other: ReadOnlyVector | Scalar) -> Self:
        self._require_writable("in-place subtraction")
        if not isinstance(other,NgsBaseVector) or other.size != self.size:
            raise ValueError(Errors.value_error("Subtracting NgsBaseVector only supported for NgsBaseVector of identical size!"))
        if self.is_complex and not other.is_complex:
            self.vec -= other.to_imag().vec
        elif not self.is_complex and other.is_complex:
            warn("Adding complex vector to real vector, converting real to complex.")
            self.vec -= other.real
        else:
            self.vec -= other.vec
        return self
    
    def __imul__(self,other: ReadOnlyVector | Scalar) -> Self:
        self._require_writable("in-place multiplication")
        if self.is_dynamic:
            self.log.debug("To have in-place multiplication of a Dynamic VectorExpression it has to be evaluated")
            self.vec = self.vec.Evaluate()
            self.is_dynamic = False
        if not isinstance(other,(float,int,complex)):
            raise ValueError(Errors.value_error("Multiplying NgsBaseVector only supported for scalars (int, float, or complex)!"))
        self.vec.data *= other
        return self
    
    def __itruediv__(self,other: ReadOnlyVector | Scalar) -> Self:
        self._require_writable("in-place division")
        if self.is_dynamic:
            self.log.debug("To have in-place division of a Dynamic VectorExpression it has to be evaluated")
            self.vec = self.vec.Evaluate()
            self.is_dynamic = False
        if not isinstance(other,(float,int,complex)):
            raise ValueError(Errors.value_error("Dividing NgsBaseVector only supported for scalars (int, float, or complex)!"))
        self.vec.data /= other
        return self
    
    def __add__(self,other: ReadOnlyVector | Scalar) -> NgsBaseVector:
        if not isinstance(other,NgsBaseVector) or other.size != self.size:
            raise ValueError(Errors.value_error("Adding NgsBaseVector only supported for NgsBaseVector of identical size!"))
        if self.is_complex and not other.is_complex:
            return self + other.to_imag()
        if not self.is_complex and other.is_complex:
            warn("Adding complex vector to real vector, converting real to complex.")
            return self + other.real
        return NgsBaseVector(self.vec + other.vec)
    
    def __radd__(self,other: ReadOnlyVector | Scalar) -> NgsBaseVector:
        return self + other
    
    def __sub__(self,other: ReadOnlyVector | Scalar) -> NgsBaseVector:
        return self + (-1*other)
    
    def __rsub__(self,other: ReadOnlyVector | Scalar) -> NgsBaseVector:
        return (-1*self) + other
    
    def __neg__(self) -> NgsBaseVector:
        return -1*self
    
    def __mul__(self,other: ReadOnlyVector | Scalar) -> NgsBaseVector:
        from regpy.operators.base import Operator,PtwMultiplication
        if isinstance(other,float) or isinstance(other,int) or isinstance(other,complex):
            return NgsBaseVector(other * self.vec)
        elif isinstance(other,Operator):
            return PtwMultiplication(other.codomain, self) * other
        else:
            raise NotImplementedError(Errors.generic_message(f"Multiplication of TupleVector with {type(other)} is not defined. It has to be either a number eg float, int or complex or an Operator."))
        
    def __rmul__(self,other: ReadOnlyVector | Scalar) -> NgsBaseVector:
        return self * other

    def __truediv__(self,other: ReadOnlyVector | Scalar) -> NgsBaseVector:
        if not isinstance(other,(float,int,complex)):
            raise ValueError(Errors.value_error("Dividing NgsBaseVector only supported for scalars (int, float, or complex)!"))
        return NgsBaseVector((1/other)*self.vec)
    
    def __getitem__(self,i: int) -> Scalar | NgsBaseVector:
        if self.is_dynamic:
            self.log.debug("To get items the Dynamic VectorExpression it has to be evaluated")
            self.vec = self.vec.Evaluate()
            self.is_dynamic = False
        if isinstance(i,BitArray):
            v = self.vec.CreateVector()
            v[i] = self.vec
            return NgsBaseVector(v)
        elif isinstance(i,int):
            return self.vec[i]
        else:
            return NgsBaseVector(self.vec[i])
    
    def __setitem__(self,i: int,val: Scalar | ReadOnlyVector) -> None:
        self._require_writable("item assignment")
        if self.is_dynamic:
            self.log.debug("To set items the DynamicVectorExpression it has to be evaluated")
            self.vec = self.vec.Evaluate()
            self.is_dynamic = False
        if isinstance(val, NgsBaseVector) and self.size == val.size:
            self.vec[i] = val.vec
        else:
            try:
                self.vec[i] = val
            except TypeError:
                raise TypeError(Errors.type_error(f"Not able to set {val} to NgsBaseVector. It has to be either an NgsBaseVector of same size or Something compatible to set to an ngsolve.la.BaseVector."))

    def __iter__(self) -> Iterator[Any]:
        if self.is_dynamic:
            self.log.debug("To iterate the DynamicVectorExpression it has to be evaluated")
            self.vec = self.vec.Evaluate()
            self.is_dynamic = False
        return self.vec

    def iter_basis(self) -> Iterator[ReadOnlyVector]:
        r"""Iter basis.
        
        
        Yields
        ------
        ReadOnlyVector
            Result described above.
        """
        v = NgsBaseVector(self.vec.CreateVector())
        for i in range(self.size):
            v[i] = 1 
            yield v
            if self.is_complex:
                v[i] = 1j
                yield v
            v[i] = 0                

    # Bitwise operators are deliberately not defined for NGSolve vectors.
    # Use the NgsVectorSpace/MaskableVectorSpace mask API instead.

    def __copy__(self) -> Self:
        if self.is_dynamic:
            self.log.debug("To copy the DynamicVectorExpression it has to be evaluated")
            self.vec = self.vec.Evaluate()
            self.is_dynamic = False
        return NgsBaseVector(self.vec.Copy())
    
    def __deepcopy__(self, memo: dict[int, Any]) -> Self:
        if self.is_dynamic:
            self.log.debug("To copy the DynamicVectorExpression it has to be evaluated")
            self.vec = self.vec.Evaluate()
            self.is_dynamic = False
        return NgsBaseVector(self.vec.Copy())


class NgsVectorSpace(MaskableVectorSpace):
    r"""A vector space wrapping an `ngsolve.FESpace`.
    
    Parameters
    ----------
    fes: ngs.FESpace
        The wrapped NGSolve vector space.
    bdr: str | None [default: None]
        Boundary of the NGSolve vector space.
    as_real_space: bool [default: True]
        Whether to use real scalars. With complex storage, True treats the
        space as real and counts real and imaginary coordinates separately.
    """

    def __init__(self, fes: ngs.FESpace, bdr: str | None=None, as_real_space: bool=True) -> None:
        if not isinstance(fes, ngs.FESpace):
            raise TypeError(Errors.not_instance(fes,ngs.FESpace))
        self.fes = fes
        self.bdr = bdr
        self.shape = (fes.ndof,)
        super().__init__(vec_type=NgsBaseVector, dimension=fes.ndof,
                         complex=fes.is_complex, as_real_space=as_real_space)
        # Checks if FES is Vector valued and stores the dimension in self.codim
        from netgen.libngpy._meshing import NgException
        try:
            self.codim = len(fes.components)
            if (isinstance(fes,ngs.VectorH1),isinstance(fes,ngs.VectorL2),isinstance(fes,ngs.VectorValued)):
                self._fes_util = ngs.VectorL2(self.fes.mesh, order = 0, dim = fes.dim, complex = self.is_complex)
            else:
                l_fes = []
                for f in fes.components:
                    if isinstance(f,ngs.ProductSpace):
                        if (isinstance(f,ngs.VectorH1),isinstance(f,ngs.VectorL2),isinstance(f,ngs.VectorValued)):
                            l_fes.append(ngs.VectorL2(self.fes.mesh, order = 0, complex = self.is_complex))
                        else:
                            raise ValueError(Errors.value_error(f""" 
        To construct a NgsVectorSpace for FES which has components which are ProductSpaces 
        the spaces have to be either VectorH1, VectorL2 or VectorValued!
        The component that was not able to process is:
            f = {f}
        in the FES:
            fes = {fes}"""))
                    else:
                        l_fes.append(ngs.L2(self.fes.mesh, order=0, dim = f.dim, complex = self.is_complex))
                self._fes_util = ngs.ProductSpace(*l_fes)
            self.product_space = True
        except NgException:
            gf = ngs.GridFunction(fes)
            if len(gf.dims) == 0:
                self.codim = 1
            elif len(gf.dims) == 1:
                self.codim = gf.dims[0]
            else:
                raise ValueError(Errors.value_error(f"Currently we cannot deal with matrix valued fems."))
            self.product_space = False
            self._fes_util = ngs.L2(self.fes.mesh, order=0, complex = self.is_complex)
        except ValueError:
            self.log.warning("Tried to initialize with a product space of product spaces, which are not VectorH1, VectorL2 or VectorValued. Thus fes_util is not available and thus random generator will not work!")
            self._fes_util = None
        if self._fes_util is not None:
            self._gfu_util = ngs.GridFunction(self._fes_util)
        self._gfu_fes = ngs.GridFunction(self.fes)
        self._help_x = NgsBaseVector(self._gfu_fes.vec)
        self._no_pickle = {*self._no_pickle,"fes"}
        self.mass_matrix_cholesky = None

    @overload
    def zeros(self, read_only: Literal[False]=False) -> NgsBaseVector:
        ...

    @overload
    def zeros(self, read_only: Literal[True]) -> ReadOnlyVector:
        ...

    @overload
    def zeros(self, read_only: bool) -> NgsBaseVector | ReadOnlyVector:
        ...

    def zeros(self, read_only: bool = False) -> Vector | ReadOnlyVector:
        r"""Create a vector whose coefficients are zero.
        
        Parameters
        ----------
        read_only: bool [default: False]
            Whether to return a read-only view.
        
        
        Returns
        -------
        Vector | ReadOnlyVector
            Result described above.
        """
        read_only = self._validate_read_only_flag(read_only)
        vec = self._gfu_fes.vec.CreateVector()
        vec.FV().NumPy()[:] = 0
        result = NgsBaseVector(vec, make_copy=False)
        return self.read_only(result) if read_only else result
    
    @overload
    def ones(self, read_only: Literal[False]=False) -> NgsBaseVector:
        ...

    @overload
    def ones(self, read_only: Literal[True]) -> ReadOnlyVector:
        ...

    @overload
    def ones(self, read_only: bool) -> NgsBaseVector | ReadOnlyVector:
        ...

    def ones(self, read_only: bool = False) -> Vector | ReadOnlyVector:
        r"""Create a vector whose coefficients are one.
        
        Parameters
        ----------
        read_only: bool [default: False]
            Whether to return a read-only view.
        
        
        Returns
        -------
        Vector | ReadOnlyVector
            Result described above.
        """
        read_only = self._validate_read_only_flag(read_only)
        if self.codim == 1:
            self._gfu_fes.Set(1)
        else:
            for gf_i in self._gfu_fes.components:
                gf_i.Set(tuple(1 for _ in range(gf_i.dim)))
        self._gfu_fes.vec.data = ngs.Projector(self.fes.FreeDofs(), range=True).Project(self._gfu_fes.vec)
        result = NgsBaseVector(self._gfu_fes.vec, make_copy=True)
        return self.read_only(result) if read_only else result

    def read_only(self, x: NgsBaseVector) -> ReadOnlyVector:
        r"""Return a read-only wrapper sharing the underlying NGSolve vector.
        
        Parameters
        ----------
        x: NgsBaseVector
            X.
        
        
        Returns
        -------
        ReadOnlyVector
            Read-only vector view.
        """
        if x not in self:
            raise ValueError(Errors.not_in_vecsp(
                x, self, vec_name="vector",
                add_info="read_only argument is not in the NgsVectorSpace."
            ))
        if x.read_only:
            return x
        return NgsBaseVector(x.vec, make_copy=False, read_only=True)
    
    def empty(self) -> Vector:
        r"""Allocate a writable vector with uninitialized coefficients.
        
        
        Returns
        -------
        Vector
            Writable result vector.
        """
        return self.zeros()
    
    def rand(self, distribution: str = "uniform", use_space_sampling: bool = False,
             mass_matrix_weighting: bool = True, seed: int | None = None,
             **kwargs: Any) -> Vector:
        r"""Sample a random finite-element vector.

        ``use_space_sampling=True`` samples native finite-element degrees of
        freedom. When ``mass_matrix_weighting=True`` the samples are multiplied
        by the inverse *adjoint* of the mass-matrix Cholesky factor, so normal
        samples have covariance M^{-1} before any boundary projection.
        ``use_space_sampling=False`` retains sampling on the auxiliary L2
        finite-element space, followed by interpolation into ``self.fes``.

        Complex coefficients are sampled using independent real and imaginary
        draws. A supplied seed initializes the RNG only once (not once for
        each component). Poisson sampling is handled by :meth:`poisson` and
        is unavailable for complex-valued spaces.
        """
        if distribution == "poisson" and self.is_complex:
            raise TypeError("Poisson sampling requires a real NgsVectorSpace.")
        if seed is not None:
            self.set_rng_seed(seed)

        def draw(size: int) -> np.ndarray:
            values = np.asarray(self._draw_sample(
                distribution=distribution, size=int(size), **kwargs))
            if self.is_complex and not np.iscomplexobj(values):
                values = values + 1j * self._draw_sample(
                    distribution=distribution, size=int(size), **kwargs)
            return values

        if use_space_sampling:
            if mass_matrix_weighting:
                if distribution == "poisson":
                    raise ValueError("Mass-matrix weighting does not preserve Poisson counts.")
                self.create_mass_matrix_cholesky()
            coefficients = draw(self.fes.ndof)
            if mass_matrix_weighting:
                # If M = L L*, then L^{- *} z has covariance M^{-1}.
                # Solving with L (rather than L*) produces the wrong covariance.
                coefficients = spsolve(
                    self.mass_matrix_cholesky.conjugate().transpose(),
                    coefficients)
            result = self.zeros()
            self._set_native_coefficients(result, coefficients)
        else:
            if self._fes_util is None:
                raise RuntimeError(Errors.runtime_error(
                    "Auxiliary finite-element space is unavailable for random sampling."))
            self._gfu_util.vec.FV().NumPy()[:] = draw(self._fes_util.ndof)
            if self.codim == 1:
                self._gfu_fes.Set(self._gfu_util)
            elif self.product_space:
                # The old implementation wrote through the nonexistent _gfu_i.
                for target, source in zip(self._gfu_fes.components,
                                          self._gfu_util.components, strict=True):
                    target.Set(source)
            else:
                # Each vector-valued component needs an independent L2 draw.
                components = [self._gfu_util]
                for _ in range(1, self.codim):
                    component = ngs.GridFunction(self._fes_util)
                    component.vec.FV().NumPy()[:] = draw(self._fes_util.ndof)
                    components.append(component)
                self._gfu_fes.Set(tuple(components))
            result = NgsBaseVector(self._gfu_fes.vec, make_copy=True)

        # Respect constrained degrees of freedom in either sampling mode.
        result.vec.data = ngs.Projector(self.fes.FreeDofs(), range=True).Project(result.vec)
        return result

    
    def poisson(self, x: ReadOnlyVector, n: int = 1,
                tol: float = 3e-15, seed: int | None = None) -> Vector:
        r"""Draw coefficient-wise Poisson counts for a real FE vector.

        The samples have the supplied *native FE coefficients* as rates.
        This avoids interpolating rates onto an auxiliary space and back,
        which can change the expectation and fails on product spaces.
        ``n`` independent draws are averaged when ``n > 1``.

        Complex FE spaces are rejected even when ``as_real_space=True``.
        The rate vector is never modified, including when negative rounding
        errors of magnitude at most ``tol`` are clipped to zero.
        """
        if self.is_complex or (isinstance(x, NgsBaseVector) and x.is_complex):
            raise TypeError("Poisson sampling requires a real NgsVectorSpace and real rates.")
        if x not in self:
            raise ValueError(Errors.not_in_vecsp(
                x, self, add_info="Poisson rates must belong to the vector space."))
        if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 1:
            raise ValueError("n must be a positive integer.")
        if (isinstance(tol, (bool, np.bool_))
                or not isinstance(tol, (int, float, np.integer, np.floating))
                or not np.isfinite(tol) or tol < 0):
            raise ValueError("tol must be finite and nonnegative.")
        rates = np.asarray(self.flatten(x))
        if np.iscomplexobj(rates):
            raise TypeError("Poisson rates must have real storage.")
        if np.any(~np.isfinite(rates)) or np.any(rates < -tol):
            raise ValueError("Poisson rates must be finite and nonnegative.")
        rates = np.maximum(rates, 0.)
        shape = rates.size if n == 1 else (int(n), rates.size)
        samples = self._draw_sample(
            distribution="poisson", lam=rates, size=shape, seed=seed)
        values = np.asarray(samples, dtype=float)
        if n > 1:
            values = values.mean(axis=0)
        return self.fromflat(values)

    def __contains__(self,x: ReadOnlyVector) -> bool:
        if not isinstance(x,NgsBaseVector):
            return False
        elif x.size != self.fes.ndof:
            return False
        elif x.is_complex:
            return self.is_complex
        else:
            return True
        
    def vdot(self, x: ReadOnlyVector, y: ReadOnlyVector) -> Scalar:
        r"""Vdot.
        
        Parameters
        ----------
        x: ReadOnlyVector
            Input vector; its coefficients are only read.
        y: ReadOnlyVector
            Input vector; its coefficients are only read.
        
        
        Returns
        -------
        Scalar
            Computed scalar value.
        """
        # DynamicVectorExpression is not a concrete NGSolve BaseVector and
        # can select the real-valued InnerProductD overload even when the
        # underlying coefficients are complex. Keep expressions and read-only
        # wrappers unchanged; materialize only temporary native vectors.
        x_native = x.vec.Evaluate() if x.is_dynamic else x.vec
        y_native = y.vec.Evaluate() if y.is_dynamic else y.vec
        if x_native.is_complex or y_native.is_complex:
            # Avoid the real-only InnerProductD overload seen with complex
            # lazy expressions. The native scalar product conjugates x.
            # This is independent of the as_real_space coordinate setting.
            return np.vdot(x_native.FV().NumPy(), y_native.FV().NumPy())
        return ngs.InnerProduct(y_native, x_native)

    def copyto(self, x: Vector, y: ReadOnlyVector, where: Mask | None=None) -> None:
        r"""Copy coefficients into an existing writable vector.
        
        Parameters
        ----------
        x: Vector
            Writable vector used by this operation.
        y: ReadOnlyVector
            Input vector; its coefficients are only read.
        where: Mask | None [default: None]
            Mask selecting the coordinates to write; None selects all coordinates.
        """
        if where is not None:
            return MaskableVectorSpace.copyto(self, x, y, where=where)
        r"""Copy ``y`` into the existing :class:`NgsBaseVector` ``x``.

        NGSolve performs vector assignment through the writable ``data``
        property of the wrapped ``BaseVector``. Dynamic destination
        expressions are evaluated once before they are used as output storage.
        """
        if x is y:
            return
        if not isinstance(x, NgsBaseVector):
            raise TypeError(Errors.not_instance(
                x, NgsBaseVector,
                add_info="The destination of NgsVectorSpace.copyto must be an NgsBaseVector."
            ))
        if not isinstance(y, NgsBaseVector):
            raise TypeError(Errors.not_instance(
                y, NgsBaseVector,
                add_info="The source of NgsVectorSpace.copyto must be an NgsBaseVector."
            ))
        if x.size != y.size:
            raise ValueError(Errors.value_error(
                f"Cannot copy NGSolve vectors of different sizes {y.size} and {x.size}."
            ))
        x._require_writable("copyto")
        if x.is_dynamic:
            x.vec = x.vec.Evaluate()
            x.is_dynamic = False
        x.vec.data = y.vec

    def fill(self, x: Vector, value: Scalar, where: Mask | None=None) -> None:
        r"""Fill selected coefficients of a writable vector with a scalar.
        
        Parameters
        ----------
        x: Vector
            Writable vector used by this operation.
        value: Scalar
            Value.
        where: Mask | None [default: None]
            Mask selecting the coordinates to write; None selects all coordinates.
        """
        if where is not None:
            return MaskableVectorSpace.fill(self, x, value, where=where)
        r"""Fill an existing :class:`NgsBaseVector` with ``value``."""
        if x not in self:
            raise ValueError(Errors.not_in_vecsp(
                x, self, vec_name="vector", add_info="fill argument is not in the NgsVectorSpace."
            ))
        if not np.isscalar(value):
            raise TypeError(Errors.type_error(
                f"fill value must be a scalar. Got {type(value)}."
            ))
        x._require_writable("fill")
        if x.is_dynamic:
            x.vec = x.vec.Evaluate()
            x.is_dynamic = False
        x.vec.FV().NumPy().fill(value)

    def _iter_native_coefficients(self, x: ReadOnlyVector) -> Iterator[Scalar]:
        r"""Iterate over the native real or complex NGSolve coefficients.
        
        Parameters
        ----------
        x: ReadOnlyVector
            Input vector; its coefficients are only read.
        
        
        Returns
        -------
        Iterator[Scalar]
            Result described above.
        """
        if x not in self:
            raise ValueError(Errors.not_in_vecsp(
                x,
                self,
                vec_name="vector",
                add_info="iter_coefficients argument is not in the NgsVectorSpace."
            ))
        vector = x.vec.Evaluate() if x.is_dynamic else x.vec
        return iter(vector.FV().NumPy())

    def _set_native_coefficients(
        self,
        x: NgsBaseVector,
        coefficients: Iterable[Scalar],
    ) -> None:
        r"""Overwrite an NGSolve vector from native coefficients.
        
        Parameters
        ----------
        x: NgsBaseVector
            X.
        coefficients: Iterable[Scalar]
            Coefficients.
        """
        if x not in self:
            raise ValueError(Errors.not_in_vecsp(
                x,
                self,
                vec_name="vector",
                add_info="set_coefficients destination is not in the NgsVectorSpace."
            ))
        x._require_writable("setting coefficients")
        if x.is_dynamic:
            x.vec = x.vec.Evaluate()
            x.is_dynamic = False
        storage = x.vec.FV().NumPy()
        iterator = iter(coefficients)
        for index in range(int(self._native_dimension)):
            try:
                coefficient = next(iterator)
            except StopIteration as error:
                raise ValueError(Errors.value_error(
                    f"Expected {self._native_dimension} coefficients, but received only "
                    f"{index}."
                )) from error
            if not self.is_complex and np.iscomplexobj(coefficient):
                coefficient = np.asarray(coefficient).item()
                if coefficient.imag != 0:
                    raise ValueError(Errors.value_error(
                        "A real NgsVectorSpace cannot consume a non-real "
                        f"coefficient {coefficient!r}."
                    ))
                coefficient = coefficient.real
            storage[index] = coefficient
        try:
            next(iterator)
        except StopIteration:
            return
        raise ValueError(Errors.value_error(
            f"Expected {self._native_dimension} coefficients, but received more."
        ))
    
    def complex_space(self) -> VectorSpaceBase:
        r"""Compute the corresponding complex vector space by creating a complex fes from the input of the old one.
        
        Returns
        -------
        VectorSpaceBase
            The complex space corresponding to this vector space.
        """
        if self.is_complex:
            return copy(self)
        return NgsVectorSpace(type(self.fes)(self.fes.mesh,order=self.fes.globalorder,bdr=self.bdr,complex=True),bdr=self.bdr)

    def real_space(self) -> VectorSpaceBase:
        r"""Compute the corresponding real vector space by creating a real fes from the input of the old one.
        
        Returns
        -------
        VectorSpaceBase
            The real space corresponding to this vector space.
        """
        if not self.is_complex:
            return copy(self)
        return NgsVectorSpace(type(self.fes)(self.fes.mesh,order=self.fes.globalorder,dirichlet=self.bdr,complex=False),bdr=self.bdr)
    

    def flatten(self, x: ReadOnlyVector) -> np.ndarray:
        r"""Contiguous coefficient coordinates; real/imag are interleaved.
        
        Parameters
        ----------
        x: ReadOnlyVector
            Input vector; its coefficients are only read.
        
        
        Returns
        -------
        np.ndarray
            Computed array.
        """
        if x not in self:
            raise ValueError("Vector outside NgsVectorSpace.")
        vector = x.vec.Evaluate() if x.is_dynamic else x.vec
        values = vector.FV().NumPy()
        if self.is_complex and self.as_real_space:
            return np.stack((values.real, values.imag), axis=-1).reshape(-1)
        return values.copy()


    def fromflat(self, x: ReadOnlyVector) -> Vector:
        r"""Fromflat.
        
        Parameters
        ----------
        x: ReadOnlyVector
            Input vector; its coefficients are only read.
        
        
        Returns
        -------
        Vector
            Writable result vector.
        """
        return super().fromflat(x)


    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other,type(self)):
            return False
        return self.fes == other.fes and self.as_real_space == other.as_real_space
    
    
    def iter_basis(self) -> Iterator[ReadOnlyVector]:
        r"""Generator iterating over the standard basis of the vector space. For efficiency,
        the same array is returned in each step, and subsequently modified in-place. If you need
        the array longer than that, perform a copy. In case of complex a vector space after each
        each array modified in its place with a real one it returns the same vector with \(1i\)
        in its place.   
        
        
        Yields
        ------
        ReadOnlyVector
            Result described above.
        """
        elm = self.zeros()
        for idx in range(self.shape[0]):
            elm[idx] = 1
            yield elm
            if self.is_complex and self.as_real_space:
                elm[idx] = 1j
                yield elm
            elm[idx] = 0

    
    
    

    @property
    def ndim(self) -> int:
        r"""Ndim.
        
        
        Returns
        -------
        int
            Computed scalar value.
        """
        return 1

    def compatible(self, other: object) -> bool:
        r"""Compatible.
        
        Parameters
        ----------
        other: object
            Other.
        
        
        Returns
        -------
        bool
            Whether the stated condition is satisfied.
        """
        return self == other

    def is_on_boundary(self,x: ReadOnlyVector) -> bool:
        r"""Is on boundary.
        
        Parameters
        ----------
        x: ReadOnlyVector
            Input vector; its coefficients are only read.
        
        
        Returns
        -------
        bool
            Whether the stated condition is satisfied.
        """
        if self.bdr is None:
            return False
        t = x.vec.CreateVector()
        t.data = x.vec
        ngs.Projector(self.fes.FreeDofs(), range=True).Project(t)
        return np.all(t.FV().NumPy() == 0)
    
    def to_gf(self, x: Vector) -> ngs.GridFunction:
        # This Workaround to connect the GridFunction to the Vector can be replaced by 
        # creating GridFunction with existing Vector. Feature was added after ngs version 6.2.2506!
        r"""To gf.
        
        Parameters
        ----------
        x: Vector
            Writable vector used by this operation.
        
        
        Returns
        -------
        ngs.GridFunction
            Result described above.
        """
        gf = ngs.GridFunction(self.fes)
        gf.vec.data = x.vec
        x.vec = gf.vec
        return gf
    
    def from_ngs(self, ngs_elem: ngs.BaseVector | ngs.GridFunction, definedon : ngs.comp.Region|None = None, copy: bool = False) -> NgsBaseVector:
        r"""From ngs.
        
        Parameters
        ----------
        ngs_elem: ngs.BaseVector | ngs.GridFunction
            Ngs elem.
        definedon: ngs.comp.Region | None [default: None]
            Definedon.
        copy: bool [default: False]
            Copy.
        
        
        Returns
        -------
        NgsBaseVector
            Result described above.
        """
        if isinstance(ngs_elem,ngs.comp.GridFunction):
            if ngs_elem.space != self.fes:
                self._gfu_fes.Set(ngs_elem)
                return NgsBaseVector(self._gfu_fes.vec,make_copy=True)
            else:
                if copy:
                    return NgsBaseVector(ngs_elem.vec,make_copy=True)
                else:
                    return NgsBaseVector(ngs_elem.vec)
        elif isinstance(ngs_elem,(ngs.BaseVector,ngs.la.DynamicVectorExpression)):
            return NgsBaseVector(ngs_elem,make_copy=copy)
        else:
            self._gfu_fes.Set(ngs_elem,definedon=definedon)
            return NgsBaseVector(self._gfu_fes.vec,make_copy = True)
            
    def create_mass_matrix_cholesky(self) -> None:
        # Builds the cholesky decomposition
        # mass = L L* (conjugate transpose for complex coefficients)
        # The output factor self.mass_matrix_cholesky corresponds to L (not L^T)
        # Warning: Current method is a hack based on scipy spLU. However,
        # correct implementation would require access to scikit.sparse.cholmod.
        r"""Create mass matrix cholesky.
        """
        if self.mass_matrix_cholesky is not None:
            return
        mass = ngs.BilinearForm(self.fes)
        u, v = self.fes.TnT()
        mass += u * v * ngs.dx
        mass.Assemble()
        mass = mass.mat
        rows,cols,vals = mass.COO()
        mass = csc_matrix((vals, (rows, cols)), shape=(mass.height, mass.width))
        
        """
        Computes sparse lower triangular B such that A ≈ B @ B.T
        using standard scipy.sparse.linalg.splu with pivoting disabled.
        """
        # 1. Force LU decomposition without pivoting
        # 'NATURAL' = No column permutations
        # diag_pivot_thresh=0 = No row permutations
        try:
            lu = splu(csc_matrix(mass), permc_spec='NATURAL', diag_pivot_thresh=0, 
                      options={'SymmetricMode': True})
        except RuntimeError:
            raise ValueError("Matrix is singular or not SPD enough for non-pivoted LU.")

        # 2. Check if permutations were actually avoided (crucial safety check)
        n = mass.shape[0]
        if not (np.array_equal(lu.perm_r, np.arange(n)) and 
                np.array_equal(lu.perm_c, np.arange(n))):
            raise ValueError("SPLU forced permutations. Matrix is likely not well-conditioned enough.")

        # 3. Construct B = L * sqrt(D)
        # splu returns L with unit diagonal. The actual pivots are on U's diagonal.
        d = lu.U.diagonal()
        
        d = np.real_if_close(d, tol=1000)
        if (np.iscomplexobj(d) or np.any(~np.isfinite(d)) or np.any(d <= 0)):
            raise ValueError("Matrix is not positive definite (negative or zero pivot found).")
            
        # Scale columns of L by sqrt(d)
        # Efficient sparse multiplication: B = L @ diag(sqrt(d))
        scale_matrix = diags(np.sqrt(d))
        self.mass_matrix_cholesky = lu.L @ scale_matrix

class NgsVectorSpaceWithInnerProduct(NgsVectorSpace):
    r"""A vector space wrapping an `ngsolve.FESpace`. That defines the inner Product 
    to include the mass matrix allowing to have easier adjoint operators.
    
    Parameters
    ----------
    fes: ngs.FESpace
        The wrapped NGSolve vector space.
    bdr: str | None [default: None]
        Boundary of the NGSolve vector space.
    **kwargs: Any
        Additional keyword arguments forwarded to the underlying implementation.
    """

    def __init__(self, fes: ngs.FESpace, bdr: str | None=None,
                 as_real_space: bool=True, **kwargs: Any) -> None:
        if not isinstance(fes, (ngs.L2,ngs.H1,ngs.VectorH1)):
            raise ValueError("The given FES is neither an H1 nor VectorH1 space of ngsolve")

        # Preserve the scalar-field choice of the underlying NgsVectorSpace.
        # Previously this keyword was silently swallowed by **kwargs.
        super().__init__(fes, bdr=bdr, as_real_space=as_real_space)
        if "definedon" in kwargs:
            self._definedon = kwargs["definedon"]
        else:
            self._definedon = None

        if "bonus_intorder" in kwargs:
            self._bonus_intorder = kwargs["bonus_intorder"]
        else:
            self._bonus_intorder = 0

        self.mass = self.compute_mass()
        self.mass_vec = self.empty().vec.CreateVector()
        self._no_pickle = {*self._no_pickle,"mass"}

    def vdot(self, x: ReadOnlyVector, y: ReadOnlyVector) -> Scalar:
        r"""Vdot.
        
        Parameters
        ----------
        x: ReadOnlyVector
            Input vector; its coefficients are only read.
        y: ReadOnlyVector
            Input vector; its coefficients are only read.
        
        
        Returns
        -------
        Scalar
            Computed scalar value.
        """
        x_native = x.vec.Evaluate() if x.is_dynamic else x.vec
        y_native = y.vec.Evaluate() if y.is_dynamic else y.vec
        # Cache the output storage, but never modify either input. This also
        # avoids passing a lazy input to the mass-matrix multiplication.
        self.mass_vec.data = self.mass * x_native
        if self.mass_vec.is_complex or y_native.is_complex:
            return np.vdot(self.mass_vec.FV().NumPy(),
                           y_native.FV().NumPy())
        return ngs.InnerProduct(y_native, self.mass_vec)
    
    @property
    def bonus_intorder(self) -> int:
        r"""Bonus intorder.
        
        
        Returns
        -------
        int
            Computed scalar value.
        """
        return self._bonus_intorder
    
    @bonus_intorder.setter
    def bonus_intorder(self, value: Scalar | ArrayLike) -> None:
        r"""Bonus intorder.
        
        Parameters
        ----------
        value: Scalar | ArrayLike
            Value.
        """
        self._bonus_intorder = value
        self.mass = self.compute_mass()

    @bonus_intorder.deleter
    def bonus_intorder(self) -> None:
        r"""Bonus intorder.
        """
        self._bonus_intorder = 0
        self.mass = self.compute_mass()

    @property
    def definedon(self) -> ngs.Region | None:
        r"""Definedon.
        
        
        Returns
        -------
        ngs.Region | None
            Result described above.
        """
        return self._definedon
    
    @definedon.setter
    def definedon(self, value: Scalar | ArrayLike) -> None:
        r"""Definedon.
        
        Parameters
        ----------
        value: Scalar | ArrayLike
            Value.
        """
        self._definedon = value
        self.mass = self.compute_mass()
    
    def compute_mass(self) -> ngs.BaseMatrix:
        r"""Compute mass.
        
        
        Returns
        -------
        ngs.BaseMatrix
            Result described above.
        """
        u, v = self.fes.TnT()
        mass_bf = ngs.BilinearForm(self.fes)
        mass_bf += u*v*ngs.dx(definedon = self.definedon, bonus_intorder = self.bonus_intorder)
        mass_bf.Assemble()
        return mass_bf.mat
