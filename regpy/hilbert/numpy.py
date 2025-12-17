import numpy as np
from scipy.sparse import csc_matrix

from regpy import vecsps
from regpy.util import memoized_property,Errors
from regpy.operators import PtwMultiplication,Pow,MatrixMultiplication,FourierTransform,CoordinateProjection

from .base import HilbertSpace

__all__ = ["L2MeasureSpaceFcts","L2UniformGridFcts","SobolevUniformGridFcts","HmDomain"]

class L2MeasureSpaceFcts(HilbertSpace):
    r"""`L2` implementation on a `regpy.vecsps.MeasureSpaceFcts`.
    
    Parameters
    ----------
    vecsp : regpy.vecsps.MeasureSpaceFcts
        Underlying discretization
    weights : array-like
        Weight in the norm.
    """

    def __init__(self, vecsp, weights=None):
        if not isinstance(vecsp,vecsps.MeasureSpaceFcts):
            raise TypeError(Errors.not_instance(vecsp,vecsps.MeasureSpaceFcts,"To define an L2MeasureSpaceFcts the vector space needs to be at least a MeasureSpaceFcts or an derivative of it."))
        super().__init__(vecsp)
        self.weights = weights

    @memoized_property
    def gram(self):
        if self.weights is None:
            if np.all(self.vecsp.measure==1):
                return self.vecsp.identity
            else:
                return PtwMultiplication(self.vecsp,self.vecsp.measure)
        else:
            return PtwMultiplication(self.vecsp, self.weights*self.vecsp.measure)


class L2UniformGridFcts(HilbertSpace):
    r"""`L2` implementation on a `regpy.vecsps.UniformGridFcts`, taking into account the volume
    element.
    """

    def __init__(self, vecsp, weights=None):
        if not isinstance(vecsp,vecsps.UniformGridFcts):
            raise TypeError(Errors.not_instance(vecsp,vecsps.UniformGridFcts,"To define an L2UniformGridFcts the vector space needs to be at a UniformGridFcts or an derivative of it."))
        super().__init__(vecsp)
        self.weights = weights

    @memoized_property
    def gram(self):
        if self.weights is None:
            return self.vecsp.volume_elem * self.vecsp.identity
        else:
            return self.vecsp.volume_elem * PtwMultiplication(self.vecsp, self.weights)


class SobolevUniformGridFcts(HilbertSpace):
    r"""`Sobolev` implementation on a `regpy.vecsps.UniformGridFcts`.

    Parameters
    ----------
    vecsp : regpy.vecsps.UniformGridFcts
        Grid on which to define the Sobolev space.
    index : float, optional
        Sobolev index, Defaults: 1
    axes : list, optional
        List of axes for which to compute in default all axes, Defaults: None
    """
    def __init__(self, vecsp, index=1, axes=None):
        if not isinstance(vecsp,vecsps.UniformGridFcts):
            raise TypeError(Errors.not_instance(vecsp,vecsps.UniformGridFcts,"To define an SobolevUniformGridFcts the vector space needs to be at a UniformGridFcts or an derivative of it."))
        super().__init__(vecsp)
        self.index = index
        if axes is None:
            axes = range(vecsp.ndim)
        self.axes = list(axes)

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return (
                self.vecsp == other.vecsp and
                self.index == other.index
            )
        else:
            return NotImplemented

    @memoized_property
    def gram(self):
        ft = FourierTransform(self.vecsp, axes=self.axes)
        mul = PtwMultiplication(
            ft.codomain,
            self.vecsp.volume_elem * (
                1 +  ft.codomain.coord_distances(axes=self.axes)**2
            )**self.index
        )
        return ft.adjoint * mul * ft

class HmDomain(HilbertSpace):
    r"""Implementation of a Sobolev space :math:`H^m(D)` for a subset :math:`D` of a `regpy.vecsps.UniformGridFcts` grid.
    :math:`D` is characterized by a binary or integer-valued mask: `D={mask==1}`.
    `{mask==0}` are Dirichlet boundaries, and `{mask==-1}` Neumann boundaries.
    `mask` may also be boolean, in this case there are only Dirichlet boundaries.
    Boundary condition at the exterior boundaries are specified by `ext_bd_cond`, default is Neumann ('Neum')

    `m=index` is a non-negative integer, the order or index of the Sobolev space.
    The gram matrix is given by :math:`(\alpha I - \Delta)^{-m}`.

    By default it is assumed that the lengths in grid are given in physical dimensions,
    and a non-dimensionalization is carried out such that the largest side length (extent) of grid is 1.

    If `weight` is specified, the Gram matrix will approximate :math:`(\alpha I-{weight}\Delta)^{-m}`. `weight` should be slowly varying.

    Parameters
    ----------
    vecsp : regpy.vecsps.UniformGridFcts
        Underlying grid functions.
    mask : array-type
        Mask to capture that subset :math:`D` on which the Sobolev space is defined. Can only contain 
        values `{-1,0,1}` or is a boolean. Shape has to match the shape of `vecsp`.
    h : tuple or None or string, optional
        The extent of the domain either given as a tuple or computed. Option key strings "physical" or 
        "normalized". (Defaults: "normalized)
    index : int, optional
        The Sobolev index :math:`m`. (Defaults: 1)
    weight : array-type, optional
        Weights to be applied to Laplacian in the gram matrix definition. (Defaults: None)
    ext_bd_cond : any, optional
        Exterior boundary conditions to be applied. If not "Neum" takes Dirichlet boundary conditions. (Defaults: "Neum")
    alpha : scalar, optional
        Parameter when computing the gram matrix as :math:`(\alpha I - \Delta)^{-m}`. (Defaults: 1)
    dtype : type, optional
        Type of underlying grid. (Defaults: float) 
    """

    def __init__(self,
                vecsp, 
                mask = None,
                h='normalized',
                index=1,
                weight=None,
                ext_bd_cond = 'Neum',
                alpha = 1,
                dtype = float):
        if not isinstance(vecsp,vecsps.UniformGridFcts):
            raise TypeError(Errors.not_instance(vecsp,vecsps.UniformGridFcts,"The underlying vecsp has to be of type UniformGridFcts"))
        if mask is None:
            mask = vecsp.ones() == 1
        elif vecsp.shape != mask.shape:
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
        elif vecsp is None:
            self.h_val=1./np.max(mask.shape)*np.ones((self.ndim,))
        elif h=='physical':
            self.h_val = vecsp.extents/(np.array(vecsp.shape)-1)
        elif h=='normalized':
            self.h_val = (2.*np.pi/np.max(vecsp.extents))* (vecsp.extents/(np.array(vecsp.shape)-1))
        else:
            raise NotImplemented

        self.index = index
        """Sobolev index.
        """
        self.alpha = alpha
        """Regularizer for Gram matrix.
        """
        self.mask = (mask==1)
        """Mask to determine the subspace D.
        """
        self.dtype = vecsp.dtype if vecsp else dtype
        """Type of the underlying grid. 
        """

        self.proj = CoordinateProjection(
                self.vecsp,
                self.mask
            )

        mask_padded = np.pad(mask.astype(int),1,'constant',constant_values= -1 if ext_bd_cond=='Neum' else 0)
        
        self.G = np.zeros(mask_padded.shape,dtype=int)
        interior_ind = mask_padded==1
        self.G[interior_ind] = 1+np.arange(np.count_nonzero(interior_ind))
        self.G[mask_padded==-1] = -1

        if weight is None:
            self.weight = None
        elif weight.shape == mask.shape:
            self.weight = np.pad(weight,1,'edge')
        else:
            raise ValueError(Errors.value_error("weight has to have the same shape as the vector space or be None",self))

    def I_minus_Delta(self):
        r"""
        I_minus_Delta is the sparse form of the sum of the `alpha*identity` and the negative Laplacian on the domain D 
        defined by masking with `mask`.
        """
        if not self.weight is None:
            w = self.weight.ravel()
        # Indices of interior points
        G1 = self.G.ravel()
        p = np.where(G1>0)[0] # list of numbers of interior points in flattened array
        N = len(p)
        # Connect interior points to themselves with 4's.
        i = []   # row indices of matrix entries
        j = []   # column indices of matrix entries
        s = []   # values of matrix entries
        dia = self.alpha * np.ones((len(p),))   # values of diagonal matrix entries; ones correspond to identity matrix
        # for k = north, east, south, west

        kval= [1]
        for d in range(self.ndim-1,0,-1):
            kval = np.concatenate([kval,[kval[-1]*self.G.shape[d]] ])
        # If G.shape = [m,n], then kval= [1,n].
        # If G.shape = [l,m,n], then kval = [1,n,m*n]

        for dir,h in enumerate(self.h_val):
            for k in  kval[dir]*np.array([-1,1]):
                # Possible neighbors in k-th direction
                Q=np.zeros_like(p)
                Q = G1[p+k]
                # Indices of points with interior neighbors
                q = np.where(Q>0)[0]
                # Connect interior points to neighbors
                i = np.concatenate([i, G1[p[q]]-1])
                j = np.concatenate([j,Q[q]-1])
                entries = np.ones(q.shape)/h**2
                if not self.weight is None:
                    entries = entries * np.sqrt(w[p[q]]*w[p[q]+k])
                s = np.concatenate([s,-entries ])
                dia[G1[p[q]]-1] += entries
                # Indices of points with neighbors on Dirichlet boundary
                q_diri = np.where(Q==0)[0]
                entries = np.ones(q_diri.shape)/h**2
                if not self.weight is None:
                    entries = entries * np.sqrt(w[p[q_diri]]*w[p[q_diri]+k])
                dia[G1[p[q_diri]]-1] += entries
        i = np.concatenate([i, G1[p]-1])
        j = np.concatenate([j, G1[p]-1])
        s = np.concatenate([s,dia])
        return csc_matrix((s, (i,j)),(N,N))

    @memoized_property
    def gram(self):
        mat = Pow(
            MatrixMultiplication(self.I_minus_Delta(),inverse='superLU',domain=self.proj.codomain,codomain=self.proj.codomain,dtype = self.dtype),
            self.index
            )
        gram = self.proj.adjoint * mat * self.proj
        gram.inverse = self.proj.adjoint * mat.inverse * self.proj
        return gram