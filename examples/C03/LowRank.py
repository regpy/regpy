from regpy.functionals import Functional
from regpy.vecsps import DirectSum
import numpy as np

class HilbertSchmidtLowRank(Functional):
    """ Functionals defined for pairs (U,V) of matrices of the form
    \[
    S(U,V) = \frac{1}{2}\|\Re(U*V^H) - \sum_j G_j G_j^H\|_Fro^2.
    \]
    Here Fro denotes the Frobenius norm, and $Re(A):= \frac{1}{2}(A+A^*)$. 
    U and V may be tensor of arbitrary identical dimensions.
    U and 
        if arguments_identical:
            toret = np.tensordot(U, np.tensordot(U,U,axes=dims), axes=1)
            for dat in self.data:
                toret -= np.tensordot(dat, np.tensordot(dat,U,axes=dims), axes=1)
            return self.domain.join(toret,toret)            
        else:
            toret1 = np.tensordot(U, np.tensordot(V,V,axes=dims), axes=1)
            toret1 += np.tensordot(V, np.tensordot(U,V,axes=dims), axes=1)
            toret1 *= 0.5

            toret2 = np.tensordot(U, np.tensordot(V,U,axes=dims), axes=1)
            toret2 += np.tensordot(V, np.tensordot(U,U,axes=dims), axes=1)
            toret2 *= 0.5

            for dat in self.data:
                toret1 -= np.tensordot(dat, np.tensordot(dat,V,axes=dims), axes=1)
                toret2 -= np.tensordot(dat, np.tensordot(dat,U,axes=dims), axes=1)
            return self.domain.join(toret1,toret2)V are flattened in all but the last dimension to obtain matrices. 
    Also the dimensions of G_j must be identical to that of U and V, except for the last one. 
    
    The matrix of which the Frobenius norm is taken is not set up for the sake of memory efficiency.

    -----------------------
    Parameters:
    domain: `vecspc.DirectSum)`
       Domain of the functional. Must have two summands
    data: list of np.arrays [default: []]
       The data (G_j)_{j=1...J}.

    """
    def __init__(self, domain,data=[]):
        assert isinstance(domain,DirectSum)
        assert len(domain.summands)==2
        self.dim = len(domain.summands[0].shape)
        assert self.dim>=2
        self.sdomain = domain.summands[0]        
        assert domain.summands[1].shape == self.sdomain.shape
        self.cshape = self.sdomain.shape[:-1]
        super().__init__(domain)
        print(self.cshape)
        print(data[0].shape)
        for dat in data:
            assert dat.shape==self.cshape or dat.shape[:-1]==self.cshape
        self.data = data

    def eval(self,X,arguments_identical=False):
        U,V = self.domain.split(X)
        U_2d = U.reshape(np.prod(self.cshape),U.shape[-1])
        V_2d = V.reshape(np.prod(self.cshape),V.shape[-1])
        data = []
        for dat in self.data:
            data.append(dat.reshape(np.prod(self.cshape),-1))        
        res = 0.
        if arguments_identical:
            for tup1 in zip(U_2d,*data):
                for tup2 in zip(V_2d,*data):
                    #bigmatrix_el= np.vdot(tup1[0],tup2[0]).real
                    bigmatrix_el= np.vdot(tup1[0],tup2[0])
                    for g1,g2 in zip(tup1[1:],tup2[1:]):
                        #bigmatrix_el -= np.vdot(g1,g2).real
                        bigmatrix_el -= np.vdot(g1,g2)
                    #res += bigmatrix_el**2
                    res += abs(bigmatrix_el)**2
        else:
            for tup1 in zip(U_2d,V_2d,*data):
                for tup2 in zip(U_2d,V_2d,*data):
                    #bigmatrix_el= 0.5*np.real(np.vdot(tup1[0],tup2[1])
                    #                 + np.vdot(tup2[0],tup1[1]))
                    bigmatrix_el= 0.5*(np.vdot(tup1[0],tup2[1])
                                     + np.vdot(tup1[1],tup2[0]))
                    for g1,g2 in zip(tup1[2:],tup2[2:]):
                        #bigmatrix_el -= np.vdot(g1,g2).real
                        bigmatrix_el -= np.vdot(g1,g2)
                    #res += bigmatrix_el**2
                    res += abs(bigmatrix_el)**2

        return res/2
    
    def subgradient(self, X, arguments_identical=False):
        U,V = self.domain.split(X)
        dims = (np.arange(self.dim-1),np.arange(self.dim-1))
        if arguments_identical:
            toret = np.tensordot(U, np.tensordot(U.conj(),U,axes=dims), axes=1)
            for dat in self.data:
                toret -=  np.tensordot(dat, np.tensordot(dat.conj(),U,axes=dims),
                                       axes=0 if dat.shape==self.cshape else 1) 
            return self.domain.join(toret,toret)
        else:
            toret1 = np.tensordot(U, np.tensordot(V.conj(),V,axes=dims), axes=1)
            toret1+= np.tensordot(V, np.tensordot(U.conj(),V,axes=dims), axes=1)
            toret1 *= 0.5

            toret2 = np.tensordot(U, np.tensordot(V.conj(),U,axes=dims), axes=1)
            toret2+= np.tensordot(V, np.tensordot(U.conj(),U,axes=dims), axes=1)
            toret2 *= 0.5            

            for dat in self.data:
                toret1 -=  np.tensordot(dat, np.tensordot(dat.conj(),V,axes=dims), 
                                        axes=0 if dat.shape==self.cshape else 1) 
                toret2 -=  np.tensordot(dat, np.tensordot(dat.conj(),U,axes=dims), 
                                        axes=0 if dat.shape==self.cshape else 1)               
            return self.domain.join(toret1,toret2)
        
    def getLipschitz(self,X):
        U,V = self.domain.split(X)
        U=U.reshape(np.prod(U.shape[:-1]), U.shape[-1])
        V=V.reshape(np.prod(V.shape[:-1]), V.shape[-1])
        self.Lipschitz= np.linalg.norm(U,2)**2 + np.linalg.norm(V,2)**2
        return self.Lipschitz

