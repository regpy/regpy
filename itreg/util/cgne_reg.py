
import numpy as np


__all__ = ['CGNE_reg']




def CGNE_reg(op, y, xref, regpar, N_CG = 1000, epsCG = 1e-2):
    ''' function CGNE_reg, which solves 

     A h = b by CGNE with
        A := G_X^{-1} F'* G_Y F' + regpar I
        b := G_X^{-1}F'^* G_Y y + regpar xref
     A is self-adjoint with respect to the inner product <u,v> = u'G_X v
    
     G_X, G_Y -> op.domx.gram, op.domy.gram
     G_X^{-1} -> op.domx.gram_inv
     F'*      -> F.adjoint
    '''   
   
    #compute rtilde=G_X b
    auxy = op.domy.gram(y)
    rtilde = op.adjoint(auxy)
    rtilde += regpar * op.domx.gram(xref)
    
    r = op.domx.gram_inv(rtilde)
   
    d = np.copy(r)
    
    norm_r = np.real(np.dot(rtilde , r))
    norm_r0 = np.copy(norm_r)
    #possibly not needed:
    #norm_h = 0
    
    h  = np.zeros(len(r))
    CGstep = 1
    print(norm_r.shape)
    while( np.sqrt(norm_r/norm_r0) > epsCG and CGstep <= N_CG):
        
        auxY = op.domy.gram(op.derivative()(d))
        Adtilde = op.adjoint(auxY)
        Adtilde += regpar* op.domx.gram(d)

        Ada = np.real(Adtilde.T * d)
        alpha = norm_r / Ada
        h += alpha * d
        rtilde -= alpha * Adtilde
        r = op.domx.gram_inv(rtilde)
        norm_r_old = np.copy(norm_r)
        norm_r = np.real(np.dot(rtilde.T , r))
        
        beta = norm_r / norm_r_old
        d = r + beta *d
        #possibly not needed:
        norm_h = np.dot(h.T ,op.domx.gram(h))
        CGstep += 1
        print(norm_h)
    return h


        
                        
        
    
htest = CGNE_reg(op, data, xs, regpar =0.9)