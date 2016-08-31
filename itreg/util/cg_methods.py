import numpy as np
import numpy.linalg as LA


__all__ = ['CGNE_reg',
           'CG'
           ]

           
def CGNE_reg(op, y, xref, regpar, cgmaxit = 1000, cg_eps = 1e-2): 
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
    h  = np.zeros(r.shape) + 0j
    cg_step = 1
    
    while np.sqrt(norm_r/norm_r0) > cg_eps and cg_step <= cgmaxit:
        auxY = op.domy.gram(op(d))
        adtilde = op.adjoint(auxY) + regpar * op.domx.gram(d)
        
        ada = np.real(np.dot(adtilde, d))
        alpha = norm_r / ada
        h += alpha * d
        rtilde -= alpha * adtilde
        r = op.domx.gram_inv(rtilde)
        norm_r_old = np.copy(norm_r)
        norm_r = np.real(np.dot(rtilde, r))
        beta = norm_r / norm_r_old
        d = r + beta *d
        
        cg_step += 1
    return h
    
  
def CG(fun, b, init, eps, maxit):
    
    
    n = 0
    y = fun(init)
    r = b - y
    d = np.copy(r)
    x = init
    q = LA.norm(r)
    q0 = np.copy(q)
    while q/q0>eps and n < maxit:
        n += 1
        y = fun(d)
        alpha = q**2/np.dot(d,y)
        x = x + alpha * d
        r = r - alpha * y
        q1 = LA.norm(r)
        beta = q1**2/q**2
        d = r + beta*d
        q = q1
    return x