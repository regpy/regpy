"""Collection of CG methods."""

import numpy as np
import numpy.linalg as LA


__all__ = ['CGNE_reg',
           'CG'
           ]

           
def CGNE_reg(op, y, xref, regpar, cgmaxit=1000, cg_eps=1e-2): 
    """CGNE method.        
    
    Used by the inner solver ``sqp.py``.
    
    The CG method solves by CGNE
    
       .. math:: A h = b,
        
    with
    
    .. math:: A := G_X^{-1} F^{' *} G_Y F' + \mbox{regpar} ~I
    .. math:: b := G_X^{-1} F^{' *} G_Y y + \mbox{regpar}~ \mbox{xref}
    
    where
    
    +--------------------+-------------------------------------+ 
    | :math:`F`          | self.op                             | 
    +--------------------+-------------------------------------+ 
    | :math:`G_X,~ G_Y`  | self.op.domain.gram, self.op.range.gram | 
    +--------------------+-------------------------------------+
    | :math:`G_X^{-1}`   | self.op.domain.gram_inv               |
    +--------------------+-------------------------------------+                  
    | :math:`F'`         | self.op.derivative()                |
    +--------------------+-------------------------------------+ 
    | :math:`F'*`        | self.op.derivative().adjoint        | 
    +--------------------+-------------------------------------+

        
    Parameters
    ----------
    op : :class:`Operator <itreg.operators.Operator>`
        The forward operator.
    y : array
        The right hand side.
    xref : array
        The difference between the starting point and the current point (of the
        solver using the CGNE_reg method).
    regpar : float
        A factor considered for stopping the inner iteration (where the CG
        method is run).
    cgmaxit : int, optional
        Maximum iterations for the inner iteration (where the CG method is run).
    cg_eps : float, optional
        Tolerance used by the while loop.
    """ 
    
    auxy = op.range.gram(y)
    rtilde = op.adjoint(auxy)
    rtilde += regpar * op.domain.gram(xref)
    r = op.domain.gram_inv(rtilde)
    d = np.copy(r)  
    norm_r = np.real(np.dot(rtilde , r))
    norm_r0 = np.copy(norm_r)
    h  = np.zeros(r.shape) + 0j
    cg_step = 1
    
    while np.sqrt(norm_r/norm_r0) > cg_eps and cg_step <= cgmaxit:
        auxY = op.range.gram(op(d))
        adtilde = op.adjoint(auxY) + regpar * op.domain.gram(d)
        
        ada = np.real(np.dot(adtilde, d))
        alpha = norm_r / ada
        h += alpha * d
        rtilde -= alpha * adtilde
        r = op.domain.gram_inv(rtilde)
        norm_r_old = np.copy(norm_r)
        norm_r = np.real(np.dot(rtilde, r))
        beta = norm_r / norm_r_old
        d = r + beta *d
        
        cg_step += 1
    return h
    
  
def CG(fun, b, init, eps, maxit):
    """CG method.
    
    Used by the solver ``irnm_kl_newton.py``.
    
    Parameters
    ----------
    fun : function
        Function used by CG, it should have one input argument of type array.
    b : array
        Right hand side.
    init : array
        Initial guess.
    eps: float
        Tolerance.
    maxit : int
        Maximum number of iterations.
    """
    
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