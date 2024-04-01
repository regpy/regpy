import numpy as np
from scipy.sparse import csc_matrix 
from scipy.sparse.linalg import spsolve
from scipy.interpolate import RegularGridInterpolator


def harmonic_extension(mask, values, damping = 0,h_val = None,ext_bd_cond='Neum'):
    r""" harmonic extension of a function on part of a 2D regular grid

    Parameters
    ----------
    mask: numpy.ndarray
        binary mask with values 1 at points where values are given (Dirichlet boundary values)
        0 at points where values must be computed
    values: numpy.ndarray
        array of function values; only values at points where mask=True are relevant
    damping: float,optional
        The extension satisfies \((-\Delta + \text{damping})u =0\), so only for the default 
        \(damping = 0\) the extension is harmonic.
    h_val: numpy.ndarray, optional
        TODO Description
    ext_bd_cond: str, optional
        By default, Neumann conditions are imposed at exterior boundaries. If ext_bd_cond is different from 'Neum', 
        Dirichlet conditions will be used.  
    Returns 
    ----------
    numpy.ndarray
        An array u which coincides with values at {mask=True}, and an approximation of a 
        harmonic function on {mask=False} which is globally continuous
    """
    ndim = mask.ndim
    need_padding = False
    for dir in range(ndim):
        sl = [slice(None)]*ndim
        sl[dir] = 0
        if np.any(mask[tuple(sl)]==0):
            need_padding = True
        sl = [slice(None)]*ndim
        sl[dir] = -1
        if np.any(mask[tuple(sl)]==0):
            need_padding = True   
    if need_padding:
        mask = np.pad(mask.astype(int),1,'constant',constant_values= -1 if ext_bd_cond=='Neum' else 1)
        values = np.pad(values,1,'edge')

    if h_val is None:
        h_val = np.ones((ndim,))/np.max(mask.shape)
    val = values.ravel()
    G = np.where(mask,0,1) # boolean to integer
    k_int = np.nonzero(G)  # integer coordinates of interior points
    k_ext = np.nonzero(1-G) # integer coordinates of exterior points
    G[k_int] = 1+np.arange(len(k_int[0]))

    G1 = G.ravel()
    #[m,n] = G.shape
    # Indices of interior points
    p = np.where(G1>0)[0] # list of numbers of interior points in flattened array
    N = len(p)
    f = np.zeros((N,),dtype=values.dtype) # right hand side of matrix equation 

    # Connect interior points to themselves with 4's.
    i = G1[p]-1
    j = G1[p]-1
    
    i = []
    j = []
    s = []
  
    dia = damping*np.ones((len(p),))

    # compute distances to neighbors in different directions 
    kval= [1]
    for d in range(ndim-1,0,-1):
        kval = np.concatenate([kval,[kval[-1]*G.shape[d]] ])
    # If G.shape = [m,n], then kval= [1,n]. 
    # If G.shape = [l,m,n], then kval = [1,n,m*n]

    for dir,h in enumerate(h_val):
        for k in  kval[dir]*np.array([-1,1]):
            # Possible neighbors in k-th direction
            Q = G1[p+k]
            # Index of points with interior neighbors in k-th direction
            q = np.where(Q>0)[0]
            # Connect interior points to neighbors with -1's.
            entries = np.ones(q.shape)/h**2
            i = np.concatenate([i, G1[p[q]]-1])
            j = np.concatenate([j,Q[q]-1])
            s = np.concatenate([s,-entries])
            dia[G1[p[q]]-1] += entries 
            q_ext = np.where(Q==0)[0]
            i_ext = G1[p[q_ext]]-1
            f[i_ext] = f[i_ext]+val[p[q_ext]+k]/h**2
            # Indices of points with neighbors on Dirichlet boundary
            entries = np.ones(q_ext.shape)/h**2
            dia[G1[p[q_ext]]-1] += entries
    i = np.concatenate([i, G1[p]-1])
    j = np.concatenate([j, G1[p]-1])
    s = np.concatenate([s,dia])   
    negLap= csc_matrix((s, (i,j)),(N,N))
    u = values.copy()
    u[k_int]=spsolve(negLap,f)
    if need_padding:
        sl = (slice(1,-1,None),)*ndim
        return u[sl]
    else:
        return u


def extension_along_lines(log_g_map,mask):
    r""" Extrapolate log(g_map) from nanotip along straight lines. Damping is modeled by linear decay on these lines.

    Parameters
    ----------
    log_g_map: numpy.ndarray
        TODO Description
    mask: numpy.ndarray
        TODO Description
        
    Returns 
    ----------
    numpy.ndarray
        TODO Description
    """   
    m,n = log_g_map.shape
    N_center=707  # origin is place at pixel (N_center,(n-1)/2)
    # artificial coordinate system for g_map; cut through nanotip is line eta = 1
    xi =  (N_center-np.arange(m))/(N_center-1)
    eta = np.linspace(-1,1,n)
    a = 0.1  # the origin is connected to the points (-a,1) and (-a,-1) by two of the straight lines
             # Furthermore (1,0) is connected to (1,1) and (1,-1)
    # Extrapolation into two trapezoids above and below eta=0 axis is based on values of log(g_map) on the lines 
    # eta = \pm d \pm alpha \xi
    d = 0.02
    alpha = 0.27
    Xi,Eta = np.meshgrid(xi,eta,indexing='ij', sparse=True)  
    log_g = RegularGridInterpolator((xi,eta),log_g_map)
    Z = log_g_map

    t = (1+a*Eta) *(alpha*Xi-Eta +d) / (1+a*Eta - alpha*a*Xi + alpha*a)
    Xpro = Xi + (a*t)*(Xi-1)/(1+a*Eta)
    Ypro = Eta + t
    ind = np.logical_and(np.logical_and(Eta>0,  mask>0.5), (np.abs(Eta)>= (-1/a)*Xi))
    Z[ind] = log_g(np.array([np.clip(Xpro[ind],xi[-1],xi[0]),Ypro[ind]]).T) + 7*(t[ind]+0.008)

    t = (1+a*Eta)*(-alpha*Xi-Eta -d) / (-1-a*Eta + alpha*a*Xi - alpha*a)
    Xpro = Xi + (a*t)*(Xi-1)/(1-a*Eta)
    Ypro = Eta - t
    ind = np.logical_and(np.logical_and(Eta<0, mask>0.5), (np.abs(Eta)>= (-1/a)*Xi))
    Z[ind] = log_g(np.array([np.clip(Xpro[ind],xi[-1],xi[0]),Ypro[ind]]).T)+7*(t[ind]+0.007)

    # around the apex a harmonic extension is used
    mask_apex = np.logical_and(mask>0.5, (np.abs(Eta)<=(-1/(a-0.03))*Xi))
    mask_apex[0,:] = False; mask_apex[-1,:] = False; mask_apex[:,-1] = False
    Z = harmonic_extension(~mask_apex,Z)
    return Z
