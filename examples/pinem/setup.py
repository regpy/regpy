import numpy as np
from scipy.io import loadmat
from operators import get_op_g_to_data
from regpy.vecsps import UniformGridFcts
import os



def load_simulated_g(filename):
    r""" Loads simulated data from file

    Parameters
    ----------
    filename : string
        path of .mat file containing simulated g

    Returns 
        ----------
        g_map : numpy.ndarray
            Simulated complex g map
        mask : numpy.ndarray
            Mask of measurements with dtype float
        mask_binary : numpy.ndarray
            Mask of measurements with dtype bool
        px_size : float
            Pixel side length in meters
    """   
    mat = loadmat(filename)
    g_map = mat['pm']['g_map'][0][0]
    mask = mat['pm']['mask'][0][0]
    mask_binary = mat['pm']['mask_binary'][0][0].astype(dtype=bool)
    px_size = mat['pm']['px_sizes'][0][0]
    px_size = px_size * 1e-9 #convert nm to m
    return g_map, mask, mask_binary, px_size

def setup_simulated_g(parallel=True,list_of_filters=None,N=30):
    r"""Loads simulated_g and sets up operator, domains and masks accordingly.

    Parameters
    ----------
    parallel : bool, optional
        Passed as parameter to operators.get_op_g_to_data.
        Defaults to True.
    list_of_filters : bool, optional
        Passed as parameter to operators.get_op_g_to_data.
        Defaults to None.
    N : bool, optional
        Passed as parameter to operators.get_op_g_to_data.
        Defaults to 30.

    Returns 
        ----------
        op : numpy.ndarray
            PINEM measurement operator
        grid : regpy.vecsps.UniformGridFcts
            Domain of measurement operator
        exact_solution : numpy.ndarray
            Solution as an element of grid
        g_map : numpy.ndarray
            Complex simulated g_map
        mask_a : numpy.ndarray
            Mask of object
        mask_p : numpy.ndarray
            Array of ones with same dimensions as mask_a
        opdata : list
            List containing grid, fresnel_number, pad_amount, a_psi0_multiplier for use as
            parameters in operators.get_op_g_to_data
    """  
    filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),'data','simulated_g.mat')
    g_map, mask, mask_binary, px_size = load_simulated_g(filename)
    mask_a = ~mask_binary
    lambda_electron = 2.51e-12
    defocus = 900e-6
    theta_divergence = 5e-6
    fresnel_number = 1./(defocus * lambda_electron - 1j*theta_divergence**2 * defocus**2/np.log(2))
    N1,N2 = mask.shape
    a_psi0_multiplier = mask.astype(complex)

    grid = UniformGridFcts(np.arange(N1)*px_size[0][0],np.arange(N2)*px_size[0][1])
    pad_amount = ((50,0),(0,0))
    opdata = [grid, fresnel_number,pad_amount,a_psi0_multiplier]
    
    op = get_op_g_to_data(*opdata, 
                list_of_filters = list_of_filters,
                N=N, 
                parallel=parallel)
    exact_solution = op.domain.join(np.log(np.abs(g_map)),
                                    np.unwrap(np.angle(g_map.T)).T)
    return op, grid, exact_solution, g_map, mask_a, np.ones_like(mask_a), opdata
