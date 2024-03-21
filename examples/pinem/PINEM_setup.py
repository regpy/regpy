import numpy as np
from scipy.io import loadmat
from regpy.operators import Operator
from operators import get_op_g_to_data, complex_get_op_g_to_data
from regpy.operators import Operator, SquaredModulus, Exponential, PtwMultiplication, VectorOfOperators
from regpy.vecsps import UniformGridFcts, DirectSum

def load_simulated_g(filename):
    mat = loadmat(filename)
    g_map = mat['pm']['g_map'][0][0]
    mask = mat['pm']['mask'][0][0]
    mask_binary = mat['pm']['mask_binary'][0][0].astype(dtype=bool)
    px_size = mat['pm']['px_sizes'][0][0]
    px_size = px_size * 1e-9 #convert nm to m
    return g_map, mask, mask_binary, px_size

def setup_simulated_g(g_is_complex=False,using_gabs_measurement=True, parallel=True,list_of_filters=None,N=30):
    filename = r"./data/FresnelPinemMap_obj_javier_2.mat"
    g_map, mask, mask_binary, px_size = load_simulated_g(filename)
    mask_a = ~mask_binary
    #fov = tuple(x*px_size for x in mask.shape)
    lambda_electron = 2.51e-12
    defocus = 900e-6
    #fresnel_number = np.prod(fov)/(defocus * lambda_electron)
    theta_divergence = 5e-6
    fresnel_number = 1./(defocus * lambda_electron - 1j*theta_divergence**2 * defocus**2/np.log(2))
    #fresnel_number = 1./(defocus * lambda_electron)
    # Uniform grid
    N1,N2 = mask.shape
    a_psi0_multiplier = mask.astype(complex)

    #grid = UniformGridFcts(np.linspace(0, 1, N1, endpoint=False),
    #                       np.linspace(0, 1, N2, endpoint=False))
    grid = UniformGridFcts(np.arange(N1)*px_size[0][0],np.arange(N2)*px_size[0][1])
    pad_amount = ((50,0),(0,0))
    opdata = [grid, fresnel_number,pad_amount,a_psi0_multiplier]
    
    if g_is_complex:
        op = complex_get_op_g_to_data(*opdata, 
            list_of_filters = list_of_filters,
            N=N, 
            parallel=parallel
        )
        if using_gabs_measurement:
            op2 = PtwMultiplication(grid,1.0-mask_a) * SquaredModulus(grid.complex_space())
            op = VectorOfOperators([op2, op])
        return op, grid, g_map, g_map, mask_a, np.ones_like(mask_a), opdata
    else:
        op = get_op_g_to_data(*opdata, 
                    list_of_filters = list_of_filters,
                    N=N, 
                    parallel=parallel
                    )
        exact_solution = op.domain.join(np.log(np.abs(g_map)),
                                        np.unwrap(np.angle(g_map.T)).T)
        if using_gabs_measurement:
            op2 = PtwMultiplication(grid,1.0-mask_a) * Exponential(grid.real_space) * ForgetSecond(grid,grid)
            op = VectorOfOperators([op2, op])

        return op, grid, exact_solution, g_map, mask_a, np.ones_like(mask_a), opdata

##################### operator needed for fixing g on parts of the grid where its values are known

class ForgetSecond(Operator):
    def __init__(self,domain1,domain2):
        self.domain2 = domain2
        super().__init__(DirectSum(domain1, domain2),domain2,linear=True)

    def _eval(self,x):
        x1,x2 = self.domain.split(x)
        return x1
   
    def _adjoint(self,y):
        return self.domain.join(y,self.domain2.zeros())