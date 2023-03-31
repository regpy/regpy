import numpy as np
from scipy.io import loadmat
from regpy.operators import Operator
from regpy.operators.PINEM import PINEM_g_to_data, complex_PINEM_g_to_data
from regpy.operators import Operator, SquaredModulus, Exponential, Ptw_Multiplication, Vector_of_operators
from regpy.discrs import UniformGrid, DirectSum

def load_simulated_g(filename):
    mat = loadmat(filename)
    g_map = mat['pm']['g_map'][0][0]
    mask = mat['pm']['mask'][0][0]
    mask_binary = mat['pm']['mask_binary'][0][0].astype(dtype=bool)
    px_size = mat['pm']['px_sizes'][0][0]
    px_size = px_size * 1e-9 #convert nm to m
    return g_map, mask, mask_binary, px_size

def setup_simulated_g(g_is_complex=True,using_g_squared_measurement=True, parallel=True,list_of_filters=None,N=30):
    filename = r"./data/FresnelPinemMap_obj_javier_2.mat"
    g_map, mask, mask_binary, px_size = load_simulated_g(filename)
    fov = tuple(x*px_size for x in mask.shape)
    lambda_electron = 2.51e-12
    defocus = 900e-6
    fresnelNumber = np.prod(fov)/(defocus * lambda_electron)
    # Uniform grid
    N1,N2 = mask.shape
    A_Psi0_Multiplier = mask.astype(complex)
    boundary_mask = np.zeros_like(mask_binary)
    boundary_mask[0,:]=True; boundary_mask[-1,:]=True
    boundary_mask[:,0]=True; boundary_mask[:,-1]=True

    grid = UniformGrid(np.linspace(0, 1, N1, endpoint=False),
                           np.linspace(0, 1, N2, endpoint=False))
    opdata = [grid, fresnelNumber,mask_binary & ~boundary_mask,A_Psi0_Multiplier]
    if g_is_complex:
        op = complex_PINEM_g_to_data(*opdata, 
            list_of_filters = list_of_filters,
            N=N, 
            parallel=parallel
        )
        if using_g_squared_measurement:
            op2 = Ptw_Multiplication(grid,1.0-mask_binary) * SquaredModulus(grid.complex_space())
            op = Vector_of_operators([op2, op])
        return op, grid, g_map, g_map, mask_binary, ~boundary_mask,opdata
    else:
        op = PINEM_g_to_data(*opdata, 
                    list_of_filters = list_of_filters,
                    N=N, 
                    parallel=parallel
                    )
        exact_solution = op.domain.join(np.log(np.abs(g_map)),
                                        np.unwrap(np.angle(g_map.T)).T)
        if using_g_squared_measurement:
            op2 = Ptw_Multiplication(grid,1.0-mask_binary) * SquaredModulus(grid.real_space()) \
                * Exponential(grid.real_space) * ForgetSecond(grid,grid)
            op = Vector_of_operators([op2, op])

        return op, grid, exact_solution, g_map, mask_binary, ~boundary_mask, opdata

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