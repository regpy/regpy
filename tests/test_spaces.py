from regpy.vecsp import *
from regpy import hilbert


# Tests if the Spaces registry works

def test_L2_uniform_grid():
    grid = UniformGridFcts(10,11)
    hilbert.L2(grid)

def test_sobolev_uniform_grid():
    grid = UniformGridFcts(10,11)
    hilbert.Sobolev(grid)

def test_L2_grid():
    grid = Grid(axisdata = (np.linspace(0,10,5),np.logspace(-1,5,15)))
    hilbert.L2(grid)

def test_L2_directsum():
    grid1 = Grid(axisdata = (np.linspace(0,10,5),np.logspace(-1,5,15)))
    grid2 = UniformGridFcts(10,11)
    grid = grid1 + grid2
    hilbert.L2(grid)

def test_sobolev_directsum():
    grid = UniformGridFcts(10,11)
    grid = grid + grid
    hilbert.Sobolev(grid)



# Does not Work 

# def test_L2Boundary_directsum():
#     grid1 = Grid(axisdata = (np.linspace(0,10,5),np.logspace(-1,5,15)))
#     grid = UniformGridFcts(10,11)
#     grid = grid + grid
#     hilbert.L2Boundary(grid)

# def test_Hm_grid():
#     # grid = Grid(axisdata = (np.linspace(0,10,5),np.logspace(-1,5,15)))
#     grid = Grid(10,11)
#     hilbert.Hm(grid,)
    
