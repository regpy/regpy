from regpy.vecsps import *
from regpy import hilbert
import numpy as np

def check_parallelogram_identity(h_space,tol=1e-10):
    u = h_space.vecsp.randn()
    v = h_space.vecsp.randn()
    assert abs(h_space.norm(u+v)**2-h_space.norm(u-v)**2-4*(h_space.inner(u,v)))<tol

# Tests if the Spaces registry works

def test_L2_uniform_grid():
    grid = UniformGridFcts(10,11)
    check_parallelogram_identity(hilbert.L2(grid))
    grid = UniformGridFcts(10,11,dtype=np.complex128)
    check_parallelogram_identity(hilbert.L2(grid))

def test_L2_grid():
    grid = GridFcts(axisdata = (np.linspace(0,10,10),np.logspace(-1,5,15)))
    check_parallelogram_identity(hilbert.L2(grid))
    grid = GridFcts(axisdata = (np.linspace(0,10,10),np.logspace(-1,5,15)),dtype=np.complex128)
    check_parallelogram_identity(hilbert.L2(grid))

def test_L2_measure_space():
    grid = MeasureSpaceFcts(measure=np.abs(np.random.randn(10,15))+0.1)
    check_parallelogram_identity(hilbert.L2(grid))
    grid = MeasureSpaceFcts(measure=np.abs(np.random.randn(10,15))+0.1,dtype=np.complex128)
    check_parallelogram_identity(hilbert.L2(grid))

def test_L2_directsum():
    grid1 = GridFcts(axisdata = (np.linspace(0,10,5),np.logspace(-1,5,15)))
    grid2 = UniformGridFcts(10,11,dtype=np.complex128)
    grid = grid1 + grid2
    check_parallelogram_identity(hilbert.L2(grid))

def test_L2_tensorprod():
    grid1 = GridFcts(axisdata = (np.linspace(0,10,5),np.logspace(-1,5,15)),dtype=np.complex128)
    grid2 = UniformGridFcts(10,11,dtype=np.complex128)
    grid = grid1 * grid2
    check_parallelogram_identity(hilbert.L2(grid), tol = 1e-9)

def test_sobolev_uniform_grid():
    grid = UniformGridFcts(10,11)
    check_parallelogram_identity(hilbert.Sobolev(grid))
    grid = UniformGridFcts(10,11,dtype=np.complex128)
    check_parallelogram_identity(hilbert.Sobolev(grid))

def test_HmDomain_uniform_grid():
    grid = UniformGridFcts(10,11)
    mask = grid.ones()
    check_parallelogram_identity(hilbert.Hm(grid,mask = mask))
    grid = UniformGridFcts(10,11,dtype=np.complex128)
    check_parallelogram_identity(hilbert.Hm(grid,mask = mask))

    
