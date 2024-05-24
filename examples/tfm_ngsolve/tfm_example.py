import logging
import numpy as np
import ngsolve as ngs
from ngsolve.webgui import Draw
from netgen.occ import *

from regpy.vecsps.ngsolve import NgsSpace
from regpy.solvers import RegularizationSetting
from regpy.solvers.nonlinear.landweber import Landweber
import regpy.stoprules as rules
from regpy.hilbert import L2Boundary, Hm0

from tfm import TFM

# ## Create Operators
# 
# First we import logging to later get printed messages during our solution algorithm.
# 
# To define `domain` and `codomain` of our operator, we need `regpy.vecsps.ngsolve.NgsSpace`. An `NgsSpace` in regpy consists of a finite element space from ngsolve and possibly a boundary. First we generate a cuboid as the mesh and label the top and bottom boundary. Then we can define the three dimensional H1 finite element space with dirichlet boundary conditions at the bottom. Finally we can define `domain` and `codomain` of our operator.
# 
# To prevent commiting an inverse crime, we use to different discretizations for `op_gen` - used to generate simulated data (the displacement) - and `op_rec` - used to reconstruct the traction forces.


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)-20s :: %(message)s'
)

'''############################### material parameters ###############################'''

# E Young's modulus, nu Poisson ratio
E, nu = 10000, 0.45
# Lamé parameters
mu = E / 2 / (1 + nu)
lam = E * nu / ((1 + nu) * (1 - 2 * nu))

'''############################### mesh of substrate ################################'''

# generate two triangular meshes of mesh-size h, order k, for generating and reconstructing traction forces
h_gen = 0.3
h_rec = 0.4
k = 3


box = Box(Pnt(-2, -0.3,-2), Pnt(2, 0.3,2))
box.faces.Max(Y).name = 'top'
box.faces.Min(Y).name = 'bottom'
geo = OCCGeometry(box)

mesh_gen = ngs.Mesh(geo.GenerateMesh(maxh=h_gen))
mesh_gen.Curve(3)

Draw(mesh_gen)

mesh_rec = ngs.Mesh(geo.GenerateMesh(maxh=h_rec))
mesh_rec.Curve(3)

'''############################### define operator ################################'''
# operator for generating data on a different discretization than reconstruction operator

fes_domain_gen = ngs.VectorH1(mesh_gen, order=k, dirichlet="bottom")
# fes_domain = ngs.H1(mesh, order=k, dim=3)
domain_gen = NgsSpace(fes_domain_gen, bdr='top')

fes_codomain_gen = ngs.VectorH1(mesh_gen, order=k, dirichlet="bottom")
# fes_codomain = ngs.H1(mesh, order=k, dirichlet="bottom|side", dim=3)
codomain_gen = NgsSpace(fes_codomain_gen)

op_gen = TFM(domain_gen, codomain=codomain_gen, mu=mu, lam=lam)


fes_domain_rec = ngs.VectorH1(mesh_rec, order=k, dirichlet="bottom")
# fes_domain = ngs.H1(mesh, order=k, dim=3)
domain_rec = NgsSpace(fes_domain_rec, bdr='top')

fes_codomain_rec = ngs.VectorH1(mesh_rec, order=k, dirichlet="bottom")
# fes_codomain = ngs.H1(mesh, order=k, dirichlet="bottom|side", dim=3)
codomain_rec = NgsSpace(fes_codomain_rec)

op_rec = TFM(domain_rec, codomain=codomain_rec, mu=mu, lam=lam)


# ## Generate Groundtruth
# 
# In this step we generate a toy surface traction force first as a `CoefficientFunction`from ngsolve. Because the surface traction force should just act on the boundary, we interpolate `traction_true_cf` to the top boundary of the generated mesh. The resulting gridfunction `traction_true_gf` acts then only on the top boundary.
# 
# To change between ngsolve and regpy, the functions `to_ngs` and `from_ngs` can be used.



a = 4
c = 1

force_intensity = (-a * (ngs.sqrt(ngs.x**2 + ngs.z**2) - 0.5)**2 + c )

cell_force = ngs.CF(force_intensity * (-ngs.x/(ngs.sqrt(ngs.x**2 + ngs.z**2)),0,-ngs.z/(ngs.sqrt(ngs.x**2 + ngs.z**2))))

cell_xz = -(ngs.sqrt(ngs.x ** 2 + ngs.z ** 2) - 1)


traction_true_cf = ngs.IfPos(c1=cell_xz, then_obj=cell_force, else_obj=ngs.CF((0, 0, 0)))

traction_true_gf = ngs.GridFunction(fes_domain_gen)
traction_true_gf.Set(traction_true_cf, definedon = mesh_gen.Boundaries( "top" ))

traction_true = domain_gen.from_ngs(traction_true_gf)

Draw(traction_true_gf, mesh_gen)


# ## Generate Data
# 
# Next, we generate the displacement by applying the forward operator `op_gen` to the true traction force. Then we add random noise to the displacement and plot noiseless displacement on the generating mesh and noisy displacement on the reconstructing mesh.



displacement_true = op_gen(traction_true)


Draw(codomain_gen.to_ngs(displacement_true))



np.random.seed(42)


noise = 1e-6 * codomain_rec.randn()
data_gf = ngs.GridFunction(fes_codomain_rec)
data_gf.Set(codomain_gen.to_ngs(displacement_true))
data = codomain_rec.from_ngs(data_gf) + noise

Draw(codomain_rec.to_ngs(data), mesh_rec)


# ## Solve the Inverse Problem with regularization
# 
# Now, with available data we can use, e.g., landweber iteration to reconstruct the groundtruth traction force from the noisy displacement.
# 
# First, we can define the regularization setting using `RegularizationSetting`. To measure the error, we have to choose a penalty Hilbert space structure. Here we choose `penalty=L2Boundary` because the traction forces are elements of $L^2(\Gamma_{Top})$. Similar to measure the data misfit, we have to choose a data fidelity Hilbert space. Here we choose our Hilbert space `data_fid=Hm0` because the displacement is an element of $H^1_{0,\Gamma_{Bottom}}(\Omega)$. `L2Boundary` and `Hm0` are `AbstractSpace`s and the according implementation can be found in `regpy.hilbert.ngsolve`. 
# 
# As a next step we define the we choose an initial guess, in this case we choose zero. 
# 
# With this we can use Landweber iteration implemented in `regpy.solvers.nonlinear.landweber` to reconstruct the exact solution from the above constructed noisy data `data`.
# 
# We stop the iterative algorithm after at most $100$ steps and have as early stopping criterion the discrepancy rule implemented. This can be easily done by summing the two instances of the `regpy.stoprules`. 
# 
# After everything is defined run the solver with the specified stopping rule using the method `run()` of the solver.




setting = RegularizationSetting(op=op_rec, penalty=L2Boundary, data_fid=Hm0)
init = domain_rec.from_ngs((0, 0, 0))

landweber = Landweber(setting, data, init)

stoprule = (
        rules.CountIterations(100) +
        rules.Discrepancy(setting.h_codomain.norm, data, noiselevel=setting.h_codomain.norm(noise), tau=1.04)
)

reco, reco_data = landweber.run(stoprule)

reco_gf = domain_rec.to_ngs(reco)


# ## Compute Error

# calculate relative L2 error on boundary
err_abs = ngs.sqrt(ngs.Integrate((reco_gf - traction_true_cf)**2, mesh_rec,definedon=mesh_rec.Boundaries('top')))
norm_true = ngs.sqrt(ngs.Integrate((traction_true_cf)**2, mesh_rec,definedon=mesh_rec.Boundaries('top'))) 
norm_rec = ngs.sqrt(ngs.Integrate((reco_gf)**2, mesh_rec,definedon=mesh_rec.Boundaries('top')))
err_rel = err_abs/norm_true * 100

print('relative error:' , err_rel, '%')


# ## Plot Reconstruction
# 
# We plot the true traction forces on the reconstruction mesh, the reconstruction and the error.


Draw(traction_true_gf, domain_rec.fes.mesh)

Draw(reco_gf)

error = traction_true_gf - domain_rec.to_ngs(reco)

Draw(error, domain_rec.fes.mesh)





