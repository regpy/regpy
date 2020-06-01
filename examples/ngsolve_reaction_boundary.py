# Run this file in IPython like
#     import netgen.gui
#     %run path/to/this/file
# to get graphical output.

import logging
import ngsolve as ngs
import numpy as np
from netgen.geom2d import SplineGeometry

import regpy.stoprules as rules
from regpy.operators.ngsolve import ReactionNeumann, ProjectToBoundary
from regpy.solvers import HilbertSpaceSetting
from regpy.solvers.landweber import Landweber
from regpy.hilbert import L2, SobolevBoundary
from regpy.discrs.ngsolve import NgsSpace

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)-40s :: %(message)s'
)

geo = SplineGeometry()
bc = "cyc"
geo.AddCircle((0, 0), r=1, bc=bc, maxh=0.2)
mesh = ngs.Mesh(geo.GenerateMesh())

fes_domain = ngs.H1(mesh, order=1)
domain = NgsSpace(fes_domain)

fes_complete_codomain = ngs.H1(mesh, order=4)
complete_codomain = NgsSpace(fes_complete_codomain, bdr=bc)

fes_codomain = ngs.H1(mesh, order=0)
codomain = NgsSpace(fes_codomain, bdr=bc)

g = 0.1*ngs.y
#The reaction coefficient operator with Neumann boundary conditions
reac = ReactionNeumann(domain, g, codomain=complete_codomain)
#Projection of distributed measurements to boundary
proj = ProjectToBoundary(complete_codomain, codomain=codomain)
op = proj * reac

exact_solution_coeff =  ngs.x + 2
exact_solution = domain.from_ngs( exact_solution_coeff )
exact_data = op(exact_solution)

noise = proj( 0*0.0005*complete_codomain.randn() )

data = exact_data+noise

init = domain.from_ngs( 2 )

setting = HilbertSpaceSetting(op=op, Hdomain=L2, Hcodomain=SobolevBoundary)

landweber = Landweber(setting, data, init, stepsize=1)
stoprule = (
        rules.CountIterations(1000) +
        rules.Discrepancy(setting.Hcodomain.norm, data, noiselevel=setting.Hcodomain.norm(noise), tau=0))

reco, reco_data = landweber.run(stoprule)

ngs.Draw(exact_solution_coeff, fes_domain.mesh, "exact")

# Draw reconstructed solution
domain.draw(reco, 'reco')

# Draw data space
codomain.draw(reco_data, 'reco_data')
codomain.draw(data, 'data')