import netgen.gui
from netgen.geom2d import SplineGeometry

geo=SplineGeometry()
geo.AddCircle ( (0, 0), r=1, bc="cyc", maxh=0.2)
ngmesh = geo.GenerateMesh()
ngmesh.Save('meshes\circle')
