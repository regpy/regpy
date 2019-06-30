import netgen.gui
from netgen.geom2d import SplineGeometry

geo=SplineGeometry()
geo.AddCircle((0,0), 1, bc="circle")
ngmesh = geo.GenerateMesh()
ngmesh.Save('meshes\circle')

