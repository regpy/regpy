from itreg import discrs, hilbert

hilbert.L2.register(discrs.DirectSum, hilbert.componentwise(hilbert.L2))
hilbert.L2.register(discrs.Discretization, hilbert.L2Generic)
hilbert.L2.register(discrs.UniformGrid, hilbert.L2UniformGrid)

hilbert.Sobolev.register(discrs.DirectSum, hilbert.componentwise(hilbert.Sobolev))
hilbert.Sobolev.register(discrs.UniformGrid, hilbert.SobolevUniformGrid)

hilbert.L2Boundary.register(discrs.DirectSum, hilbert.componentwise(hilbert.L2Boundary))

hilbert.SobolevBoundary.register(discrs.DirectSum, hilbert.componentwise(hilbert.SobolevBoundary))
