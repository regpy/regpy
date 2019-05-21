from ngsolve import *

class PDEBase:
    def __init__(self):
        return
    
    def Solve(self, params, bilinear, rhs):
#        pre = Preconditioner(bilinear, 'local')
#        a.Assemble()
#        inv = CGSolver(bilinear.mat, pre.mat, maxsteps=1000)
#        gfu.vec.data = inv * f.vec
        return bilinear.mat.Inverse(freedofs=params.fes.FreeDofs()) * rhs
#        return inv*rhs
        


