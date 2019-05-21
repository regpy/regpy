from ngsolve import *

class PDEBase:
    def __init__(self):
        return
    
    def Solve(self, params, bilinear, rhs):
        return bilinear.mat.Inverse(freedofs=params.fes.FreeDofs()) * rhs


