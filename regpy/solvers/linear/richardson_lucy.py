from regpy.solvers import Solver 
import numpy as np

class RichardsonLucy(Solver):
    def __init__(self,op,data,x_init=None,sigma=0):
        super().__init__()
        self.op=op
        self.data=data
        if(x_init==None):
            x_init=self.op.domain.ones()
        self.sigma = sigma
        self.x=x_init
        self.y=self.op(self.x)
        self.adj_ones=self.op.adjoint(self.op.codomain.ones())
        assert (self.adj_ones>0).all()


    def _next(self):
        multiplier=self.op.adjoint(self.data/(self.y+self.sigma*self.op.codomain.ones()))
        self.x=multiplier*self.x/self.adj_ones
        self.y=self.op(self.x)
