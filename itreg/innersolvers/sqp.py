class SQP(Inner_Solver):


    def __init__(self, op, data, init, stepsize=None):
        super().__init__(logging.getLogger(__name__))
        self.op = op
        self.data = data
        self.setx(init)
        self.stepsize = stepsize or 1 / self.deriv.norm()
        
        
        #some parameters
        # maximum number of CG iterations
        self.N_CG = 50;
        # replace KL(a,b) by KL(a+offset, b+offset)
        self.offset0 =2e-6;
        # offset is reduced in each Newton step by a factor offset_step
        self.offset_step = 0.8;
        # relative tolerance value for termination of inner iteration
        self.update_ratio = 0.01;
        # max number of inner iterations to minimize the KL-functional
        self.inner_kl_it = 10;
        
        def setx(self, x):
        """Set the current point of the solver.

        Update the function value and the derivative accordingly.

        Parameters
        ----------
        x : array
            The new point.

        """
        self.x = x
        self.y = self.op(self.x)
        self.deriv = self.op.derivative()
        # These are pre-computed as they are needed for the next step *and* for
        # logging
        self._residual = self.y - self.data
        self._gy_residual = self.op.domy.gram(self._residual)
