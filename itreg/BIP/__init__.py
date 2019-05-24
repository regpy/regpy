from itreg.util import classlogger
import numpy as np

class Solver_BIP:
    

    log = classlogger

    def __init__(self):
        self.num=0

    
    def MC(self, maxhits=None):
        if maxhits==None:
            maxhits=self.maxhits
        return self.evaluation.MC(maxhits)
    
    
class State(object):
    __slots__ = ('positions', 'log_prob')

class HMCState(State):
    __slots__ = ('positions', 'momenta', 'log_prob')

class PDF(object):
    """Abstract class for probability density functions
    """
    def log_prob(self, state):
        raise NotImplementedError

    def gradient(self, state):
        raise NotImplementedError 
        

                



