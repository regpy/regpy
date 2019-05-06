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



