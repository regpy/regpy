from . import StopRule, MissingValueError
import numpy as np

class RelativeChangeData(StopRule):
    
    """Stops if the relative change in the residual becomes small

    Stops at the first iterate at which the difference between the old residual
    and the new residual is smaller than a pre-determined cutoff::

        ||y_k-y_{k+1}|| < delta

    Parameters
    ----------
    norm : callable
        The norm with respect to which the difference should be measured.
        Usually this will be the `norm` method of some :class:`~itreg.spaces.Space`.
    cutoff : float
        The cutoff value at which the iteration should be stopped
    data : np array
        The data array
    """
    
    def __init__(self, norm, data, cutoff):
        super().__init__()
        self.norm=norm
        self.cutoff=cutoff
        self.data_old=data
        
    def __repr__(self):
        return 'RelativeChangeData(cutoff={})'.format(
            self.cutoff)
        
    def _stop(self, x, y=None):
        if y is None:
            raise MissingValueError
        change=self.norm(y-self.data_old)
        self.data_old=y
        self.log.info('RelativeChangeData = {}, cutoff = {}'.format(
            change, self.cutoff))
        return change < self.cutoff

class RelativeChangeSol(StopRule):    

    """Stops if the relative change in the solution space becomes small

    Stops at the first iterate at which the difference between the old estimate
    and the new estimate is smaller than a pre-determined cutoff::

        ||y_k-y_{k+1}|| < delta

    Parameters
    ----------
    norm : callable
        The norm with respect to which the difference should be measured.
        Usually this will be the `norm` method of some :class:`~itreg.spaces.Space`.
    cutoff : float
        The cutoff value at which the iteration should be stopped
    init : np array
        initial guess
    """

    def __init__(self, norm, init, cutoff):
        super().__init__()
        self.norm=norm
        self.cutoff=cutoff
        self.sol_old=init
        
    def __repr__(self):
        return 'RelativeChangeSol(cutoff={})'.format(
            self.cutoff)
        
    def _stop(self, x, y=None):
        change=self.norm(x-self.sol_old)
        self.sol_old=x
        self.log.info('RelativeChangeSol = {}, cutoff = {}'.format(
            change, self.cutoff))
        return change < self.cutoff
    
class Monotonicity(StopRule):
    
    """Stops if the residual is growing again.
    
        Parameters
    ----------
    norm : callable
        The norm with respect to which the difference should be measured.
        Usually this will be the `norm` method of some :class:`~itreg.spaces.Space`.
    cutoff : float
        The cutoff value at which the iteration should be stopped
    data : np array
        The data array
    init_data : np array
        initial guess in data space
    """
        
    def __init__(self, norm, data, init_data):
        self.norm=norm
        self.data=data
        self.residual=self.norm(self.data-init_data)
        
    def __repr__(self):
        return 'Monotonicty'
    
    def _stop(self, x, y=None):
        if y is None:
            raise MissingValueError
        residual=self.norm(self.data-y)
        change=self.residual-residual
        self.residual=residual
        self.log.info('Monotonicity = {}'.format(
            change))
        return change<0
        
    