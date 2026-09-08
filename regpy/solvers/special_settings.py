
from __future__ import annotations
import math as ma
import numpy as np
from typing import Callable
from copy import deepcopy

from regpy.util import ClassLogger, Errors
from regpy.vecsps import Vector, ReadOnlyVector
from regpy.operators import Operator, ObsDataT
from regpy.hilbert import HilbertSpace
from regpy.functionals.base import  as_functional, Composed
from regpy.functionals import Functional
from regpy.stoprules import (
    StopRule, DualityGapStopping, CountIterations, ErrorMonitor,
)
from regpy.hilbert import L2

from .setting import Setting

__all__ = ['HilbertSetting']

"""
Special settings are mainly convenience classes. 
Their main purpose of specialized setting is to define default inner and outer solvers 
that work "best" (or at least reasonably good) for certain combinations of data fidelity 
and penalty term. This default solver can be employed by the specialized setting's run 
method and allows to quickly get a solution to an inverse problem without having 
to perform tests of all applicable solvers. This holds rue in particular to nonlinear 
inverse problems where inner solvers are often needed. Specialized settings allow to 
to set up such combined solvers in a simple way.

Moreover, some specialized settings implement formulas for the duality gap that 
avoid loss of accuracy by cancellation effects.
"""

class HilbertSetting(Setting):
    def  __init__(self,
            op: Operator, 
            penalty: Functional|HilbertSpace|Callable|None = None, 
            data_fid:Functional|HilbertSpace|Callable|None = None,
            regpar: float|None =None,
            data: ReadOnlyVector | ObsDataT | None = None, 
            exact_data: ReadOnlyVector | None = None, 
            init: ReadOnlyVector | None = None,
            incorporate_init_in_penalty: bool = True,
            codomain_free: bool = False,
            logging_level = "INFO",
            explicitDataWarning:bool = True,                 
            _primal_setting:Setting | None =None # intended only for internal use in get_dual_setting()
             )->None:
              
        penalty = L2(op.domain) if penalty is None else penalty
        data_fid = L2(op.codomain) if data_fid is None else data_fid
        super().__init__(op, penalty, data_fid, regpar, 
                              data = data, 
                              exact_data = exact_data, 
                              init=init, 
                              incorporate_init_in_penalty=incorporate_init_in_penalty,
                              codomain_free=codomain_free, 
                              logging_level=logging_level, 
                              explicitDataWarning=explicitDataWarning, 
                              _primal_setting=_primal_setting,
                              default_linear_regmethod = 'CGNE',
                              default_nonlinear_regmethod = 'NewtonCG',
                              default_inner_solver = 'TikhCG'
                              )
        if not self.penalty.is_norm_squared_type:
            raise RuntimeError(Errors.runtime_error('Penalty of HilbertSetting must be of squared norm type.',self.penalty))
        if not self.data_fid.is_norm_squared_type:
            raise RuntimeError(Errors.runtime_error('Data fidelity term of HilbertSetting must be of squared norm type.',self.data_fid))

    def duality_gap(self, primal = None, dual = None):

        return super().duality_gap(primal,dual)

