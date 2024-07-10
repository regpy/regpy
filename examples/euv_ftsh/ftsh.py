# %%
import numpy as np
from copy import deepcopy
import regpy
# from regpy.operators import Identity, Operator, DirectSum
# from regpy.operators import ImaginaryPart, RealPart, SquaredModulus
# from regpy.operators import Exponential, FourierTransform, PtwMultiplication
# from regpy.operators import MatrixOfOperators, VectorOfOperators
# from regpy.operators.parallel_operators import ParallelVectorOfOperators
# from regpy.operators import CoordinateProjection, CoordinateMask
# from regpy.vecsps import GridFcts, UniformGridFcts
# from regpy.vecsps import DirectSum as DirectSumDis
# CONSTRUCTION SITE! NOT READY FOR USE!!

#%%
class sumoverwl(regpy.operators.Operator):
    """Operator that sums along a given axis.
    
    Parameters
    ----------
    domain: underlying VectorSpace.
    freqaxis: the axis along which to sum.
    Returns
    -------
    regpy.operators.Operator
    """
    def __init__(self, domain, freqax=0, dirsum=False):
        
        self.freqax = freqax
        self.dirsum = dirsum
        if dirsum is True:
            assert type(domain) is regpy.vecsps.DirectSum
            assert domain.__len__() == 1
            if type(domain[0]) is regpy.vecsps.GridFcts:
                codomain = regpy.vecsps.GridFcts(*np.delete(np.array(domain[0].axes, dtype=object), freqax),dtype=domain.dtype)
            elif type(domain[0]) is regpy.vecsps.UniformGridFcts:
                codomain = regpy.vecsps.UniformGridFcts(*np.delete(np.array(domain[0].axes, dtype=object), freqax),dtype=domain.dtype)

            self.matrixshape = tuple([dim if ii == self.freqax else 1 for ii,dim in enumerate(domain[0].shape)])
        
        else:
            if type(domain) is regpy.vecsps.GridFcts:
                codomain = regpy.vecsps.GridFcts(*np.delete(np.array(domain.axes, dtype=object), freqax),dtype=domain.dtype)
            elif type(domain) is regpy.vecsps.UniformGridFcts:
                codomain = regpy.vecsps.UniformGridFcts(*np.delete(np.array(domain.axes, dtype=object), freqax),dtype=domain.dtype)
        
            self.matrixshape = tuple([dim if ii == self.freqax else 1 for ii,dim in enumerate(domain.shape)])
        
        self.domain = domain
        self.codomain = codomain

        super().__init__(
            self.domain,
            self.codomain,
            linear=True
        )

    def _eval(self, x, differentiate = False):
        if self.dirsum is True:
            x2 = self.domain.split(x)[0]
            return np.sum(x2, axis=self.freqax)
        else:
            return np.sum(x, axis=self.freqax)

    def _adjoint(self, y):
        if self.dirsum is True:
            z = self.domain.join(np.tile(np.expand_dims(y, self.freqax), self.matrixshape))
        else:
            z = np.tile(np.expand_dims(y, self.freqax), self.matrixshape)
        return z

# %%
def spectroscopic_FTH(domain, freqaxis=0, imageaxes=(1, 2),
                      delays=[0.0], frequencies=[1.], masking_operator=None,
                      parallel=False):
    """
    Operator that takes a number of complex-valued spectral modes and their corresponding frequencies,
    and calculates for each delay the sum.

    Parameters
    ----------
    domain: regpy.vecsps.UniformGridFcts
        Should be of shape 1+2, representing frequencies and
        x and y pixel coordinates of the sample plane.
    freqaxis: int, optional
        Axis index of the wavelength axis, by default 0
    imageaxes: tuple, optional
        Indices of the sample and detector plane axes, by default (1,2)
    delays: list of floats, optional
        List of delays between reference and probe field, by default 0.0
    parallel : bool
        Calculate individual delays in parallel

    Returns
    -------

    """
    assert domain.is_complex
    assert not np.any(np.isin(imageaxes, freqaxis))

    frequencies = np.array(frequencies)

    if masking_operator is not None:
        masking = masking_operator
        embedding = masking.adjoint
    else:
        embedding = regpy.operators.Identity(domain)

    oplist = []
    for delay in delays:
        phase_factor = np.exp(-1j * 2 * np.pi
                              * np.moveaxis(frequencies[:, np.newaxis, np.newaxis], 0, freqaxis)
                              * delay)
        phasemultiplier = regpy.operators.PtwMultiplication(embedding.codomain, phase_factor)
        summing = sumoverwl(phasemultiplier.codomain, freqax=freqaxis)

        oplist.append(summing * phasemultiplier)

    if parallel:
        vecofops = regpy.operators.ParallelVectorOfOperators(oplist)
    else:
        vecofops = regpy.operators.VectorOfOperators(oplist)

    return vecofops * embedding

# %%
