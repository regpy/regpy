
import numpy as np

from regpy.operators import Operator
from regpy.vecsps import UniformGridFcts
from examples.volterra.volterra import Volterra

def test_volterra():
    grid = UniformGridFcts(np.linspace(0, 2 * np.pi, 200))

    exact_solution = (1-np.cos(grid.coords[0]))**2/4 

    from regpy.hilbert import L2, Sobolev
    from regpy.functionals import HilbertNorm, TV
    from regpy.solvers import RegularizationSetting, TikhonovRegularizationSetting
    import regpy.stoprules as rules
    from regpy.solvers.linear.tikhonov import TikhonovCG

    op = Volterra(grid)

    exact_data = op(exact_solution)
    noise = 0.03 * op.domain.randn()
    data = exact_data + noise

    setting = RegularizationSetting(op, L2, L2)

    solver = TikhonovCG(setting, data, regpar=0.01)
    stoprule = (
        rules.CountIterations(1000) +
        rules.Discrepancy(
            setting.h_codomain.norm, data,
            noiselevel=setting.h_codomain.norm(noise),
            tau=1.1
        )
    )

    reco, reco_data = solver.run(stoprule)

    from regpy.solvers.nonlinear.landweber import Landweber

    op = Volterra(grid,exponent=2)

    exact_data = op(exact_solution)
    noise = 0.03 * op.domain.randn()
    data = exact_data + noise
    init = op.domain.ones()*0.05

    setting = RegularizationSetting(op, Sobolev, L2)

    solver = Landweber(setting, data, init, stepsize=0.01)
    stoprule = (
        # Landweber is slow, so need to use large number of iterations
        rules.CountIterations(max_iterations=100000) +
        rules.Discrepancy(
            setting.h_codomain.norm, data,
            noiselevel=setting.h_codomain.norm(noise),
            tau=1.1
        )
    )

    reco, reco_data = solver.run(stoprule)

    from regpy.solvers.nonlinear.fista import FISTA

    op = Volterra(grid, exponent=2)

    # Impulsive Noise
    sigma = 0.01*np.ones(grid.coords.shape[1])
    sigma[100:110] = 0.5

    exact_data = op(exact_solution)
    noise = sigma * op.domain.randn()
    data = exact_data + noise
    init = op.domain.ones()

    #The penalty term |f|_{TV}
    setting = TikhonovRegularizationSetting(
        op=op, 
        penalty=TV(h_domain=Sobolev), 
        data_fid=HilbertNorm(h_space=L2), 
        data_fid_shift = data,
        regpar = 0.01
    )

    proximal_pars = {
            'stepsize' : 0.001,
            'maxiter' : 100
            }
    # """Parameters for the inner computation of the proximal operator with the Chambolle algorithm"""

    solver = FISTA(setting, init, proximal_pars=proximal_pars)
    stoprule = (
        # Method is slow, so need to use large number of iterations
        rules.CountIterations(max_iterations=100000) +
        rules.Discrepancy(
            setting.h_codomain.norm, data,
            noiselevel=setting.h_codomain.norm(noise),
            tau=1.1
        )
    )

    reco, reco_data = solver.run(stoprule)

    """from regpy.solvers.linear.admm import ADMM

    # Operator need to be linear 
    op = Volterra(grid, exponent=1)

    # Impulsive Noise
    sigma = 0.3*np.ones(grid.coords.shape[1])
    sigma[100:110] = 2

    exact_data = op(exact_solution)
    noise = sigma * op.domain.randn()
    data = exact_data + noise
    init = {
        "v1" : op.codomain.ones(),
        "v2" : op.domain.ones(),
        "p1" : op.codomain.ones(),
        "p2" : op.domain.ones(),
    }

    #construct the data misfit functional as combination of norm with Shifted operator.
    from regpy.operators import Identity
    setting = RegularizationSetting(
        op=op, 
        penalty=L2, 
        data_fid=HilbertNorm(h_space=L2) * (Identity(op.codomain) - data)
    )

    regpar = 0.01
    gamma = 1

    solver = ADMM(setting, init, regpar = regpar, gamma=gamma)
    stoprule = (
        rules.CountIterations(max_iterations=10) +
        rules.Discrepancy(
            setting.h_codomain.norm, data,
            noiselevel=setting.h_codomain.norm(noise),
            tau=1.1
        )
    )


    reco, reco_data = solver.run(stoprule)
    """



