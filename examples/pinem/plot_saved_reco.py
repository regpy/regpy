from examples.pinem.plotting import plot_exact_solution_data,plot_reco, plot_stats, init_plot_stats
from examples.pinem.setup import setup_simulated_g
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np

"""Script used to view saved pinem solutions.
"""

#Put name of file here.
filename = 'some_file.mat'


# If False, the simulated data corresponding to the displayed iterate are not shown to save time
show_reco_data = True
rec = loadmat(filename)
stats = {'ampl_err': rec['ampl_err'][0], 'phase_err': rec['phase_err'][0], \
       'complex_err': rec['complex_err'][0], 'residuals': rec['residuals'][0], \
        'nr_inner_steps': rec['nr_inner_steps'][0], 'N': rec['N'][0], 'Newton step' : rec['Newton step'][0]}
op, grid, exact_solution, g_map, mask_a_org, mask_p, opdata \
        = setup_simulated_g(g_is_complex=False,N=30,parallel=False)
data = loadmat('./NewtonCG_5e7/test_data.mat')['data'][0]
data_comp = op.codomain.split(data)
fig1,fig2 = plot_exact_solution_data(g_map,data_comp,plot_log_g = True)      
fig3, axs3 = init_plot_stats()
reco_ampl = rec['reco_amp']
reco_phase = rec['reco_phase']

if show_reco_data:
        reco_data = op(op.domain.join(np.log(reco_ampl),reco_phase))
        reco_data_comp = op.codomain.split(reco_data)
        plot_reco(fig1,fig2,reco_ampl,reco_phase,reco_data_comp,g_map,data_comp,stats['Newton step'][-1],mask_a = mask_a_org)
else:
        plot_reco(fig1,fig2,reco_ampl,reco_phase,None,g_map,data_comp,stats['Newton step'][-1],mask_a = mask_a_org)
plot_stats(axs3,stats,plot_inner_its = True)
plt.show(block=True)