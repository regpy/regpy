import numpy as np
import matplotlib.pyplot as plt
from regpy.util.ImShowFig import ImShowFig, complex_to_rgb, complex_to_rgb_log

def plot_exactSolution_data(g_map,data_comp,using_gabs_measurement = False,plot_log_g = True):
    fig1 = ImShowFig(3, 3)
    if plot_log_g:
        plotdata1 = [{'pos': (0, 0), 'data': np.log(np.abs(g_map.T)), 'title': 'log(|g|)'}]
    else:
        plotdata1 = [{'pos': (0, 0), 'data': np.abs(g_map.T), 'title': '|g|'}]                    
    plotdata1.append({'pos': (0, 1), 'data': complex_to_rgb(g_map.T), 'title': 'g with phase'})
    plotdata1.append({'pos': (0, 2), 'data': complex_to_rgb_log(g_map.T), 'title': 'log(g) with phase'})                  
    fig1.plot(plotdata1)

    nr_data = len(data_comp)
    fig2 = ImShowFig(3, nr_data)
    plot_data2 = [{'pos': (0, j), 'data': data_comp[j].T, 'title':'sim. data'}
                    for j in range(nr_data)]
    if using_gabs_measurement and nr_data ==3:
        plot_data2[0]['title'] = 'sim. ampl'
        plot_data2[1]['title'] = 'sim. gain'
        plot_data2[2]['title'] = 'sim. loss'    
    if not using_gabs_measurement and nr_data ==2:
        plot_data2[0]['title'] = 'sim. gain'
        plot_data2[1]['title'] = 'sim. loss'  
    fig2.plot(plot_data2)
    return fig1, fig2

def plot_reco(fig1,fig2,reco_amp,reco_phase,reco_data_comp,g_map,data_comp,newton_step,
              plot_log_g = True, mask_a = None
              ):
        plotdata = []
        if plot_log_g:
            plotdata.append({'pos': (1, 0), 'data': np.log(reco_amp.T),
                        'title': 'log(|g_rec|) it.{}'.format(newton_step)})
        else:
            plotdata.append({'pos': (1, 0), 'data': reco_amp.T,
                        'title': '|g_rec| it.{}'.format(newton_step)})
        plotdata.append({'pos': (2, 0), 'data': reco_amp.T-np.abs(g_map.T),
                            'title': 'Error |g|-|g_rec|  it.{}'.format(newton_step)})
        plotdata.append({'pos': (1, 1), 'data': complex_to_rgb(reco_amp.T*np.exp(1j*reco_phase.T)),
                            'title': 'g_rec with phase it.{}'.format(newton_step)})
        plotdata.append({'pos': (1, 2), 'data': complex_to_rgb_log(reco_amp.T*np.exp(1j*reco_phase.T)),
                            'title': 'log(g_rec) it.{}'.format(newton_step)})

        if mask_a is None:
            plotdata.append({'pos': (2, 1), 'data': np.abs(reco_amp.T*np.exp(1j*reco_phase.T)-g_map.T),
                            'title': 'error |g_rec-g| it.{}'.format(newton_step)})
        else:
            plotdata.append({'pos': (2, 1), 'data': (1.-mask_a.T) * np.abs(reco_amp.T*np.exp(1j*reco_phase.T)-g_map.T),
                            'title': 'ext. error |g_rec-g| it.{}'.format(newton_step)})
            plotdata.append({'pos': (2, 2), 'data': mask_a.T.astype(float) * np.abs(reco_amp.T*np.exp(1j*reco_phase.T)-g_map.T),
                            'title': 'int. error |g_rec-g| it.{}'.format(newton_step)})
        #plotdata.append({'pos': (2, 1), 'data': np.abs(np.exp(1j*reco_phase.T)-(g_map/(np.abs(g_map)+1e-16)).T),
        #                    'title': '|g_rec/|g_rec|-g/|g|| it.{}'.format(newton_step)})
        fig1.plot(plotdata)

        if not reco_data_comp is None:
            nr_data = len(reco_data_comp)
            plotdata = [{'pos': (1, j), 'data': reco_data_comp[j].T,
                            'title':'rec. data it.{}'.format(newton_step)}
                        for j in range(nr_data)]
            for j in range(nr_data):
                plotdata.append({'pos': (2, j), 'data': reco_data_comp[j].T-data_comp[j].T,
                                'title': 'diff'})
            fig2.plot(plotdata)

def init_plot_stats():
    return plt.subplots(3, 1, sharex=False, sharey=False)

def plot_stats(axs3,stats,plot_inner_its=True):
    axs3[0].cla()        
    axs3[0].plot(stats['Newton step'],stats['ampl_err']/stats['ampl_err'][0], label='amplitude error')
    axs3[0].plot(stats['Newton step'],stats['phase_err']/stats['phase_err'][0], label='phase error')
    axs3[0].plot(stats['Newton step'],stats['complex_err']/stats['complex_err'][0], label='complex error')
    axs3[0].legend()
    axs3[1].cla()
    axs3[1].semilogy(stats['Newton step'],stats['residuals'], label='residuals')
    axs3[1].legend()
    if plot_inner_its:
        axs3[2].cla()
        axs3[2].plot(stats['Newton step'],stats['nr_inner_steps'], label='number of inner CG steps')
        axs3[2].legend()
    plt.show(block=False)
    plt.pause(1e-4)