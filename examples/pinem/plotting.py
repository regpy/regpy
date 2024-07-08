import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb

def plot_exact_solution_data(g_map,data_comp,plot_log_g = True):
    r""" Plots solution g_map and corresponding data

    Parameters
    ----------
    g_map : numpy.ndarray
        solution \(g\) to be plotted
    data_comp : numpy.ndarray
        simulated data
    plot_log_g : bool,optional  
        If True, \(\log(|g|)\) is plotted else \(|g|\) is plotted. Default: True 

    Returns 
        ----------
        fig1 : ImShowFig
            Figure with plots of g
        fig2 : ImShowFig
            Figure with plots of computed data
    """   
    fig1 = ImShowFig(3, 3)
    if plot_log_g:
        plot_data1 = [{'pos': (0, 0), 'data': np.log(np.abs(g_map.T)), 'title': 'log(|g|)'}]
    else:
        plot_data1 = [{'pos': (0, 0), 'data': np.abs(g_map.T), 'title': '|g|'}]                    
    plot_data1.append({'pos': (0, 1), 'data': complex_to_rgb(g_map.T), 'title': 'g with phase'})
    plot_data1.append({'pos': (0, 2), 'data': complex_to_rgb_log(g_map.T), 'title': 'log(g) with phase'})                  
    fig1.plot(plot_data1)

    nr_data = len(data_comp)
    fig2 = ImShowFig(3, nr_data)
    plot_data2 = [{'pos': (0, j), 'data': data_comp[j].T, 'title':'sim. data'}
                    for j in range(nr_data)]   
    if nr_data ==2:
        plot_data2[0]['title'] = 'sim. gain'
        plot_data2[1]['title'] = 'sim. loss'  
    fig2.plot(plot_data2)
    return fig1, fig2

def plot_reco(fig1,fig2,reco_amp,reco_phase,reco_data_comp,g_map,data_comp,newton_step,
              plot_log_g = True, mask_a = None
              ):
    r"""Plots reconstruction information

    Parameters
    ----------
    fig1 : ImshowFig
        Object managing the plotting of information about g.
    fig2 : ImshowFig
        Object managing the plotting of computed data
    reco_amp : numpy.ndarray
        reconstructed amplitude
    reco_phase : numpy.ndarray
        reconstructed phase
    reco_data_comp : np.ndarray
        reconstructed data complete
    g_map : numpy.ndarray
        g parameter of PINEM
    data_comp : numpy.ndarray
        complete original data
    newton_step : int 
        iteration step
    plot_log_g : bool
        If True the logarithm of the amplitude of g is plotted. Defaults to True.
    mask_a : numpy.ndarray
        Mask for domain. Defaults to None.
    """
    plot_data = []
    if plot_log_g:
        plot_data.append({'pos': (1, 0), 'data': np.log(reco_amp.T),
                    'title': 'log(|g_rec|) it.{}'.format(newton_step)})
    else:
        plot_data.append({'pos': (1, 0), 'data': reco_amp.T,
                    'title': '|g_rec| it.{}'.format(newton_step)})
    plot_data.append({'pos': (2, 0), 'data': reco_amp.T-np.abs(g_map.T),
                        'title': 'Error |g|-|g_rec|  it.{}'.format(newton_step)})
    plot_data.append({'pos': (1, 1), 'data': complex_to_rgb(reco_amp.T*np.exp(1j*reco_phase.T)),
                        'title': 'g_rec with phase it.{}'.format(newton_step)})
    plot_data.append({'pos': (1, 2), 'data': complex_to_rgb_log(reco_amp.T*np.exp(1j*reco_phase.T)),
                        'title': 'log(g_rec) it.{}'.format(newton_step)})

    if mask_a is None:
        plot_data.append({'pos': (2, 1), 'data': np.abs(reco_amp.T*np.exp(1j*reco_phase.T)-g_map.T),
                        'title': 'error |g_rec-g| it.{}'.format(newton_step)})
    else:
        plot_data.append({'pos': (2, 1), 'data': (1.-mask_a.T) * np.abs(reco_amp.T*np.exp(1j*reco_phase.T)-g_map.T),
                        'title': 'ext. error |g_rec-g| it.{}'.format(newton_step)})
        plot_data.append({'pos': (2, 2), 'data': mask_a.T.astype(float) * np.abs(reco_amp.T*np.exp(1j*reco_phase.T)-g_map.T),
                        'title': 'int. error |g_rec-g| it.{}'.format(newton_step)})
    fig1.plot(plot_data)

    if not reco_data_comp is None:
        nr_data = len(reco_data_comp)
        plot_data = [{'pos': (1, j), 'data': reco_data_comp[j].T,
                        'title':'rec. data it.{}'.format(newton_step)}
                    for j in range(nr_data)]
        for j in range(nr_data):
            plot_data.append({'pos': (2, j), 'data': reco_data_comp[j].T-data_comp[j].T,
                            'title': 'diff'})
        fig2.plot(plot_data)

def init_plot_stats():
    r"""Initializes plot of convergence statistics
    Returns 
    ----------
    matplotlib.figure.Figure
        Figure with one column and three rows. One for errors, one for residuals and one for number of inner CG-steps
    """
    return plt.subplots(3, 1, sharex=False, sharey=False)

def plot_stats(axs3,stats,plot_inner_its=True):
    r"""Plots convergence statistics

    Parameters
    ----------
    axs3 : tuple, list or numpy.ndarray of matplotlib.axes._axes.Axes
        Contains the axes objects used for plotting. Should contain at least 2 and 
        at least 3 if plot_inner_its is set to True.
    stats : dict
        Dictonary containing convergence statistics. 
        Required keys: Newton step, ampl_err, phase_err, complex_err, residuals, 
        nr_inner_steps if plot_inner_its is set to True.
    plot_inner_its : bool
        If True the number of inner iterations of the solver is plotted. Defaults to True.
    """
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

####################### conversion routines for plotting complex-valued fields

def complex_to_rgb(z):
    r"""Converts array of complex numbers into array of RGB color values for plotting. The hue corresponds to the argument.
    The brighntess corresponds to the absolut value.  

    Parameters
    ----------
    z : numpy.ndarray
        array of complex numbers

    Returns 
    ----------
    numpy.ndarray
        Array that contains three values for each value in z containing the RGB representation of this value.
    """  
    HSV = np.dstack( (np.mod(np.angle(z)/(2.*np.pi),1), 1.0*np.ones(z.shape), np.abs(z)/np.max((np.abs(z[:]))), ))
    return hsv_to_rgb(HSV)

def complex_to_rgb_log(z):
    r"""Converts array of complex numbers into array of RGB color values for plotting. The hue corresponds to the argument.
    The brighntess corresponds to the logarithm of the absolut value.  

    Parameters
    ----------
    z : numpy.ndarray
        array of complex numbers

    Returns 
    ----------
    numpy.ndarray
        Array that contains three values for each value in z containing the RGB representation of this value.
    """  
    logdat = np.log(np.abs(z))
    minlog = np.min(logdat)
    maxlog = np.max(logdat)
    HSV = np.dstack( (np.mod(np.angle(z)/(2.*np.pi),1), 1.0*np.ones(z.shape), (logdat-minlog)/(maxlog-minlog) ))
    return hsv_to_rgb(HSV)

class ImShowFig:
    r""" 
    Class used for plotting intermediate results of image producing inversion iterations
    
    Parameters
    ----------
    nr_rows : int
        number of rows in figure
    nr_rows : int
        number of columns in figure

    """
    def __init__(self,nr_rows,nr_cols):
        self.nr_rows = nr_rows
        self.nr_cols = nr_cols
        self.fig, self.ax = plt.subplots(nr_rows, nr_cols, sharex=True, sharey=True)
        """information about pyplot subplots used for plotting"""
        self.im = np.empty((nr_rows,nr_cols),dtype = object)
        """numpy array of images to be plotted"""
        self.cb = np.empty((nr_rows,nr_cols),dtype = object)
        """numpy array of color bars used as legends"""

    def plot(self,plot_data):
        r"""Converts array of complex numbers into array of RGB color values for plotting. The hue corresponds to the argument.
        The brighntess corresponds to the logarithm of the absolut value.  

        Parameters
        ----------
        plot_data : iterable of dict
            The data to be plotted. Each dictionary has to contain a tuple on length 2 at 'pos' indicating the position
            and an image at 'data' which is plotted using imshow. An iterable at 'kwargs' can be used for further imshow parameters. 
        """  
        for datum in plot_data:
            row,col = datum['pos']
            assert row <= self.nr_rows
            assert col <= self.nr_cols
            if 'kwargs' in datum:
                self.im[row,col] = self.ax[row,col].imshow(datum['data'],**datum['kwargs'])
            else:
                self.im[row,col] = self.ax[row,col].imshow(datum['data'])
            if 'title' in datum:
                self.ax[row,col].set_title(datum['title'])
            if self.cb[row,col]:
                self.cb[row,col].remove()
            self.cb[row,col]= self.fig.colorbar(self.im[row,col], ax=self.ax[row,col])
        plt.pause(1e-4)