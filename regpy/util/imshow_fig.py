import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb

####################### conversion routines for plotting complex-valued fields

def complex_to_rgb(z):
    HSV = np.dstack( (np.mod(np.angle(z)/(2.*np.pi),1), 1.0*np.ones(z.shape), np.abs(z)/np.max((np.abs(z[:]))), ))
    return hsv_to_rgb(HSV)

def complex_to_rgb_log(z):
    logdat = np.log(np.abs(z))
    minlog = np.min(logdat)
    maxlog = np.max(logdat)
    HSV = np.dstack( (np.mod(np.angle(z)/(2.*np.pi),1), 1.0*np.ones(z.shape), (logdat-minlog)/(maxlog-minlog) ))
    return hsv_to_rgb(HSV)

###################### ImShowFig

class ImShowFig:
    def __init__(self,nr_rows,nr_cols):
        self.nr_rows = nr_rows
        self.nr_cols = nr_cols
        self.fig, self.ax = plt.subplots(nr_rows, nr_cols, sharex=True, sharey=True)
        self.im = np.empty((nr_rows,nr_cols),dtype = object)
        self.cb = np.empty((nr_rows,nr_cols),dtype = object)

    def plot(self,plot_data):
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