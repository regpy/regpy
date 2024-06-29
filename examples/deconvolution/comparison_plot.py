import matplotlib as mplib
import matplotlib.pyplot as plt
import numpy as np

def comparison_plot(grid,truth,reco,title_right='exact',title_left='reconstruction',residual=None):
    plt.rcParams.update({'font.size': 22})
    extent = [grid.axes[0][0],grid.axes[0][-1], grid.axes[1][0], grid.axes[1][-1]]
    maxval = np.max(truth[:]); minval = np.min(truth[:])
    mycmap = mplib.colormaps['hot']
    mycmap.set_over((0,0,1.,1.))  # use blue as access color for large values
    mycmap.set_under((0,1,0,1.))  # use green as access color for small values
    if not (residual is None):
        fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize = (22,16))
    else:
        fig, (ax1,ax2) = plt.subplots(1,2,figsize = (22,8))
    im1= ax1.imshow(reco.T,extent=extent,origin='lower',
                    vmin=1.05*minval-0.05*maxval, vmax =1.05*maxval-0.05*minval,
                    cmap=mycmap
                    )
    ax1.title.set_text(title_left)
    fig.colorbar(im1,extend='both')
    im2= ax2.imshow(truth.T,extent=extent, origin='lower',
                    vmin=1.05*minval-0.05*maxval, vmax =1.05*maxval-0.05*minval,
                    cmap=mycmap
                    )
    ax2.title.set_text(title_right)
    fig.colorbar(im2,extend='both',orientation='vertical')
    if not (residual is None):
        maxv = np.max(reco[:]-truth[:])
        im3 = ax3.imshow(truth.T-reco.T,extent=extent, origin='lower',vmin= -maxv,vmax=maxv,cmap='RdYlBu_r')
        ax3.title.set_text('reconstruction error')
        fig.colorbar(im3)

        maxv = np.max(residual[:])
        im4 = ax4.imshow(residual.T,extent=extent, origin='lower',vmin= -maxv,vmax=maxv,cmap='RdYlBu_r')
        ax4.title.set_text('data residual')
        fig.colorbar(im4)