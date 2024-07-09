import numpy as np
from numpy import ndarray, complex128, complex64, sum
from numpy.random import poisson
import skimage
import skimage.transform

from scipy.ndimage import center_of_mass, maximum_position
from scipy.interpolate import interpn
from datetime import datetime
import os
from glob import glob


__all__ = ['updatelineplot','numdir','gauss','disk','wavelengthscaling_uniformgrid','wavelengthscaling_axes','trim_zeros_andmiddle','roll_to_pos', 'generate_siemens', 'noisify_intensity']

def updatelineplot(fig,ax,lines,ydatas):
    for (line, ydata) in zip(lines, ydatas):
        line.set_ydata(ydata)
        line.set_xdata(np.arange(len(ydata)))
    ax.relim()
    ax.autoscale_view(True,True,True)
    fig.canvas.draw()
    fig.canvas.flush_events()

def numdir(path, foldername='reconst', giveallpaths=True, args=None):
    path = os.path.normpath(path)                
    if not os.path.exists(path):
        os.makedirs(path)
    newpath=os.path.join(path, '{}'.format(datetime.now().strftime('%Y_%m_%d')))
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    folderlist = glob(os.path.join(newpath, r'{}*'.format(foldername), r''))
    jj = 0
    for folder in folderlist:
        jj+=1
    outputpath = os.path.join(newpath, r'{}_{}'.format(foldername,jj), r'')
    os.makedirs(outputpath)
    if args is not None:
        argpaths= []
        for arg in args:
            argpath = os.path.join(outputpath, arg, r'')
            argpath = os.path.join(os.path.normpath(argpath), r'')
            os.makedirs(argpath)
            argpaths.append(argpath)
        if giveallpaths is True:
            return outputpath, *argpaths
        else:
            return outputpath
    else:
        return outputpath

def shift_array(arr: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """
    Alias for np.roll, shifts array in y and x directions (0,1 axes)
    from gsmj_tools
    """
    temp = np.roll(arr, (dy, dx), (0, 1))
    return temp


def roll_to_pos(arr: np.ndarray, y: int = 0, x: int = 0, pos: tuple = None, move_maximum: bool = False,
                by_abs_val: bool = True) -> np.ndarray:
    """
    Shift the center of mass of an array to the given position by cyclic permutation
    from gsmj_tools

    :param arr: 2d array, works best for well-centered feature with limited support
    :param y: position parameter
    :param x: position parameter for second dimension
    :param pos: tuple with the new position, overriding y,x values. should be used for higher-dimensional arrays
    :param move_maximum: if true, look only at max-value
    :param by_abs_val: take abs value for the determination of max-val or center-of-mass
    :return: array like original
    """
    if move_maximum:
        if by_abs_val or arr.dtype in [np.complex64, np.complex128]:
            old = np.floor(maximum_position(abs(arr)))
        else:
            old = np.floor(maximum_position(arr))
    else:
        if by_abs_val or arr.dtype in [np.complex64, np.complex128]:
            old = np.floor(center_of_mass(abs(arr)))
        else:
            old = np.floor(center_of_mass(arr))
    if pos is not None:  # dimension-independent method
        shifts = tuple([int(np.round(pos[i]-old[i])) for i in range(len(pos))])
        dims = tuple([i for i in range(len(pos))])
        temp = np.roll(arr, shift=shifts, axis=dims)
    else:  # old method
        temp = shift_array(arr, int(y - old[0]), int(x - old[1]))
    return temp

def noisify_intensity(truth: ndarray, total_counts: int, background_counts: float = 0.0) -> ndarray:
    """
    Given a perfect intensity map, simulate Poissonian noise
    from gsmj_tools

    :param truth: intensity  map, np.array
    :param total_counts: over the full array (results will vary within ~sqrt(total_counts))
    :param background_counts: average number of counts in the background
    :return: noisy array like truth
    """
    assert truth.dtype not in [complex128, complex64], 'Dtype should be real valued!'
    _truth = truth*total_counts/sum(truth)
    countmap = poisson(_truth)
    if background_counts != 0.0:
        countmap += poisson(lam=background_counts, size=_truth.shape)
    return countmap

def toRadians(theta):
    theta = theta * np.pi / 180.
    return theta

def toDegrees(theta):
    theta = theta * (np.pi / 180.) ** -1
    return theta

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def gauss(xx, yy, center, width, amplitude=1):
    x0, y0 = center
    dx, dy = width
    return amplitude * np.exp(-((xx - x0) ** 2 / dx ** 2 + (yy - y0) ** 2 / dy ** 2))

def disk(xx, yy, center, width, amplitude=1):
    x0, y0 = center
    dx, dy = width
    return np.where((xx - x0) ** 2 / dx ** 2 + (yy - y0) ** 2 / dy ** 2 < 1, amplitude, 0)

def trim_zeros(arr):
    """Returns a trimmed view of an n-D array excluding any outer
    regions which contain only zeros.
    taken from https://stackoverflow.com/questions/55917328/numpy-trim-zeros-in-2d-or-3d
    """
    slices = tuple(slice(idx.min(), idx.max() + 1) for idx in np.nonzero(arr))
    return arr[slices]

def trim_zeros_andmiddle(arr):
    """Returns a trimmed view of an n-D array excluding any outer
    regions which contain only zeros and the indices of the initial 
    position of the center. 
    Adapted from https://stackoverflow.com/questions/55917328/numpy-trim-zeros-in-2d-or-3d
    """
    slices = tuple(slice(idx.min(), idx.max() + 1) for idx in np.nonzero(arr))
    centers = tuple(
        int((sl.start+sl.stop)/2) if sl.stop-sl.start != arr.shape[k] else 
        None for k,sl in enumerate(slices))

    return arr[slices], centers

def trim_zeros_andslices(arr, increasewidthby = (0,0)):
    """Returns a trimmed view of an n-D array excluding any outer
    regions which contain only zeros and the indices of the initial 
    position of the center. 
    Adapted from https://stackoverflow.com/questions/55917328/numpy-trim-zeros-in-2d-or-3d
    """
    if len(increasewidthby) ==1:
        increasewidthby = (increasewidthby, increasewidthby)
    slices = tuple(slice(idx.min(), idx.max() + 1)  for idx in np.nonzero(arr))
    centers = tuple(
        int((sl.start+sl.stop)/2) if sl.stop-sl.start != arr.shape[k] else 
        None for k,sl in enumerate(slices))

    return arr[slices], slices

def slicewherenonzero(arr, increasewidthby = np.array(((0,0),(0,0),(0,0)))):
    """Returns a trimmed view of an n-D array excluding any outer
    regions which contain only zeros and the indices of the initial 
    position of the center. 
    Adapted from https://stackoverflow.com/questions/55917328/numpy-trim-zeros-in-2d-or-3d
    """
    if type(increasewidthby) is int:
        increasewidthby = np.array((increasewidthby, increasewidthby))
    if increasewidthby.ndim < arr.ndim:
        increasewidthby = np.broadcast_to(increasewidthby, (arr.ndim, 2))
    slices = []
    for kk, idx in enumerate(np.nonzero(arr)):
        minsli = idx.min()
        maxsli = idx.max()
        if minsli - increasewidthby[kk,0] >= 0:
            minsli = minsli - increasewidthby[kk,0]
        else:
            minsli = None
        if maxsli + 1 + increasewidthby[kk,1] < arr.shape[kk]:
            maxsli = maxsli + 1 + increasewidthby[kk,1]
        else:
            maxsli = None
        slices.append(slice(minsli, maxsli))
    slices = tuple(slices)
    return slices


        
    # slices = tuple(slice(idx.min(), idx.max() + 1)  for idx in np.nonzero(arr))
    # centers = tuple(
    #     int((sl.start+sl.stop)/2) if sl.stop-sl.start != arr.shape[k] else 
    #     None for k,sl in enumerate(slices))

    # return slices


def wavelengthscaling_axes(arrin, axes, scales):
    '''
    Function that returns an array of dimension 
    (len(axes[0]),len(axes[0]), len(scales)), in which scaled repetitions
    of arrin are stacked.
    '''
    xx,yy = np.meshgrid(*axes)
    arrout = np.array(
        [interpn((axes[0],axes[1]), arrin, (xx*scale,yy*scale), 
                 bounds_error=False, fill_value=0) for k, scale in enumerate(scales)])
    return arrout

def wavelengthscaling_uniformgrid(arrin, grid0, scales, imaxes = (-2,-1)):
    axes = [ax if ax.flags['C_CONTIGUOUS'] else ax.copy(order = 'C') for ll, ax
            in enumerate(grid0.axes)]
    arrout = np.array(
        [interpn((axes[imaxes[0]],axes[imaxes[1]]), arrin, 
        (grid0.coords[imaxes[0]][0]*scale,grid0.coords[imaxes[1]][0]*scale), 
        bounds_error=False, fill_value=0 ) for k, scale in enumerate(scales)])
    return arrout
