from __future__ import division
import numpy as np
import pandas as pd
from skimage import io, measure
from PIL import Image
from collections import Counter, Iterable
import dask
from dask import delayed, multiprocessing
from dask.diagnostics import ProgressBar
from blob_extraction import get_image_array


def _get_image_intensity(image_file, rgb_sum=False):
    '''Function to calculate intensity metrics

    Parameters
    ----------
    image_file : str
        Image file to be analyzed.
    rgb_sum : bool, optional
        Option to use simple RGB sum for grayscale conversion (default is to
        use weighted RGB sum).

    Returns
    -------
    intensity_dict : dict
        Dictionary with intensity metrics

    '''
    image = get_image_array(image_file, rgb_sum=rgb_sum)
    intensity_dict = {'mean': image.mean(), 'max': image.max()}
    for percentile in [16, 50, 84]:
        key = 'percentile_{}'.format(percentile)
        intensity_dict[key] = np.percentile(image, percentile)

    return intensity_dict


def get_intensity_metrics(files, rgb_sum=False, n_jobs=1):
    '''Calculates various metrics related to the image intensity

    Parameters
    ----------
    files : str, sequence
        Image file path (or sequence of file paths) to be analyzed.
    rgb_sum : bool, optional
        Option to use simple RGB sum for grayscale conversion (default is to
        use weighted RGB sum).
    n_jobs : int, optional
        The number of jobs to run in parallel (default is 1).

    Returns
    -------
    image_intensities : pandas.DataFrame
        DataFrame with intensity metrics

    '''
    if isinstance(files, str):
        files = [files]

    image_intensities = [delayed(_get_image_intensity)(f) for f in files]
    image_intensities = delayed(pd.DataFrame.from_records)(image_intensities)

    with ProgressBar() as bar:
        get = dask.get if n_jobs == 1 else multiprocessing.get
        image_intensities = image_intensities.compute(get=get,
                                                      num_workers=n_jobs)

    return image_intensities


def _get_cumulative_hist(hist):
    '''Calculates cumulative histogram from differential histogram

    Parameters
    ----------
    hist : np.array
        Differential histogram of an image's RGB sum values

    Returns
    -------
    cumulative_hist : np.array
        Cumulative histogram of an image's RGB sum values.
        Each bin contains N_pixels > RGB sum value.
        
    '''
    cumulative_sum = np.cumsum(hist)
    npixels = np.sum(hist)
    cdf = cumulative_sum / npixels
    cumulative_hist = (1.-cdf) * npixels
    return cumulative_hist


def get_rgb_hists(files, cumulative=False, n_jobs=1):
    '''Calculates histograms of the pixel RGB sum distributions 

    Parameters
    ----------
    files : str, sequence
        Image file path (or sequence of file paths) to be analyzed.
    cumulative : bool, optional
        Option to calculate cumulative histograms. Histogrammed quantities 
        will be N pixels > threshold.
    n_jobs : int, optional
        The number of jobs to run in parallel (default is 1).

    Returns
    -------
    hists : pandas.DataFrame
        Dataframe containing histograms of pixel RGB sums.
        Each row of the Dataframe corresponds to a single image
        and each column corresponds to an RGB sum value. 
    '''
    if isinstance(files, str):
        files = [files]

    # Create integer bins spanning RGB sum values
    # np.histogram bins inclusively on the lower bin edge,
    # i.e. values within [lower_edge, upper_edge)
    rgb_max = 256*3
    bins = np.linspace(0, rgb_max+1, rgb_max+2)

    # Load images 
    images = [delayed(get_image_array)(f, rgb_sum=True) for f in files]
    # Bin pixel intensities
    hists = [delayed(np.histogram)(image.flatten(), bins=bins)[0] for image in images]
    if cumulative:
        hists = [delayed(_get_cumulative_hist)(hist) for hist in hists]
    # Create dataframe
    rgb_hists = delayed(pd.DataFrame.from_records)(hists)

    with ProgressBar() as bar:
        get = dask.get if n_jobs == 1 else multiprocessing.get
        rgb_hists = rgb_hists.compute(get=get, num_workers=n_jobs)

    return rgb_hists
