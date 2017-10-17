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


def get_rgb_hists(files, cumulative=False, n_jobs=1):
    '''Calculates histograms of the pixel RGB sum distributions 

    Parameters
    ----------
    files : str, sequence
        Image file path (or sequence of file paths) to be analyzed.
    cumulative : bool, optional
        Option to calculate cumulative histograms. Histogrammed quantities 
        will be N pixels >= threshold.
    n_jobs : int, optional
        The number of jobs to run in parallel (default is 1).

    Returns
    -------
    hists : pandas.DataFrame
        Dataframe containing histograms of pixel RGB sums
    '''

    # Create integer bins spanning RGB sum values
    # np.histogram bins inclusively on the lower bin edge,
    # i.e. values within [lower_edge, upper_edge)
    rgb_max = 256*3
    bins = np.linspace(0, rgb_max+1, rgb_max+2)

    # Load images 
    images = [delayed(get_image_array)(f, rgb_sum=True) for f in files]
    # Bin images
    histos = [delayed(np.histogram)(image.flatten(), bins=bins)[0] for image in images]
    # Crate dataframe
    rgb_hists = delayed(pd.DataFrame.from_records)(histos)

    with ProgressBar() as bar:
        get = dask.get if n_jobs == 1 else multiprocessing.get
        rgb_hists = rgb_hists.compute(get=get, num_workers=n_jobs)

    return rgb_hists

if __name__ == '__main__':
   
    import decotools as dt

    device_ids = {'Matt'    : 'BBAFF84F-7BC3-4774-A34F-8DD71C9E0B8F',
                  'Justin'  : 'F216114B-8710-4790-A05D-D645C9C79C27',
                  'Miles'   : 'D8D8E48D-7D3F-4693-A927-A402CF127D25'
                 } 
    device_id = device_ids['Matt']

    files = dt.get_iOS_files(verbose=1, n_jobs=20, device_id=device_id, return_metadata=True,
                             include_events=True, start_date='08.23.2017', end_date='08.31.2017')

    histos = get_rgb_hists(files.image_file.values, n_jobs=20)
    print histos
