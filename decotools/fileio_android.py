from __future__ import division
import os
import time
import logging
import numpy as np
import pandas as pd

# Specify logging settings
logging.basicConfig(
    format='%(levelname)s: %(name)s - %(message)s')
logging_level_dict = {0: logging.ERROR, 1: logging.INFO, 2: logging.DEBUG}
logger = logging.getLogger(__name__)


def validate_filter_input(user_input):

    if user_input is None:
        return None

    if isinstance(user_input, str):
        user_input = [user_input]
    if not isinstance(user_input, (list, tuple, set, np.ndarray)):
        raise TypeError('Input must be array-like, '
                        'got {}'.format(type(user_input)))

    return user_input


def filter_dataframe(df, metadata_key, desired_values=None):

    if not isinstance(df, pd.DataFrame):
        raise TypeError('df must be a pandas.DataFrame, '
                        'got {}'.format(type(df)))
    if desired_values and not isinstance(desired_values, (list, tuple, set, np.ndarray)):
        raise TypeError('desired_values must be array-like')

    if desired_values is not None:
        return df[ df[metadata_key].isin(desired_values) ]
    else:
        return df


@np.vectorize
def db_path_to_image_file(path, data_dir='/net/deco/deco_data'):
    '''Function to convert paths stored in andriod db to image file paths

    Parameters
    ----------
    path : str, array-like
        Path from android database.
    data_dir : str, optional
        Path to directory containing images (default is /net/deco/deco_data).

    Returns
    -------
    image_files : str, array-like
        Image file paths.

    '''
    date, basename = os.path.split(path)
    image_id = basename.split('_')[0]
    image_file = os.path.join(data_dir, date + 'Z', image_id+'.jpg')

    return image_file


@np.vectorize
def db_path_to_date(path, data_dir='/net/deco/deco_data'):
    '''Function to extract date from paths stored in andriod db

    Parameters
    ----------
    path : str, array-like
        Path from android database.
    data_dir : str, optional
        Path to directory containing images (default is /net/deco/deco_data).

    Returns
    -------
    dates : str, array-like
        Dates corresponding to path.

    '''
    date, basename = os.path.split(path)
    return pd.to_datetime(date, utc=True)


def get_android_files(start_date=None, end_date=None, data_dir='/net/deco/deco_data',
                      db_file='/net/deco/db_hourly_safe.csv',
                      include_events=True, include_min_bias=False,
                      device_id=None, return_metadata=False, verbose=0):
    '''Function to retrieve deco android image files

    Parameters
    ----------
    start_date : str, optional
        Starting date for the files to retrieve. Use any common
        date format (e.g. '2017-01-01', '20170101', 'Jan 1, 2017', etc).
        Default starting date is '2010.01.01'.
    end_date : str, optional
        Ending date for the files to retrieve. Use any common
        date format (e.g. '2017-01-01', '20170101', 'Jan 1, 2017', etc).
        Default is the current date.
    data_dir : str, optional
        Base directory to retrieve android image files.
    db_file : str, optional
        File path to android database.
    include_events : bool, optional
        Option to include images files flagged as events. Default is True.
    include_min_bias : bool, optional
        Option to include minimum bias image files. Default is False.
    device_id : str or array-like, optional
        Option to specify which devices you want to look at. Can
        either be a string, e.g. 'DECO-00000000-450a-7561-433f-0516209b4922',
        or a list of device IDs, e.g. ['DECO-00000000-450a-7561-433f-0516209b4922',
        'DECO-ffffffff-bd6f-e5fb-842b-56b10033c587']. Default is to include all
        device IDs.
    return_metadata : boolean, optional
        Return a DataFrame with metadata information for each image file
        (default is False).
    verbose : int {0, 1, or 2}
        Option to have verbose output when getting files. Where 0 is
        least verbose, while 2 is the most verbose.

    Returns
    -------
    numpy.ndarray
        Numpy array containing files that match specified criteria

    '''
    # Check that data_dir and db_file exist:
    if not os.path.isdir(data_dir):
        raise IOError('data_dir, {}, does not exist.'.format(data_dir))
    if not os.path.exists(db_file):
        raise IOError('db_file, {}, does not exist.'.format(db_file))
    # Validate include_min_bias and include_events:
    if not any([include_events, include_min_bias]):
        raise ValueError('At least one of include_events or include_min_bias '
                         'must be True.')
    # Validate user input for verbose parameter
    if not isinstance(verbose, int):
        raise ValueError('Expecting an int for verbose, '
                         'got {}'.format(type(verbose)))
    logger.setLevel(logging_level_dict[verbose])
    # Validate user input for filtering values
    device_id = validate_filter_input(device_id)

    # If no end_date specified, set as today's date
    if not end_date:
        end_date = time.strftime('%Y.%m.%d')
    # If no start_date specified, set as some early date
    if not start_date:
        start_date = '2010.01.01'
    # Get list of dates between start_date and end_date
    try:
        dates = pd.date_range(start_date, end_date)
        dates = dates.strftime('%Y.%m.%d')
    except ValueError:
        raise ValueError('Invalid start_date or end_date entered')

    # Read in hourly dump of android database
    df = pd.read_csv(db_file, parse_dates=['acquisition_time'])
    # Drop nan paths
    df = df.dropna(axis=0, how='any', subset=['path'])
    # Get date from path (acquisition_time can be unreliable)
    df['time'] = df['path'].apply(db_path_to_date)

    # Filter out all image files not within the start_date to end_date range
    date_mask = (df.time >= start_date) & (df.time <= end_date)
    df = df[date_mask]
    if len(df) == 0:
        logger.warning('No files for found for the specified date range')
        return df

    # If specified, filter out events/minimum bias images appropriately
    min_bias_mask = df['minbias'].astype(bool)
    if include_events and include_min_bias:
        pass
    elif include_events and not include_min_bias:
        df = df[~min_bias_mask].reset_index(drop=True)
    elif not include_events and include_min_bias:
        df = df[min_bias_mask].reset_index(drop=True)
    if len(df) == 0:
        logger.warning('No files remaining after event vs. minimum bias filtering')
        return file_list

    # Filter out image files based on user input
    df = (df.pipe(filter_dataframe, metadata_key='sensor_id', desired_values=device_id)
            .reset_index(drop=True)
         )

    # Construct proper image file paths from the db path
    df['image_file'] = df['path'].apply(db_path_to_image_file)

    # Log info about images
    num_images_str = 'Found {} image files'.format(df.shape[0])
    n_devices = len(df['sensor_id'].unique())
    num_devices_str = 'Found {} unique devices'.format(n_devices)
    logger.info('\n\t'.join(['',num_images_str, num_devices_str]))

    if return_metadata:
        return df

    file_array = df['image_file'].values

    return file_array
