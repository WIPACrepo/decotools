from __future__ import division
import os
import glob
import time
import logging
import plistlib
from xml.parsers.expat import ExpatError
import numpy as np
import pandas as pd
import dask
from dask import delayed, multiprocessing
from dask.diagnostics import ProgressBar

# Specify logging settings
logging.basicConfig(
    format='%(levelname)s: %(name)s - %(message)s')
logging_level_dict = {0: logging.ERROR, 1: logging.INFO, 2: logging.DEBUG}
logger = logging.getLogger(__name__)


# Define custom exceptions related to parsing metadata files
class NoMetadataFile(Exception):
    '''Raised when no metadata file exists for an image file'''
    pass

class EmptyMetadataFile(Exception):
    '''Raised when a metadata files is empty'''
    pass

class UnknownMetadataIssue(Exception):
    '''Raised when there is an unknown issue parsing a metadata file'''
    pass


def xmlfile_to_dict(xml_file):
    '''Returns a dictionary with metadata information stored in xml_file
    '''
    try:
        xml_dict = plistlib.readPlist(xml_file)
    except IOError:
        raise NoMetadataFile('No metadata file found for {}...'.format(xml_file))
    except ExpatError:
        if os.path.getsize(xml_file) == 0:
            raise EmptyMetadataFile('The metadata file {} is empty'.format(xml_file))
        else:
            raise UnknownMetadataIssue('Ran into an error parsing {}'.format(xml_file))

    return xml_dict


def image_file_to_xml_file(image_file):
    '''Function to return the xml file path for a corresponding image file path
    '''
    directory, image_file_basename = os.path.split(image_file)
    xml_file_basename = 'metadata-' + image_file_basename.replace('.png', '.xml')
    xml_file = os.path.join(directory, xml_file_basename)
    if os.path.exists(xml_file):
        return xml_file
    else:
        # if xml_file does not contain 'metadata-' in the name, try without it    
        xml_file_basename = image_file_basename.replace('.png','.xml')
        xml_file = os.path.join(directory,xml_file_basename)
        return xml_file


def get_id_from_filename(image_file):
    file_basename = os.path.basename(image_file)
    phone_id = file_basename.split('_')[0]

    return phone_id[1:]


def get_time_from_filename(image_file):
    file_basename = os.path.basename(image_file)
    file_name, ext = os.path.splitext(file_basename)
    date = file_name.split('_')[1]
    time = file_name.split('_')[2]

    return pd.to_datetime(date + ' ' + time, utc=True)


def get_metadata_dataframe_batches(files):

    # If files is empty, then just return an empty DataFrame
    if len(files) == 0:
        return pd.DataFrame()

    xml_data = []
    for idx, image_file in enumerate(files):
        xml_file = image_file_to_xml_file(image_file)
        try:
            xml_dict = xmlfile_to_dict(xml_file)
            xml_dict['metadata_exists'] = True
            xml_dict['metadata_empty'] = False
            xml_dict['metadata_parsing_error'] = False
        except NoMetadataFile as exception:
            logger.debug(exception)
            xml_dict = {'metadata_exists': False}
        except EmptyMetadataFile as exception:
            logger.debug(exception)
            xml_dict = {'metadata_empty': True}
        except UnknownMetadataIssue as exception:
            logger.debug(exception)
            xml_dict = {'metadata_parsing_error': True}

        xml_dict['image_file'] = image_file
        xml_dict['min_bias'] = True if 'minBias' in image_file else False
        # Time and device ID info can be extracted from the image file name
        # (so we have some info if there is an issue w/metadata file)
        xml_dict['time'] = get_time_from_filename(image_file)
        xml_dict['id'] = get_id_from_filename(image_file)

        xml_df = pd.DataFrame.from_records(xml_dict, index=[idx])
        xml_data.append(xml_df)

    df = pd.concat(xml_data, ignore_index=True)

    return df


def get_metadata_dataframe(files, n_jobs=1):

    # Split files into batches to be processed (potentially) in parallel
    batches = np.array_split(files, min(100, len(files)))
    df_list = [delayed(get_metadata_dataframe_batches)(batch) for batch in batches]
    df_merged = delayed(pd.concat)(df_list, ignore_index=True)
    print('Extracting metadata information:')
    with ProgressBar():
        if n_jobs == 1:
            df = df_merged.compute(get=dask.get)
        else:
            df = df_merged.compute(get=multiprocessing.get,
                                   num_workers=n_jobs)

    return df


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


def get_date_files(dates, data_dir, image_ext='png'):
    file_list = []
    for date in dates:
        date_files_pattern = os.path.join(data_dir, date, '*.{}'.format(image_ext))
        file_list.extend( glob.glob(date_files_pattern) )
    return file_list


def get_iOS_files(start_date=None, end_date=None, data_dir='/net/deco/iOSdata',
                  include_events=True, include_min_bias=False,
                  phone_model=None, device_id=None, return_metadata=False,
                  n_jobs=1, verbose=0):
    '''Function to retrieve deco iOS image files

    Parameters
    ----------
    start_date : str, optional
        Starting date for the iOS files to retrieve. Use any common
        date format (e.g. '2017-01-01', '20170101', 'Jan 1, 2017', etc).
        Default starting date is '2016.01.01'.
    end_date : str, optional
        Ending date for the iOS files to retrieve. Use any common
        date format (e.g. '2017-01-01', '20170101', 'Jan 1, 2017', etc).
        Default is the current date.
    data_dir : str, optional
        Base directory to search for iOS image files.
    include_events : bool, optional
        Option to include images files flagged as events. Default is True.
    include_min_bias : bool, optional
        Option to include minimum bias image files. Default is False.
    phone_model : str or array-like, optional
        Option to specify which phone models you would like to look at.
        Can be either a string, e.g. 'iPhone 7', or a list of models,
        e.g. ['iPhone 5', 'iPhone 5s']. Default is to include all
        phone models.
    device_id : str or array-like, optional
        Option to specify which devices you want to look at. Can
        either be a string, e.g. 'EFD5764E-7209-4579-B0A8-EAF80C950147', or
        a list of device IDs, e.g. ['EFD5764E-7209-4579-B0A8-EAF80C950147',
        'F216114B-8710-4790-A05D-D645C9C79C27']. Default is to include all
        device IDs.
    return_metadata : boolean, optional
        Return a DataFrame with metadata information for each image file
        (default is False).
    n_jobs : int, optional
        The number of jobs to run in parallel (default is 1).
    verbose : int {0, 1, or 2}
        Option to have verbose output when getting files. Where 0 is
        least verbose, while 2 is the most verbose.

    Returns
    -------
    numpy.ndarray
        Numpy array containing files that match specified criteria

    '''

    # Validate that data_dir exists:
    if os.path.isdir(data_dir) == False:
        raise IOError('data_dir, {}, cannot be located.'.format(data_dir))
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
    phone_model = validate_filter_input(phone_model)
    device_id = validate_filter_input(device_id)

    # If no end_date specified, set as today's date
    if not end_date:
        end_date = time.strftime('%Y.%m.%d')
    # If no start_date specified, set as some early date???
    if not start_date:
        start_date = '2016.01.01'

    # Get list of dates between start_date and end_date
    try:
        dates = pd.date_range(start_date, end_date)
        dates = dates.strftime('%Y.%m.%d')
    except ValueError:
        raise ValueError('Invalid start_date or end_date entered')

    # Build up list of all image files within the start_date to end_date range
    file_list = get_date_files(dates, data_dir, image_ext='png')
    if len(file_list) == 0:
        logger.warning('No files for found for the specified date range')
        return file_list

    # If specified, filter out events/minimum bias images appropriately
    if include_events and include_min_bias:
        pass
    elif include_events and not include_min_bias:
        file_list = [f for f in file_list if 'minBias' not in f]
    elif not include_events and include_min_bias:
        file_list = [f for f in file_list if 'minBias' in f]
    if len(file_list) == 0:
        logger.warning('No files remaining after event vs. minimum bias filtering')
        return file_list

    if not any([phone_model, device_id, return_metadata]):
        file_array = np.asarray(file_list)
    else:
        # Construct DataFrame containing metadata to use for filtering
        df = get_metadata_dataframe(file_list, n_jobs=n_jobs)
        num_image_files = df.shape[0]

        # Filter out image files based on user input
        df = (df.pipe(filter_dataframe, metadata_key='Model', desired_values=phone_model)
                .pipe(filter_dataframe, metadata_key='LensID', desired_values=device_id)
                .reset_index(drop=True)
             )

        # Log info about images
        num_images_str = 'Found {} image files'.format(df.shape[0])
        n_devices = len(df['id'].unique())
        num_devices_str = 'Found {} unique devices'.format(n_devices)
        models_str = 'Models present:'
        models = df['Model'].unique()
        for model in models:
            model_frac = np.sum(df['Model'] == model)/df.shape[0]
            models_str += '\n\t\t{} [ {:0.1%} ]'.format(model, model_frac)
        logger.info('\n\t'.join(['',num_images_str, num_devices_str, models_str]))

        if return_metadata:
            return df

        file_array = df['image_file'].values

    return file_array
