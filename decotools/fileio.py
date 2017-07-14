
from __future__ import division
import os
import glob
import time
import logging
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import multiprocessing as mp

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


def xmlfile_to_dict(xmlfile):
    '''Function to parse xml metadata file
    '''
    try:
        tree = ET.parse(xmlfile)  # Initiates the tree Ex: <user-agents>
    except IOError:
        raise NoMetadataFile('No metadata file found for {}...'.format(xmlfile))
    except ET.ParseError:
        if os.path.getsize(xmlfile) == 0:
            raise EmptyMetadataFile('The metadata file {} is empty'.format(xmlfile))
        else:
            raise UnknownMetadataIssue('Ran into an error parsing {}'.format(xmlfile))
    root = tree.getroot()  # Starts the root of the tree Ex: <user-agent>
    all_records = []  # This is our record list which we will convert into a dataframe
    headers = []  # Subchildren tags will be parsed and appended here
    for i, child in enumerate(root):  # Begin looping through our root tree
        record = []  # Place holder for our record
        # iterate through the subchildren to user-agent, Ex: ID, String,
        # Description.
        for subchild in child:
            # Extract the text and append it to our record list
            record.append(subchild.text)
            # Check the header list to see if the subchild tag <ID>,
            # <String>... is in our headers field. If not append it. This will
            # be used for our headers.
            if subchild.tag not in headers:
                headers.append(subchild.tag)
        all_records.append(record)  # Append this record to all_records.
    xml_parsed = np.asarray(all_records[0])
    keys = xml_parsed[::2]
    values = xml_parsed[1::2]
    xml_dict = dict(zip(keys, values))

    return xml_dict


def image_file_to_xml_file(image_file):
    '''Function to return the xml file path for a corresponding image file path
    '''
    directory, image_file_basename = os.path.split(image_file)
    xml_file_basename = 'metadata-' + image_file_basename.replace('.png', '.xml')
    xml_file = os.path.join(directory, xml_file_basename)

    return xml_file


def _get_metadata_dataframe(files):
    xml_data = []
    for f in files:
        xml_file = image_file_to_xml_file(f)
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

        xml_df = pd.DataFrame.from_records(xml_dict, index=[f])
        xml_data.append(xml_df)

    df = pd.concat(xml_data)

    return df


def _get_metadata_dataframe_multiprocess(files, n_jobs=1):

    pool = mp.Pool(processes=n_jobs)
    async_results = [pool.apply_async(_get_metadata_dataframe, args=(f,))
                for f in np.array_split(files, n_jobs)]
    dataframes = []
    for result in async_results:
        dataframes.append(result.get())

    df = pd.concat(dataframes)

    return df


def get_metadata_dataframe(files, n_jobs=1):
    if n_jobs == 1:
        return _get_metadata_dataframe(files)
    else:
        return _get_metadata_dataframe_multiprocess(files, n_jobs=n_jobs)


def validate_filter_input(user_input):

    if user_input is None:
        return None

    if isinstance(user_input, str):
        user_input = [user_input]
    if not isinstance(user_input, (list, tuple, set, np.ndarray)):
        raise TypeError('Input must be array-like, got {}'.format(type(user_input)))

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


def get_date_files(dates, data_dir):
    file_list = []
    for date in dates:
        date_files_pattern = os.path.join(data_dir, date, '*.png')
        file_list.extend( glob.glob(date_files_pattern) )
    return file_list


def get_iOS_files(start_date=None, end_date=None, data_dir='/net/deco/iOSdata',
                  include_events=True, include_min_bias=False,
                  phone_model=None, device_id=None, n_jobs=1, verbose=0):
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
    file_list = get_date_files(dates, data_dir)
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

    if not any([phone_model, device_id]):
        file_array = np.asarray(file_list)
    else:
        # Construct DataFrame containing metadata to use for filtering
        df = get_metadata_dataframe(file_list, n_jobs=n_jobs)
        # Get info about any issues with metadata files
        # This needs to be done before filtering!
        num_no_metadata_files = np.sum(df['metadata_exists'] == False)
        num_empty_metadata_files = np.sum(df['metadata_empty'] == True)
        num_parsing_metadata_errors = np.sum(df['metadata_parsing_error'] == True)
        num_image_files = df.shape[0]

        # Filter out image files based on user input
        df = filter_dataframe(df, metadata_key='Model', desired_values=phone_model)
        df = filter_dataframe(df, metadata_key='LensID', desired_values=device_id)

        # Log info about images
        num_images_str = 'Found {} iOS image files'.format(df.shape[0])
        n_devices = len(df['LensID'].unique())
        num_devices_str = 'Found {} unique devices'.format(n_devices)
        models_str = 'Models present:'
        models = df['Model'].unique()
        for model in models:
            model_frac = np.sum(df['Model'] == model)/df.shape[0]
            models_str += '\n\t\t{} [ {:0.1%} ]'.format(model, model_frac)
        missing_metadata_str = 'Number of missing metadata files: {}'.format(num_no_metadata_files)
        empty_metadata_str = 'Number of empty metadata files: {}'.format(num_empty_metadata_files)
        parsing_metadata_error_str = 'Number of metadata parsing errors: {}'.format(num_parsing_metadata_errors)
        num_image_files_str = 'Number of image files: {}'.format(num_image_files)
        logger.info('\n\t'.join(['',num_images_str, num_devices_str, models_str,
                num_image_files_str, missing_metadata_str, empty_metadata_str,
                parsing_metadata_error_str]))

        file_array = df.index.values

    return file_array
