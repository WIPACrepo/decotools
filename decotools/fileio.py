
from __future__ import division
import os
import glob
import time
import warnings
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd


def xml_to_dict(xmlfile):
    tree = ET.parse(xmlfile)  # Initiates the tree Ex: <user-agents>
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
    directory, image_file_basename = os.path.split(image_file)
    xml_file_basename = 'metadata-' + image_file_basename.replace('.png', '.xml')
    xml_file = os.path.join(directory, xml_file_basename)

    return xml_file


def get_metadata_dataframe(files):
    xml_data = []
    for f in files:
        try:
            xml_file = image_file_to_xml_file(f)
            xml_dict = xml_to_dict(xml_file)
            xml_df = pd.DataFrame.from_records(xml_dict, index=[f])
            xml_data.append(xml_df)
        except IOError:
            warnings.warn('No metadata file found for {}...'.format(f))
    df = pd.concat(xml_data)

    return df


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
                  phone_model=None, device_id=None, verbose=0):
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
    verbose : int (0 or 1)
        Option to have verbose output when getting files. Where 0 is
        least verbose, while 1 is the most verbose.

    Returns
    -------
    numpy.ndarray
        Numpy array containing files that match specified criteria

    '''

    # Validate include_min_bias and include_events:
    if not any([include_events, include_min_bias]):
        raise ValueError('At least one of include_events or include_min_bias '
                         'must be True.')
    # Validate user input for verbose parameter
    if not isinstance(verbose, int):
        raise ValueError('Expecting an int for verbose, '
                         'got {}'.format(type(verbose)))
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
        warnings.warn('No files for found for the specified date range')
        return file_list

    # If specified, filter out events/minimum bias images appropriately
    if include_events and include_min_bias:
        pass
    elif include_events and not include_min_bias:
        file_list = [f for f in file_list if 'minBias' not in f]
    elif not include_events and include_min_bias:
        file_list = [f for f in file_list if 'minBias' in f]

    # Construct DataFrame containing metadata to use for filtering
    df = get_metadata_dataframe(file_list)
    # Filter out image files based on user input
    df = filter_dataframe(df, metadata_key='Model', desired_values=phone_model)
    df = filter_dataframe(df, metadata_key='LensID', desired_values=device_id)

    # Optional verbose output
    if verbose:
        print('Found {} iOS image files'.format(df.shape[0]))
        n_devices = len(df.LensID.unique())
        print('Found {} unique devices'.format(n_devices))
        print('Models present:')
        models = df.Model.unique()
        for model in models:
            model_frac = np.sum(df.Model == model)/df.shape[0]
            print('\t {} [ {:0.1%} ]'.format(model, model_frac))

    file_array = df.index.values

    return file_array
