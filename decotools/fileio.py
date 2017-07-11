
import os
import glob
import time
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


def get_phone_model(image_file):
    xml_file = image_file_to_xml_file(image_file)
    xml_dict = xml_to_dict(xml_file)

    return xml_dict['Model']


def get_id(image_file):
    xml_file = image_file_to_xml_file(image_file)
    xml_dict = xml_to_dict(xml_file)

    return xml_dict['LensID']


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
    phone_model : str or list, optional
        Option to specify which phone models you would like to look at.
        Can be either a string, e.g. 'iPhone 7', or a list of models,
        e.g. ['iPhone 5', 'iPhone 5s']. Default is to include all
        phone models.
    device_id : str or list, optional
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
   
    # Validate that data_dir exists:
    if os.path.isdir(data_dir)==False:
        raise IOError('data_dir, {}, cannot be located.'.format(data_dir))
    # Validate include_min_bias and include_events:
    if not any([include_events, include_min_bias]):
        raise ValueError('At least one of include_events or include_min_bias '
                         'must be True.')
    # Validate user input for verbose parameter
    if not isinstance(verbose, int):
        raise ValueError('Expecting an int for verbose, '
                         'got {}'.format(type(verbose)))

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
    except:
        raise ValueError('Invalid start_date or end_date entered')

    # Build up list of all image files within the start_date to end_date range
    file_list = []
    for date in dates:
        date_files_pattern = os.path.join(data_dir, date, '*.png')
        date_file_list = glob.glob(date_files_pattern)
        file_list.extend(date_file_list)

    # If specified, filter out events/minimum bias images appropriately
    if include_events and include_min_bias:
        pass
    elif include_events and not include_min_bias:
        file_list = [f for f in file_list if 'minBias' not in f]
    elif not include_events and include_min_bias:
        file_list = [f for f in file_list if 'minBias' in f]

    file_model_list = []
    file_device_list = []

    # If specified, only keep files with desired phone model(s)
    if phone_model is not None:
        # Validate phone_model input
        if isinstance(phone_model, str):
            phone_model_list = [phone_model]
        assert isinstance(phone_model_list, (list, tuple, np.ndarray))
        
        filtered_list = []
        # Filter out non-matching phone models
        for idx, f in enumerate(file_list):
            try:
                if get_phone_model(f) in phone_model_list: 
                    filtered_list.append(f)
            except:
                continue
        file_model_list = filtered_list

    # If specified, only keep files with desired device ID(s)
    if device_id is not None:
        # Validate device_id input
        if isinstance(device_id,str):
            device_id_list = [device_id]
        assert isinstance(device_id_list, (list, tuple, np.ndarray))
 
        filtered_list = []
        # Filter out non-matching device IDs
        for idx, f in enumerate(file_list):
            try:
                if get_id(f) in device_id_list:
                    filtered_list.append(f)
            except:
                continue
        
        file_device_list = filtered_list
    

    # Merge the two lists, or specify which one is the correct file_list
    if device_id is not None or phone_model is not None:    
        if file_model_list and file_device_list:
            file_list = file_model_list + file_device_list
        elif file_model_list and not file_device_list:
            file_list = file_model_list
        elif file_device_list and not file_model_list:
            file_list = file_device_list
        else:
            # if the model list and device list are both empty,
            # return an empty list
            file_list = []


    # Get rid of duplicates
    file_list = list(set(file_list))


    # Cast file_list from a python list to a numpy.ndarray
    file_array = np.asarray(file_list, dtype=str)

    # Optional verbose output
    if verbose:
        print('Found {} iOS image files'.format(file_array.shape[0]))

    return file_array
