
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
    return


def get_phone_model(image_file):
    xml_file = image_file_to_xml_file(image_file)
    xml_dict = xml_to_dict(xml_file)

    return xml_dict['Model']


def get_iOS_files(start_date=None, stop_date=None, include_min_bias=False, phone_model=None):
    '''Function to retrieve deco iOS data files'''

    base_path = '/net/deco/iOSdata/'
    file_list = []

    # If no stop_date specified, set as today's date
    if not stop_date:
        stop_date = time.strftime('%Y.%m.%d')
    # If no start_date specified, set as some early date???
    if not start_date:
        start_date = '2016.01.23'

    # Get list of dates between start_date and stop_date
    try:
        dates = pd.date_range(start_date, stop_date)
        dates = dates.strftime('%Y.%m.%d')
    except:
        raise ValueError('Invalid start_date ({}) or stop_date ({}) entered'.format(
            start_date, stop_date))

    # Get all image files for the dates in the range from start_date to stop_date
    for date in dates:
        date_file_str = os.path.join(base_path, date, '*.png')
        date_file_list = glob.glob(date_file_str)
        file_list.extend(date_file_list)

    # If specified, remove minimum bias images
    if not include_min_bias:
        file_list = [f for f in file_list if 'minBias' not in f]

    # If specified, only keep files with desired phone model(s)
    if phone_model:
        if isinstance(phone_model, str):
            phone_model = [phone_model]
        for model in phone_model:
            file_list = [f for f in file_list if get_phone_model(f) == model]

    # Cast file_list from a python list to a numpy.ndarray
    file_list = np.asarray(file_list, dtype=str)

    return file_list
