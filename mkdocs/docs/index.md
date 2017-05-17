# Welcome to the tools documentation!

Decotools is a python package for analyzing [DECO data](https://wipac.wisc.edu/deco/home).

## File I/O

```python
def get_iOS_files(start_date=None, end_date=None, data_dir='/net/deco/iOSdata',
                  include_events=True, include_min_bias=False,
                  phone_model=None, verbose=0)
```

Function to retrieve deco iOS image files

**Parameters**

* `start_date` : str, optional

    Starting date for the iOS files to retrieve. Use any common
    date format (e.g. '2017-01-01', '20170101', 'Jan 1, 2017', etc).
    Default starting date is '2016.01.01'.

* `end_date` : str, optional

    Ending date for the iOS files to retrieve. Use any common
    date format (e.g. '2017-01-01', '20170101', 'Jan 1, 2017', etc).
    Default is the current date.

* `data_dir` : str, optional

    Base directory to search for iOS image files.

* `include_events` : bool, optional

    Option to include images files flagged as events. Default is True.

* `include_min_bias` : bool, optional

    Option to include minimum bias image files. Default is False.

* `phone_model` : str or list, optional

    Option to specify which phone models you would like to look at.
    Can be either a string, e.g. 'iPhone 7', or a list of models,
    e.g. ['iPhone 5', 'iPhone 5s']. Default is to include all
    phone models.

* `verbose` : int (0 or 1)

    Option to have verbose output when getting files. Where 0 is
    least verbose, while 1 is the most verbose.

**Returns**

* numpy.ndarray

    Numpy array containing files that match specified criteria


## Blob detection
