
import pytest
from collections import Counter
import numpy as np
from skimage.io import imsave

from ..blob_extraction import is_hotspot, get_intensity_metrics


def test_is_hotspot_threshold_1():
    # Check that identical x and y coordinates with a threshold of 1
    # identifies all coordinate pairs as hot spots
    np.random.seed(2)
    x = np.random.randint(low=0, high=100, size=200)
    y = x
    hotspots = is_hotspot(x, y, threshold=1)
    np.testing.assert_equal(hotspots, np.ones_like(hotspots))


def test_is_hotspot_inconsistent_coords():
    # Check that x and y coordinates of different shape raises an error
    with pytest.raises(ValueError) as excinfo:
        x = range(10)
        y = range(11)
        is_hotspot(x, y)
    error = 'x_coords and y_coords must have the same shape.'
    assert error == str(excinfo.value)


def test_get_intensity_metrics_no_files():
    # Ensure that an empty input file list gives an empty DataFrame
    df_metrics = get_intensity_metrics(files=[])
    assert df_metrics.empty


def test_get_intensity_metrics_bad_file():
    # Ensure that non-existant input file raises IOError w/ reasonable message
    with pytest.raises(IOError) as excinfo:
        bad_file = '/this/file/is/not/here.png'
        get_intensity_metrics(bad_file)
    assert 'No such file' in str(excinfo.value)


def test_get_intensity_metrics_columns(tmpdir):
    # Create and save a test image to a temporary file
    tmpfile = tmpdir.join('temp_image.png')
    imsave(str(tmpfile), np.random.random((5, 5, 4)))

    columns = ['max', 'mean', 'percentile_16', 'percentile_50',
               'percentile_84']
    df_metrics = get_intensity_metrics(str(tmpfile))

    assert Counter(df_metrics.columns) == Counter(columns)
