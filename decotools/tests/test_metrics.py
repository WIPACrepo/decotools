
import pytest
from collections import Counter
import numpy as np
from skimage.io import imsave

from decotools.metrics import get_intensity_metrics


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
