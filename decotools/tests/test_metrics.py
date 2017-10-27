
import pytest
from collections import Counter
import numpy as np
from skimage.io import imsave

from decotools.metrics import get_intensity_metrics, get_rgb_hists
from decotools.blob_extraction import get_image_array
from decotools.tests.utils import save_test_images


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
    # Create and save a test images
    files = save_test_images(tmpdir, n_images=10)

    columns = ['max', 'mean', 'percentile_16', 'percentile_50',
               'percentile_84']
    df_metrics = get_intensity_metrics(files)

    assert Counter(df_metrics.columns) == Counter(columns)


def test_get_rgb_hists_shape(tmpdir):
    # Create and save a test images
    files = save_test_images(tmpdir, n_images=10)

    df_rgb_hists = get_rgb_hists(files)
    assert df_rgb_hists.shape == (len(files), 769)

    df_rgb_hists_cumulative = get_rgb_hists(files, cumulative=True)
    assert df_rgb_hists_cumulative.shape == (len(files), 769)


def test_get_rgb_hists_sum(tmpdir):
    # Test that the rgb histogram for each image (each row in the ouput from
    # get_rgb_hists) sums to the number of pixels in that image.

    # Create and save a test images
    files = save_test_images(tmpdir, n_images=10)

    df_rgb_hists = get_rgb_hists(files)
    n_pixels = np.asarray([get_image_array(f).size for f in files])

    np.testing.assert_array_equal(df_rgb_hists.sum(axis=1), n_pixels)


def test_get_rgb_hists_cumulative_npixels(tmpdir):
    # Test that the counts in the first bin of the cumulative rgb histogram
    # for each image is equal to the total number of pixels in that image.

    # Create and save a test images
    files = save_test_images(tmpdir, n_images=10)

    df_cumulative_rgb_hists = get_rgb_hists(files, cumulative=True)
    n_pixels = [get_image_array(f).size for f in files]

    np.testing.assert_array_equal(df_cumulative_rgb_hists.loc[:, 0], n_pixels)
