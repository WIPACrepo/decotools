
import pytest
import numpy as np

from decotools.blob_extraction import is_hotspot


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
