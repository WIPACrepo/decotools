
import pytest

from decotools.convnet import CNN


def test_CNN_load_model_fails():
    with pytest.raises(IOError) as excinfo:
        model_file = '/this/file/does/not/exist.h5'
        cnn = CNN()
        cnn._load_model(model_file)
    error = 'No model file found for {}...'.format(model_file)
    assert error == str(excinfo.value)


def test_CNN_load_weights_fails():
    with pytest.raises(IOError) as excinfo:
        weights_file = '/this/file/does/not/exist.h5'
        cnn = CNN()
        cnn._load_weights(weights_file)
    error = 'No weights file found for {}...'.format(weights_file)
    assert error == str(excinfo.value)


def test_CNN_smooth_labels_fails():
    with pytest.raises(ValueError) as excinfo:
        cnn = CNN()
        smooth_factor = 2
        y = [[0, 1], [3, 3]]
        cnn._smooth_labels(y, smooth_factor=smooth_factor)
    error = 'Invalid label smoothing factor: {}'.format(smooth_factor)
    assert error == str(excinfo.value)
