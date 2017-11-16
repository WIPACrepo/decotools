
import pytest
import numpy as np

from decotools.convnet import (CNN, get_brightest_pixel, get_crop_range,
                               process_image_files)
from decotools.tests.utils import save_test_images


default_cnn = CNN()
training_cnn = CNN(training=True)

np.random.seed(2)

n_training = 10
train_images = np.random.random((n_training, 100, 100, 1))
train_labels = np.random.randint(2, size=n_training)

n_testing = 10
test_images = np.random.random((n_testing, 100, 100, 1))
test_labels = np.random.randint(2, size=n_testing)


def test_CNN_load_model_fails():
    with pytest.raises(IOError) as excinfo:
        model_file = '/this/file/does/not/exist.h5'
        default_cnn._load_model(model_file)
    error = 'No model file found for {}...'.format(model_file)
    assert error == str(excinfo.value)


def test_CNN_load_weights_fails():
    with pytest.raises(IOError) as excinfo:
        weights_file = '/this/file/does/not/exist.h5'
        default_cnn._load_weights(weights_file)
    error = 'No weights file found for {}...'.format(weights_file)
    assert error == str(excinfo.value)


def test_CNN_smooth_labels_fails():
    with pytest.raises(ValueError) as excinfo:
        smooth_factor = 2
        y = [[0, 1], [3, 3]]
        default_cnn._smooth_labels(y, smooth_factor=smooth_factor)
    error = 'Invalid label smoothing factor: {}'.format(smooth_factor)
    assert error == str(excinfo.value)


def test_CNN_fit_passes():
    training_cnn.fit(train_images, train_labels, test_images, test_labels,
                     epochs=1, batch_size=n_training)


def test_CNN_fit_with_kfold_passes():
    training_cnn.fit_with_kfold(train_images, train_labels, epochs=1,
                                batch_size=1, k_folds=2)


def test_CNN_fit_batchsize_raises():
    with pytest.raises(ValueError) as excinfo:
        training_cnn.fit(train_images, train_labels,
                         test_images, test_labels,
                         batch_size=n_training+1)
    error = 'batch_size must be <= the number of training images'
    assert error == str(excinfo.value)


def test_CNN_fit_with_kfold_batchsize_raises():
    with pytest.raises(ValueError) as excinfo:
        training_cnn.fit_with_kfold(train_images, train_labels,
                                    test_images, test_labels,
                                    batch_size=n_training+1)
    error = 'batch_size must be <= the number of training images'
    assert error == str(excinfo.value)


def test_CNN_fit_train_samples_raises():
    with pytest.raises(ValueError) as excinfo:
        training_cnn.fit(train_images[:-1], train_labels,
                         test_images, test_labels,
                         batch_size=n_training+1)
    error = ('The number of training labels does not match '
             'the number of training images')
    assert error == str(excinfo.value)


def test_CNN_fit_with_kfold_train_samples_raises():
    with pytest.raises(ValueError) as excinfo:
        training_cnn.fit_with_kfold(train_images[:-1], train_labels,
                                    batch_size=n_training+1)
    error = ('The number of training labels does not match '
             'the number of training images')
    assert error == str(excinfo.value)


def test_CNN_fit_test_samples_raises():
    with pytest.raises(ValueError) as excinfo:
        training_cnn.fit(train_images, train_labels,
                         test_images[:-1], test_labels,
                         batch_size=n_training+1)
    error = ('The number of testing labels does not match '
             'the number of testing images')
    assert error == str(excinfo.value)


def test_get_brightest_pixel():
    img = np.zeros((100, 100, 3))
    max_x = 23
    max_y = 76
    img[max_x, max_y] = [1, 1, 1]
    brightest_y, brightest_x = get_brightest_pixel(img)
    assert (max_x, max_y) == (brightest_y, brightest_x)


def test_get_crop_range():
    max_x = 5
    max_y = 23
    size = 17
    result = (max_x-size, max_x+size, max_y-size, max_y+size)
    assert result == get_crop_range(max_x, max_y, size=size)


def test_process_image_files_edge_filter(tmpdir):
    # Create and save a test images
    files = save_test_images(tmpdir, n_images=10, scale=50, shape=(500, 500),
                             seed=23)

    images, edge_filter = process_image_files(files, return_edge_filter=True)
    test_filter = [False, True, True, True, True, True, True, False,
                   False, True]
    np.testing.assert_array_equal(test_filter, edge_filter)


def test_process_image_files_shape(tmpdir):
    # Create and save a test images
    n_images = 20
    files = save_test_images(tmpdir, n_images=n_images, scale=50,
                             shape=(500, 500))
    size = 32
    images, edge_filter = process_image_files(files, size=size,
                                              return_edge_filter=True)
    test_shape = (edge_filter.sum(), 2*size, 2*size, 1)
    np.testing.assert_array_equal(images.shape, test_shape)
