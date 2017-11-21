
import pytest
import os
import numpy as np
from sklearn.model_selection import KFold
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Cropping2D
from keras.layers import Flatten, Dense, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.constraints import maxnorm

from decotools.convnet import (CNN, get_brightest_pixel, get_crop_range,
                               process_image_files, _get_cv_outfile,
                               _check_shape)
from decotools.tests.utils import save_test_images


np.random.seed(2)

# Construct data set for running tests
n_classes = 4

n_training = 10
train_images = np.random.random((n_training, 100, 100, 1))
train_labels = np.random.randint(n_classes, size=n_training)

n_testing = 10
test_images = np.random.random((n_testing, 100, 100, 1))
test_labels = np.random.randint(n_classes, size=n_testing)

# Construct simplified model to speed up testing
model = Sequential()
model.add(Cropping2D(cropping=18, input_shape=(100, 100, 1)))
model.add(Dropout(0.2))
model.add(Conv2D(16, (3, 3), padding='same'))
model.add(LeakyReLU(alpha=0.3))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(16, kernel_constraint=maxnorm(3)))
model.add(LeakyReLU(alpha=0.3))
model.add(Dropout(0.4))
model.add(Dense(n_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

cnn = CNN(custom_model=model)
training_cnn = CNN(training=True, custom_model=model)
default_cnn = CNN(training=True)


def test_CNN_load_model_fails():
    with pytest.raises(IOError) as excinfo:
        model_file = '/this/file/does/not/exist.h5'
        cnn._load_model(model_file)
    error = 'No model file found for {}...'.format(model_file)
    assert error == str(excinfo.value)


def test_CNN_load_weights_fails():
    with pytest.raises(IOError) as excinfo:
        weights_file = '/this/file/does/not/exist.h5'
        cnn._load_weights(weights_file)
    error = 'No weights file found for {}...'.format(weights_file)
    assert error == str(excinfo.value)


def test_CNN_smooth_labels_fails():
    with pytest.raises(ValueError) as excinfo:
        smooth_factor = 2
        y = [[0, 1], [3, 3]]
        cnn._smooth_labels(y, smooth_factor=smooth_factor)
    error = 'Invalid label smoothing factor: {}'.format(smooth_factor)
    assert error == str(excinfo.value)


def test_CNN_smooth_labels():
    smooth_factor = 0
    y = np.array([[0, 1], [1, 0]], dtype=float)
    y_smoothed = cnn._smooth_labels(y, smooth_factor=smooth_factor)

    np.testing.assert_array_equal(y, y_smoothed)


def test_CNN_fit_training_raises():
    with pytest.raises(ValueError) as excinfo:
        cnn.fit(train_images, train_labels,
                epochs=1, batch_size=n_training)
    error = 'CNN class initialized with training=\'False\', must be \'True\''
    assert error == str(excinfo.value)


def test_default_CNN_fit_passes():
    default_cnn.fit(train_images, train_labels,
                    test_images=test_images, test_labels=test_labels,
                    epochs=1, batch_size=n_training)


def test_CNN_fit_passes():
    training_cnn.fit(train_images, train_labels,
                     test_images=test_images, test_labels=test_labels,
                     epochs=1, batch_size=n_training)


def test_CNN_fit_no_validation_passes():
    training_cnn.fit(train_images, train_labels,
                     epochs=1, batch_size=n_training)


def test_CNN_fit_CV_passes():
    training_cnn.fit(train_images, train_labels,
                     epochs=1, batch_size=1, cv=2)


def test_CNN_fit_kfold_CV():
    kfold = KFold(n_splits=2, shuffle=True, random_state=2)
    training_cnn.fit(train_images, train_labels,
                     epochs=1, batch_size=1, cv=kfold)

    train_indices, test_indices = [], []
    for train_index, test_index in kfold.split(train_images):
        train_indices.append(train_index)
        test_indices.append(test_index)

    np.testing.assert_array_equal(train_indices, training_cnn.train_indices)
    np.testing.assert_array_equal(test_indices, training_cnn.test_indices)


def test_CNN_fit_CV_raises():
    with pytest.raises(TypeError) as excinfo:
        training_cnn.fit(train_images, train_labels,
                         epochs=1, batch_size=1,
                         cv='Not a proper cross-validator')
    error = ('cv must be an integer or an instance of '
             'sklearn.model_selection.BaseCrossValidator')
    assert error == str(excinfo.value)


def test_CNN_fit_batchsize_raises():
    with pytest.raises(ValueError) as excinfo:
        training_cnn.fit(train_images, train_labels,
                         test_images, test_labels,
                         batch_size=n_training+1)
    error = 'batch_size must be <= the number of training images'
    assert error == str(excinfo.value)


def test_CNN_fit_train_samples_raises():
    with pytest.raises(ValueError) as excinfo:
        training_cnn.fit(train_images[:-1], train_labels,
                         test_images, test_labels,
                         batch_size=n_training)
    error = 'Input arrays have shape mismatch'
    assert error == str(excinfo.value)


def test_CNN_fit_test_samples_raises():
    with pytest.raises(ValueError) as excinfo:
        training_cnn.fit(train_images, train_labels,
                         test_images[:-1], test_labels,
                         batch_size=n_training)
    error = 'Input arrays have shape mismatch'
    assert error == str(excinfo.value)


def test_CNN_fit_cv_test_images_raises():
    with pytest.raises(ValueError) as excinfo:
        training_cnn.fit(train_images, train_labels,
                         test_images=test_images, test_labels=test_labels,
                         batch_size=1, cv=2)
    error = ('Cross-validation fitting and an explicitly '
             'given testing set can\'t be used at the '
             'same time')
    assert error == str(excinfo.value)


def test_CNN_output_files(tmpdir):
    check_point = str(tmpdir.join('best_checkpointed_model.h5'))
    model_outfile = str(tmpdir.join('my_model.h5'))
    weights_outfile = str(tmpdir.join('my_weights.h5'))
    history_outfile = str(tmpdir.join('my_history.npy'))

    training_cnn.fit(train_images, train_labels,
                     test_images, test_labels,
                     epochs=1, batch_size=n_training,
                     check_point=check_point,
                     save_model=model_outfile,
                     save_weights=weights_outfile,
                     save_history=history_outfile)
    for f in [check_point, model_outfile, weights_outfile, history_outfile]:
        # Check that output file exists
        assert os.path.exists(f)
        # Check that output file isn't empty
        assert os.path.getsize(f) > 0


def test_CNN_cv_output_files(tmpdir):
    cv = 2
    check_point = str(tmpdir.join('best_checkpointed_model.h5'))
    model_outfile = str(tmpdir.join('my_model.h5'))
    weights_outfile = str(tmpdir.join('my_weights.h5'))
    history_outfile = str(tmpdir.join('my_history.npy'))
    training_cnn.fit(train_images, train_labels,
                     epochs=1, batch_size=1, cv=cv,
                     check_point=check_point,
                     save_model=model_outfile,
                     save_weights=weights_outfile,
                     save_history=history_outfile)
    for f in [check_point, model_outfile, weights_outfile, history_outfile]:
        for fold_idx in range(cv):
            cv_outfile = _get_cv_outfile(f, fold_idx)
            # Check that output file exists
            assert os.path.exists(cv_outfile)
            # Check that output file isn't empty
            assert os.path.getsize(cv_outfile) > 0


def test_CNN_model_file(tmpdir):
    # Test that the predictions from a trained model are the same as the
    # predictions from the corresponding saved model file.
    model_outfile = str(tmpdir.join('my_model.h5'))
    training_cnn.fit(train_images, train_labels,
                     epochs=1, batch_size=1,
                     save_model=model_outfile)
    # Get predictions from trained model
    pred_trained = training_cnn.predict(test_images)
    eval_trained = training_cnn.evaluate(test_images, test_labels)
    # Get predictions from saved model
    saved_cnn = CNN(model_file=model_outfile)
    pred_saved = saved_cnn.predict(test_images)
    eval_saved = saved_cnn.evaluate(test_images, test_labels)

    np.testing.assert_array_equal(pred_trained, pred_saved)
    np.testing.assert_array_equal(eval_trained, eval_saved)


def test_CNN_weights_file(tmpdir):
    # Test that the predictions from a trained model are the same as the
    # predictions from the corresponding saved weights file.
    weights_outfile = str(tmpdir.join('my_weights.h5'))
    training_cnn.fit(train_images, train_labels,
                     epochs=1, batch_size=1,
                     save_weights=weights_outfile)
    # Get predictions from trained model
    pred_trained = training_cnn.predict(test_images)
    eval_trained = training_cnn.evaluate(test_images, test_labels)
    # Get predictions from saved model
    saved_cnn = CNN(custom_model=model, weights_file=weights_outfile)
    pred_saved = saved_cnn.predict(test_images)
    eval_saved = saved_cnn.evaluate(test_images, test_labels)

    np.testing.assert_array_equal(pred_trained, pred_saved)
    np.testing.assert_array_equal(eval_trained, eval_saved)


def test_CNN_repr():
    cnn_repr = repr(default_cnn)
    print('cnn_repr = {}'.format(cnn_repr))
    expected_repr = ('CNN(weights_file=None, model_file=None, '
                     'custom_model=None, training=True, n_classes=4)')
    assert cnn_repr == expected_repr


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


def test_get_cv_outfile():
    file_path = '/path/to/my_saved_model.h5'
    fold_idx = 2
    cv_outfile = _get_cv_outfile(file_path, fold_idx)
    assert cv_outfile == '/path/to/my_saved_model_{}.h5'.format(fold_idx)


def test_check_shape():
    x = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
    y = [1, 2, 3, 4, 5]
    x_checked, y_checked = _check_shape(x, y)
    np.testing.assert_array_equal(x, x_checked)
    np.testing.assert_array_equal(y, y_checked)
