from __future__ import division, print_function
import os
import sys
import numpy as np
from PIL import Image
import keras
from keras.models import Sequential, load_model
from keras.layers import (Conv2D, MaxPooling2D, Cropping2D, Flatten,
                          Dense, Dropout)
from keras.layers.advanced_activations import LeakyReLU
from keras.constraints import maxnorm
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import BaseCrossValidator, StratifiedKFold


def get_crop_range(maxX, maxY, size=32):
    """Define region of image to crop
    """
    return maxX-size, maxX+size, maxY-size, maxY+size


def pass_edge_check(maxX, maxY, img_shape, crop_size=64):
    """Checks if image is on the edge of the sensor
    """
    x0, x1, y0, y1 = get_crop_range(maxX, maxY, size=crop_size/2)
    checks = np.array([x0 >= 0, x1 <= img_shape[0],
                       y0 >= 0, y1 <= img_shape[1]])
    return checks.all()


def get_brightest_pixel(img):
    """Get brightest image pixel indices
    """
    img = np.array(img)
    summed_img = np.sum(img, axis=-1)
    return np.unravel_index(summed_img.argmax(), summed_img.shape)


def convert_images(images):
    """
    Convert an array of images to the format required by the CNN

    Parameters
    ----------
    images : numpy.ndarray
        Array (tensor) containing images to be passed to the CNN model. For
        predictions, array should have shape:(n_images, 64, 64, 3). For
        training, array should have shape:(n_images, 100, 100, 3)

    Returns
    -------
    numpy.ndarray
        Array of normalized, grayscale images with
        shape:(n_images, n_rows, n_cols, 1)
    """
    images = np.array(images, dtype='float32')
    images = np.mean(images/255., axis=-1, keepdims=True)
    if len(images.shape) == 3:
        images = np.expand_dims(images, axis=0)
    return images


def process_image_files(image_files, size=32, return_edge_filter=False,
                        verbose=False):
    """Process image files for use in CNN

    Parameters
    ----------
    image_files : array_like
        Iterable of image file paths
    size : int, optional
        Size to zoom in on brightest pixel. Note that the shape of the zoomed
        image will be of shape ``2*size``-by-``2*size``. So for a 64-by-64
        zoomed image, use ``size=32`` (default is 32).
    return_edge_filter : bool, optional
        Option to return a boolean array indicating which images passed the
        edge filter (default is False).
    verbose : bool, optional
        Option for verbose output (default is False).

    Returns
    -------
    images : numpy.ndarray
        Array of normalized, grayscale images with shape:(n_images, n_rows,
        n_cols, 1)
    edge_filter : numpy.ndarray
        Only returned if ``return_edge_filter`` is True. Boolean array
        indicating which image files passed the edge filter.
    """
    images = []
    edge_filter = np.ones_like(image_files, dtype=bool)
    n_edge = 0
    for idx, image_file in enumerate(image_files):
        image = Image.open(image_file).convert('RGB')
        maxY, maxX = get_brightest_pixel(image)
        if pass_edge_check(maxX, maxY, image.size, crop_size=2*size):
            x0, x1, y0, y1 = get_crop_range(maxX, maxY, size=size)
            cropped_img = image.crop((x0, y0, x1, y1))
            cropped_img = np.asarray(cropped_img)
            images.append(cropped_img)
        else:
            n_edge += 1
            edge_filter[idx] = False
            continue
        if verbose:
            edge_str = ('\rNumber of images that failed the edge '
                        'filter: {} of {}'.format(n_edge, idx+1))
            sys.stdout.write(edge_str)
            sys.stdout.flush()

    images = convert_images(images)

    if return_edge_filter:
        output = images, edge_filter
    else:
        output = images

    return output


class CNN(object):
    """CNN class

    Parameters
    ----------
    weights_file : str, optional
        Path and file name of an hdf5 file containing the trained model
        weights to be used by the CNN (default is None).
    model_file : str, optional
        Path and file name of an hdf5 file containing a trained model.
        Typically, this should only be used when continuing an existing
        training session (default is None).
    custom_model : keras model, optional
        User-defined, compiled keras model to be used in place of the
        default (default is None).
    training : bool, optional
        If True, initializes the model structure used for training. If
        False, initializes the model structure used for predictions
        (default is False).
    n_classes : int, optional
        Number of classes to be used by the CNN (default is 4).

    """
    def __init__(self, weights_file=None, model_file=None, custom_model=None,
                 training=False, n_classes=4):
        self.weights_file = weights_file
        self.model_file = model_file
        self.custom_model = custom_model
        self.n_classes = n_classes
        self.training = training
        if custom_model:
            self.model = custom_model
        else:
            self._build_model()
        if weights_file:
            self._load_weights(weights_file)
        if model_file:
            self._load_model(model_file)

    def __repr__(self):
        attributes = ['weights_file', 'model_file', 'custom_model',
                      'training', 'n_classes']
        attr_str = ['{}={}'.format(att, getattr(self, att, None))
                    for att in attributes]
        rep = 'CNN({})'.format(', '.join(attr_str))
        return rep

    def _build_model(self):
        """ Define CNN model structure
        """
        self.model = Sequential()

        if self.training:
            self.model.add(Cropping2D(cropping=18, input_shape=(100, 100, 1)))
            self.model.add(Dropout(0.2))
            self.model.add(Conv2D(64, (3, 3), padding='same'))
        else:
            self.model.add(Conv2D(64, (3, 3), padding='same',
                           input_shape=(64, 64, 1)))

        self.model.add(LeakyReLU(alpha=0.3))
        self.model.add(Conv2D(64, (3, 3), padding='same'))
        self.model.add(LeakyReLU(alpha=0.3))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.2))

        self.model.add(Conv2D(128, (3, 3), padding='same'))
        self.model.add(LeakyReLU(alpha=0.3))
        self.model.add(Conv2D(128, (3, 3), padding='same'))
        self.model.add(LeakyReLU(alpha=0.3))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.2))

        self.model.add(Conv2D(256, (3, 3), padding='same'))
        self.model.add(LeakyReLU(alpha=0.3))
        self.model.add(Conv2D(256, (3, 3), padding='same'))
        self.model.add(LeakyReLU(alpha=0.3))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.2))

        self.model.add(Conv2D(512, (3, 3), padding='same'))
        self.model.add(LeakyReLU(alpha=0.3))
        self.model.add(Conv2D(512, (3, 3), padding='same'))
        self.model.add(LeakyReLU(alpha=0.3))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.2))

        self.model.add(Flatten())
        self.model.add(Dense(2048, kernel_constraint=maxnorm(3)))
        self.model.add(LeakyReLU(alpha=0.3))
        self.model.add(Dropout(0.4))
        self.model.add(Dense(2048, kernel_constraint=maxnorm(3)))
        self.model.add(LeakyReLU(alpha=0.3))
        self.model.add(Dropout(0.4))
        self.model.add(Dense(self.n_classes, activation='softmax'))
        self.model.compile(loss=keras.losses.categorical_crossentropy,
                           optimizer=keras.optimizers.Adadelta(),
                           metrics=['accuracy'])

    def _calculate_class_weights(self, labels):
        """ calculate class weights to be used for training
        """
        classes = np.arange(self.n_classes)
        counts = np.array([])
        for i in classes:
            counts = np.append(counts, np.sum(labels == i))
        counts = np.amax(counts)/counts
        self.class_weights = dict(zip(classes, counts))

    def _load_model(self, model):
        """ Load an existing model structure and weights
        """
        if os.path.isfile(model):
            self.model = load_model(model)
        else:
            raise IOError('No model file found for {}...'.format(model))

    def _load_weights(self, weights):
        """ Load trained model weights; assumes default model structure
        """
        if os.path.isfile(weights):
            self.model.load_weights(weights)
        else:
            raise IOError('No weights file found for {}...'.format(weights))

    def _smooth_labels(self, y, smooth_factor):
        """
        Convert a matrix of one-hot row-vector labels into smoothed versions.

        Parameters
        ----------
        y : numpy.ndarray
            Array of one-hot row-vector labels
        smooth_factor : float
            value between 0-1 to be used for label smoothing

        Returns
        -------
        numpy.ndarray
            Array containing smoothed one-hot row-vector labels
        """
        y = np.array(y, copy=True)
        assert len(y.shape) == 2
        if 0 <= smooth_factor <= 1:
            y *= 1 - smooth_factor
            y += smooth_factor / y.shape[1]
        else:
            raise ValueError(
                    'Invalid label smoothing factor: {}'.format(smooth_factor))
        return y

    def evaluate(self, images, labels, batch_size=32, verbose=0):
        """Evaluate accuracy and loss of model predictions

        Parameters
        ----------
        images : numpy.ndarray
            Array of grayscale, normalized images to be used for evaluation.
            Input shape = (n_image,n_row,n_cols,1).
        labels : numpy.ndarray
            Array of labels to be used for evaluation, shape=(n_images,).
        batch_size : int, optional
            Batch size to use for predictions (default is 32).
        verbose : int, optional
            Verbosity mode to use, 0 or 1 (default is 0).

        Returns
        -------
        list
            list containing [test_loss, test_accuracy]
        """
        labels = to_categorical(labels, self.n_classes)
        score = self.model.evaluate(images, labels, batch_size,
                                    verbose=verbose)
        return score

    def model_summary(self):
        """ Print summary of currently loaded model
        """
        print(self.model.summary())

    def predict(self, images, batch_size=32, verbose=0):
        """ Predict classifications for an input image array

        Parameters
        ----------
        images : numpy.ndarray
            Array of grayscale, normalized images to be used for class
            predictions. Input shape = (n_image,n_row,n_cols,1)
        batch_size : int, optional
            Batch size to use for predictions (default is 32).
        verbose : int, optional
            Verbosity mode to use, 0 or 1 (default is 0).

        Returns
        -------
        numpy.ndarray
            Array containing class probabilities for each image. The array
            output is ordered as follows: [n_images, p(worm), p(spot),
            p(track), p(noise)]

        """
        return self.model.predict(images, batch_size=batch_size,
                                  verbose=verbose)

    def _fit(self, train_images, train_labels, test_images=None,
             test_labels=None, batch_size=64, seed=None, epochs=10,
             initial_epoch=0, smooth_factor=0.004, horizontal_flip=True,
             vertical_flip=True, width_shift_range=0.08,
             height_shift_range=0.08, rotation_range=180.,
             zoom_range=(0.9, 1.1), fill_mode='constant', cval=0,
             shuffle=True, save_model=None, save_weights=None,
             save_history=None, check_point=None,
             check_point_weights_only=True, verbose=False):

        # Convert labels to one-hot and apply smoothing
        train_labels = to_categorical(train_labels, self.n_classes)
        if smooth_factor:
            train_labels = self._smooth_labels(train_labels, smooth_factor)
        if test_labels is not None:
            test_labels = to_categorical(test_labels, self.n_classes)
            validation_data = (test_images, test_labels)
        else:
            validation_data = None

        # Setup checkpointer
        if check_point is not None:
            checkpointer = [ModelCheckpoint(check_point,
                            monitor='val_loss', verbose=0,
                            save_weights_only=check_point_weights_only,
                            save_best_only=True, mode='auto')]
        else:
            checkpointer = None

        # Preprocess images
        datagen = ImageDataGenerator(
                        horizontal_flip=horizontal_flip,
                        vertical_flip=vertical_flip,
                        width_shift_range=width_shift_range,
                        height_shift_range=height_shift_range,
                        rotation_range=rotation_range,
                        zoom_range=zoom_range,
                        fill_mode=fill_mode,
                        cval=cval)

        # Fit the model
        datagen.fit(train_images)
        steps_per_epoch = train_images.shape[0] // batch_size
        hist = self.model.fit_generator(datagen.flow(train_images,
                                                     train_labels,
                                                     batch_size=batch_size,
                                                     seed=seed,
                                                     shuffle=shuffle),
                                        steps_per_epoch=steps_per_epoch,
                                        epochs=epochs,
                                        class_weight=self.class_weights,
                                        callbacks=checkpointer,
                                        validation_data=validation_data,
                                        initial_epoch=initial_epoch)

        # Evaluate  model
        if validation_data:
            score = self.model.evaluate(test_images, test_labels, verbose=0)
            if verbose:
                print('Final test loss:', score[0])
                print('Final test accuracy:', score[1])

        # Save model, weights, history
        if save_model:
            self.model.save(save_model)
        if save_weights:
            self.model.save_weights(save_weights)
        if save_history:
            hist_vals = np.array([hist.history['acc'],
                                  hist.history['val_acc'],
                                  hist.history['loss'],
                                  hist.history['val_loss']])
            np.savetxt(save_history,
                       np.transpose(hist_vals),
                       delimiter=',',
                       header='acc,val_acc,loss,val_loss')

        return self

    def fit(self, train_images, train_labels, test_images=None,
            test_labels=None, cv=None, batch_size=64, seed=None, epochs=10,
            initial_epoch=0, smooth_factor=0.004, horizontal_flip=True,
            vertical_flip=True, width_shift_range=0.08,
            height_shift_range=0.08, rotation_range=180.,
            zoom_range=(0.9, 1.1), fill_mode='constant', cval=0,
            shuffle=True, save_model=None, save_weights=None,
            save_history=None, check_point=None,
            check_point_weights_only=True, verbose=False):
        """Fit CNN

        Parameters
        ----------
        train_images : numpy.ndarray
            Array of grayscale, normalized images to be used for training the
            CNN. Input shape = (n_image,n_row,n_cols,1).
        train_labels : numpy.ndarray
            Array of training labels, shape=(n_images,).
        test_images : numpy.ndarray, optional
            Array of grayscale, normalized images to be used for testing the
            CNN. Input shape = (n_image,n_row,n_cols,1).
        test_labels : numpy.ndarray, optional
            Array of testing labels, shape=(n_images,).
        cv : int, scikit-learn cross validator, None, optional
            Option for cross-validation fitting. If ``cv`` is an integer
            ``sklearn.model_selection.StratifiedKFold`` will be used with
            ``cv`` number of folds. Other cross validators from
            ``sklearn.model_selection`` can be passed to ``cv`` as well
            (default is None).
        batch_size : int, optional
            Number of samples per gradient update (default is 64).
        seed : int, optional
            Random seed to be used for reproducibility. (default is None).
        epochs : int, optional
            Number of epochs to train the model. Note that in conjunction with
            initial_epoch, the parameter epochs is to be understood as
            "final epoch". (default is 10).
        initial_epoch : int, optional
            Epoch at which to start training. Useful for resuming a previous
            training run (default is 0).
        smooth_factor : float in range (0, 1), optional
            Level of smoothing to apply to one-hot label vector. Ex.
            smooth_factor of 0.004 applied to [0, 1, 0, 0], results in
            [0.001, 0.997, 0.001, 0.001] (default is 0.004).
        horizontal_flip : bool, optional
            Randomly flip inputs horizontally (default is True).
        vertical_flip : bool, optional
            Randomly flip inputs vertically (default is True).
        width_shift_range : float, optional
            Range for random horizontal shifts (default is 0.08).
        height_shift_range : float, optional
            Range for random vertical shifts (default is 0.08).
        rotation_range : int, optional
            Degree range for random rotations (default is 180).
        zoom_range : float or (lower, upper), optional
            Range for random zoom. If a float,
            ``(lower, upper) = (1-zoom_range, 1+zoom_range)`` (default
            is ``(0.9, 1.1)``).
        fill_mode : {"constant", "nearest", "reflect" or "wrap"}, optional
            Points outside the boundaries of the input are filled according to
            the given mode (default is "constant").
        cval : float or int, optional
            Value used for interpolated pixels when ``fill_mode="constant"``
            (default is 0).
        shuffle : bool, optional
            Whether to shuffle the order of the batches at the beginning of
            each epoch (default is True).
        save_model : str, optional
            If specified, a copy of the model from the final training epoch
            will be saved. For example, ``save_model='my_model.h5'``.
            Typically used for continued training (default is None).
        save_weights : str, optional
            If specified, a copy of the model weights from the final training
            epoch will be saved. For example, ``save_weights='my_weights.h5'``
            (default is None).
        save_history : str, optional
            If specified, the training history (accuracy and loss for training
            and testing) from each epoch will be saved to a CSV file.
            For example, ``save_history='my_history.csv'`` (default is None).
        check_point : str, optional
            If specified, saves a running copy of the model corresponding to
            the lowest validation loss epoch. Each time a new low is reached,
            the previous best model is over-written by the new one.
            For example, ``check_point='my_checkpoint.h5'`` (default is None).
        check_point_weights_only : bool, optional
            If True, only the model's weights will be saved in the check
            point. Otherwise the full model is saved. Ignored if
            ``check_point=False`` (default is True).
        verbose : bool, optional
            Option for verbose output.

        Returns
        -------
        self : CNN
            Trained CNN.
        """
        # Validate user input
        if not self.training:
            raise ValueError('CNN class initialized with training=\'False\', '
                             'must be \'True\'')

        train_images, train_labels = _check_shape(train_images, train_labels)
        if test_labels is not None:
            test_images, test_labels = _check_shape(test_images, test_labels)

        if batch_size > train_images.shape[0]:
            raise ValueError(
                    'batch_size must be <= the number of training images')
        if (cv is not None) and (test_labels is not None):
            raise ValueError('Cross-validation fitting and an explicitly '
                             'given testing set can\'t be used at the '
                             'same time')

        if seed:
            np.random.seed(seed)

        # Calculate class weights
        self._calculate_class_weights(train_labels)

        args = dict(batch_size=batch_size, seed=seed, epochs=epochs,
                    initial_epoch=initial_epoch, smooth_factor=smooth_factor,
                    horizontal_flip=horizontal_flip,
                    vertical_flip=vertical_flip,
                    width_shift_range=width_shift_range,
                    height_shift_range=height_shift_range,
                    rotation_range=rotation_range, zoom_range=zoom_range,
                    fill_mode=fill_mode, cval=cval, shuffle=shuffle,
                    verbose=verbose,
                    check_point_weights_only=check_point_weights_only)

        if cv is None:
            self._fit(train_images, train_labels,
                      test_images=test_images, test_labels=test_labels,
                      check_point=check_point,
                      save_model=save_model,
                      save_weights=save_weights,
                      save_history=save_history, **args)
        else:
            # Split into testing and training sets
            if isinstance(cv, int):
                skf = StratifiedKFold(n_splits=cv, shuffle=shuffle,
                                      random_state=seed)
                splitter = skf.split(train_images, train_labels)
            elif isinstance(cv, BaseCrossValidator):
                splitter = cv.split(train_images, train_labels)
            else:
                raise TypeError('cv must be an integer or an instance of '
                                'sklearn.model_selection.BaseCrossValidator')

            # Store initial weights
            init_weights = self.model.get_weights()

            train_indices, test_indices = [], []
            for fold_idx, (train_index, test_index) in enumerate(splitter):
                if verbose:
                    print('Training fold number {}'.format(fold_idx))
                # Get testing and training split for this fold
                train_images_fold = train_images[train_index]
                test_images_fold = train_images[test_index]
                train_labels_fold = train_labels[train_index]
                test_labels_fold = train_labels[test_index]

                train_indices.append(train_index)
                test_indices.append(test_index)

                # Setup model, weights, and history output files
                check_outfile = None
                model_outfile = None
                weights_outfile = None
                history_outfile = None
                if save_model:
                    model_outfile = _get_cv_outfile(save_model, fold_idx)
                if save_weights:
                    weights_outfile = _get_cv_outfile(save_weights, fold_idx)
                if save_history:
                    history_outfile = _get_cv_outfile(save_history, fold_idx)
                if check_point:
                    check_outfile = _get_cv_outfile(check_point, fold_idx)

                self.model.set_weights(init_weights)
                self._fit(train_images_fold, train_labels_fold,
                          test_images=test_images_fold,
                          test_labels=test_labels_fold,
                          check_point=check_outfile,
                          save_model=model_outfile,
                          save_weights=weights_outfile,
                          save_history=history_outfile,
                          **args)

            self.train_indices = np.asarray(train_indices)
            self.test_indices = np.asarray(test_indices)

        return self


def _get_cv_outfile(file_path, fold_idx):
    head, tail = os.path.split(file_path)
    root, ext = os.path.splitext(tail)
    file_path_fold = os.path.join(head, '{}_{}{}'.format(root, fold_idx, ext))
    return file_path_fold


def _check_shape(x, y):
    """Checks that x and y are have consistent shapes.

    Casts x and y to numpy.ndarrays, using numpy.asarray, and checks that
    x and y have the same number of samples (i.e. x.shape[0] == y.shape[0])

    Parameters
    ----------
    x : array_like
        First input array.
    y : array_like
        Second input array.

    Returns
    -------
    x : numpy.ndarray
        Numpy array version of x.
    y : numpy.ndarray
        Numpy array version of y.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if not x.shape[0] == y.shape[0]:
        raise ValueError('Input arrays have shape mismatch')

    return x, y
