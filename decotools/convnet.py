from __future__ import division, print_function
import os
import numpy as np
from PIL import Image
import keras
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Cropping2D
from keras.layers import Flatten, Dense, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.constraints import maxnorm
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import StratifiedKFold


def get_crop_range(maxX, maxY, size=32):
    """define region of image to crop"""
    return maxX-size, maxX+size, maxY-size, maxY+size


def pass_edge_check(maxX, maxY, img_shape, crop_size=64):
    """checks if image is on the edge of the sensor"""
    x0, x1, y0, y1 = get_crop_range(maxX, maxY, size=crop_size/2)
    checks = np.array([x0 >= 0, x1 <= img_shape[0],
                       y0 >= 0, y1 <= img_shape[1]])
    return checks.all()


def get_brightest_pixel(img):
    """get brightest image pixel indices"""
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


def process_image_files(image_files):
    for image_file in image_files:
        image = Image.open(image_file).convert('RGB')
        maxY, maxX = get_brightest_pixel(image)
        if pass_edge_check(maxX, maxY, image.size):
            x0, x1, y0, y1 = get_crop_range(maxX, maxY)
            cropped_img = image.crop((x0, y0, x1, y1))
            gray_image = convert_images(cropped_img)
            yield gray_image
        else:
            yield None


class CNN(object):
    """class for CNN training and predictions"""
    def __init__(self, weights_file=None, model_file=None, custom_model=None,
                 training=False, n_classes=4):
        """
        Initialize CNN class

        Parameters
        ----------
        weights_file : str, optional
            Path and file name of an hdf5 file containing the trained model
            weights to be used by the CNN. (default: None)
        model_file : str, optional
            Path and file name of an hdf5 file containing a trained model.
            Typically, this should only be used when continuing an existing
            training session. (default: None)
        custom_model : keras model, optional
            User-defined, compiled keras model to be used in place of the
            default (default: None)
        training : bool, optional
            If True, initializes the model structure used for training. If
            False, initializes the model structure used for predictions.
            (default: False)
        n_classes : int
            Number of classes to be used by the CNN. (default: 4)
        """
        self.n_classes = n_classes
        self.training = training
        self.train_indices = []
        self.test_indices = []
        if custom_model:
            self.model = custom_model
        else:
            self._build_model()
        if weights_file:
            self._load_weights(weights_file)
        if model_file:
            self._load_model(model_file)

    def _build_model(self):
        """ Define CNN model structure """
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
        """ calculate class weights to be used for training """
        classes = np.arange(self.n_classes)
        counts = np.array([])
        for i in classes:
            counts = np.append(counts, np.sum(labels == i))
        counts = np.amax(counts)/counts
        self.class_weights = dict(zip(classes, counts))

    def _load_model(self, model):
        """ Load an existing model structure and weights """
        if os.path.isfile(model):
            self.model = load_model(model)
        else:
            raise IOError('No model file found for {}...'.format(model))

    def _load_weights(self, weights):
        """ Load trained model weights; assumes default model structure """
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
        y = np.asarray(y)
        assert len(y.shape) == 2
        if 0 <= smooth_factor <= 1:
            y *= 1 - smooth_factor
            y += smooth_factor / y.shape[1]
        else:
            raise ValueError(
                    'Invalid label smoothing factor: {}'.format(smooth_factor))
        return y

    def evaluate(self, images, labels, batch_size=32, verbose=0):
        """
        Evaluate accuracy and loss of model predictions

        Parameters
        ----------
        images : numpy.ndarray
            Array of grayscale, normalized images to be used for evaluation.
            Input shape = (n_image,n_row,n_cols,1)
        labels : numpy.ndarray
            Array of labels to be used for evaluation, shape=(n_images,)
        batch_size : int
            Batch size to use for predictions (default: 32)
        verbose : int
            Verbosity mode to use, 0 or 1 (default: 0)

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
        """
        Print summary of currently loaded model
        """
        print(self.model.summary())

    def predict(self, images, batch_size=32, verbose=0):
        """
        Predict classifications for an array of images

        Parameters
        ----------
        images : numpy.ndarray
            Array of grayscale, normalized images to be used for class
            predictions. Input shape = (n_image,n_row,n_cols,1)
        batch_size : int
            Batch size to use for predictions (default: 32)
        verbose : int
            Verbosity mode to use, 0 or 1 (default: 0)

        Returns
        -------
        numpy.ndarray
            Array containing class probabilities for each image. The array
            output is ordered as follows: [n_images, p(worm), p(spot),
            p(track), p(noise)]

        """
        return self.model.predict(images, batch_size=batch_size,
                                  verbose=verbose)

    def train(self, train_images, train_labels, test_images, test_labels,
              batch_size=32, seed=None, epochs=10, initial_epoch=0,
              smooth_factor=None, check_point=True,
              check_point_weights_only=True, horizontal_flip=True,
              vertical_flip=True, width_shift_range=0.08,
              height_shift_range=0.08, rotation_range=180.,
              zoom_range=[0.9, 1.1], fill_mode="constant", cval=0,
              shuffle=True, save_model=None, save_weights=None,
              save_history=None, output_dir=None):
        """
        Train CNN

        Parameters
        ----------
        train_images : numpy.ndarray
            Array of grayscale, normalized images to be used for training the
            CNN. Input shape = (n_image,n_row,n_cols,1)
        train_labels : numpy.ndarray
            Array of training labels, shape=(n_images,)
        test_images : numpy.ndarray
            Array of grayscale, normalized images to be used for testing the
            CNN. Input shape = (n_image,n_row,n_cols,1)
        test_labels : numpy.ndarray
            Array of testing labels, shape=(n_images,)
        batch_size : int
            Number of samples per gradient update (default: 32)
        seed : int
            Random seed to be used for reproducibility. (default: None)
        epochs : int
            Number of epochs to train the model. Note that in conjunction with
            initial_epoch, the parameter epochs is to be understood as
            "final epoch". (default: 10)
        initial_epoch : int
            Epoch at which to start training (useful for resuming a previous
            training run). (default: 0)
        smooth_factor : float between 0,1
            Level of smoothing to apply to one-hot label vector. Ex.
            smooth_factor of 0.004 applied to [0, 1, 0, 0], results in
            [0.001, 0.997, 0.001, 0.001]. (default: None)
        check_point : bool
            If True, saves a running copy of the model corresponding to the
            lowest validation loss epoch. Each time a new low is reached, the
            previous best model is over-written by the new one. Model saved as
            'best_checkpointed_model.h5'. (default: True)
        check_point_weights_only : bool
            If True, then only the model's weights will be saved, else the
            full model is saved. Ignored if check_point = False.
            (default: True)
        horizontal_flip : bool
            Randomly flip inputs horizontally. (default: True)
        vertical_flip : bool
            Randomly flip inputs vertically. (default: True)
        width_shift_range : Float (fraction of total width)
            Range for random horizontal shifts. (default: 0.08)
        height_shift_range : Float (fraction of total height)
            Range for random vertical shifts. (default: 0.08)
        rotation_range : int
            Degree range for random rotations (default: 180)
        zoom_range : float or [lower, upper].
            Range for random zoom. If a float, [lower, upper] = [1-zoom_range,
            1+zoom_range] (default: [0.9, 1.1])
        fill_mode : One of {"constant", "nearest", "reflect" or "wrap"}
            Points outside the boundaries of the input are filled according to
            the given mode. (default: "constant")
        cval : float or int
            Value used for interpolated pixels when fill_mode = "constant".
            (default: 0)
        shuffle : bool
            Whether to shuffle the order of the batches at the beginning of
            each epoch. (default: True)
        save_model : str
            If specified, a copy of the model from the final training epoch
            will be saved. ex. save_model='my_model.h5'. Typically used for
            continued training (default: None)
        save_weights : str
            If specified, a copy of the model weights from the final training
            epoch will be saved. ex. save_weights='my_weights.h5'.
            (default: None)
        save_history : str
            If specified, the training history (accuracy and loss for training
            and testing) from each epoch will be saved to a '.csv'.
            ex. save_history='my_history.csv'. (default: None)
        output_dir : str
            If specified, all model outputs will be saved to the specified
            directory. (default: current working directory)
        """
        if not self.training:
            raise ValueError('CNN class initialized with training=\'False\','
                             'must be \'True\'')

        if seed:
            np.random.seed(seed)

        # Set output directory
        if not output_dir:
            output_dir = os.getcwd()

        # Calculate class weights
        self._calculate_class_weights(train_labels)

        # Convert labels to one-hot and apply smoothing
        train_labels = to_categorical(train_labels, self.n_classes)
        if smooth_factor:
            train_labels = self._smooth_labels(train_labels, smooth_factor)
        test_labels = to_categorical(test_labels, self.n_classes)

        # Setup checkpointer
        checkpointer = None
        if check_point:
            filepath = '{}/best_checkpointed_model.h5'.format(output_dir)
            checkpointer = ModelCheckpoint(
                            filepath, monitor='val_loss', verbose=0,
                            save_weights_only=check_point_weights_only,
                            save_best_only=True, mode='auto')

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
        hist = self.model.fit_generator(datagen.flow(
                        train_images,
                        train_labels,
                        batch_size=batch_size,
                        seed=seed,
                        shuffle=shuffle),
                        steps_per_epoch=train_images.shape[0] // batch_size,
                        epochs=epochs,
                        class_weight=self.class_weights,
                        validation_data=(test_images, test_labels),
                        callbacks=[checkpointer],
                        initial_epoch=initial_epoch)

        # Evaluate  model
        score = self.model.evaluate(test_images, test_labels, verbose=0)
        print('Final test loss:', score[0])
        print('Final test accuracy:', score[1])

        # Save model, weights, history
        if save_model:
            self.model.save('{}/{}'.format(output_dir, save_model))
        if save_weights:
            self.model.save_weights('{}/{}'.format(output_dir, save_weights))
        if save_history:
            hist_vals = np.array([hist.history['acc'],
                                  hist.history['val_acc'],
                                  hist.history['loss'],
                                  hist.history['val_loss']])
            np.savetxt('{}/{}'.format(output_dir, save_history),
                       np.transpose(hist_vals),
                       delimiter=',',
                       header='acc,val_acc,loss,val_loss')

    def train_with_kfold(self, images, labels, k_folds=10, seed=None,
                         shuffle=True, batch_size=32, epochs=10,
                         initial_epoch=0, smooth_factor=None, check_point=True,
                         check_point_weights_only=True, horizontal_flip=True,
                         vertical_flip=True, width_shift_range=0.08,
                         height_shift_range=0.08, rotation_range=180.,
                         zoom_range=[0.9, 1.1], fill_mode="constant", cval=0,
                         save_model=None, save_weights=None, save_history=None,
                         output_dir=None):

        """
        Train CNN using kfold cross validation

        Parameters
        ----------
        images : numpy.ndarray
            Array of grayscale, normalized images to be used for training and
            testing the CNN. shape = (n_image,n_row,n_cols,1)
        labels : numpy.ndarray
            Array of labels to be used for training and testing the CNN.
            shape=(n_images,)
        k_folds : int
            Number of folds to use for k-fold cross validation. (default: 10)
        seed : int
            Random seed to be used for reproducibility. (default: None)
        batch_size : int
            Number of samples per gradient update (default: 32)
        epochs : int
            Number of epochs to train the model. Note that in conjunction with
            initial_epoch, the parameter epochs is to be understood as
            "final epoch". (default: 10)
        initial_epoch : int
            Epoch at which to start training (useful for resuming a previous
            training run). (default: 0)
        smooth_factor : float between 0,1
            Level of smoothing to apply to one-hot label vector. Ex.
            smooth_factor of 0.004 applied to [0, 1, 0, 0], results in
            [0.001, 0.997, 0.001, 0.001]. (default: None)
        check_point : bool
            If True, saves a running copy of the model corresponding to the
            lowest validation loss epoch. Each time a new low is reached, the
            previous best model is over-written by the new one. Model saved
            as 'best_checkpointed_model_k.h5', where k is the current fold
            being trained. (default: True)
        check_point_weights_only : bool
            If True, then only the model's weights will be saved, else the
            full model is saved. Ignored if check_point = False.
            (default: True)
        horizontal_flip : bool
            Randomly flip inputs horizontally. (default: True)
        vertical_flip : bool
            Randomly flip inputs vertically. (default: True)
        width_shift_range : Float (fraction of total width)
            Range for random horizontal shifts. (default: 0.08)
        height_shift_range : Float (fraction of total height)
            Range for random vertical shifts. (default: 0.08)
        rotation_range : int
            Degree range for random rotations (default: 180)
        zoom_range : float or [lower, upper].
            Range for random zoom. If a float, [lower, upper] = [1-zoom_range,
            1+zoom_range] (default: [0.9, 1.1])
        fill_mode : One of {"constant", "nearest", "reflect" or "wrap"}
            Points outside the boundaries of the input are filled according to
            the given mode. (default: "constant")
        cval : float or int
            Value used for interpolated pixels when fill_mode = "constant".
            (default: 0)
        shuffle : bool
            Whether to shuffle the order of the batches at the beginning of
            each epoch. (default: True)
        save_model : str
            If specified, a copy of the model from the final training epoch
            will be saved. ex. save_model='my_model.h5' will save models as
            'my_model_k.h5', where k is the current fold being trained.
            Typically used for continued training (default: None)
        save_weights : str
            If specified, a copy of the model weights from the final training
            epoch will be saved. ex. save_weights='my_weights.h5' will save
            weights as 'my_weights_k.h5', where k is the current fold being
            trained. (default: None)
        save_history : str
            If specified, the training history (accuracy and loss for training
            and testing) from each epoch will be saved to a '.csv'. ex.
            save_history='my_history.csv' will save history as
            'my_history_k.csv', where k is the current fold being trained.
            (default: None)
        output_dir : str
            If specified, all model outputs will be saved to the specified
            directory. (default: current working directory)
        """
        if not self.training:
            raise ValueError('CNN class initialized with training=\'False\','
                             'must be \'True\'')

        if seed:
            np.random.seed(seed)

        # Set output directory
        if not output_dir:
            output_dir = os.getcwd()

        # Calculate class weights
        self._calculate_class_weights(labels)

        # Store initial weights
        init_weights = self.model.get_weights()

        # Split into testing and training sets
        skf = StratifiedKFold(n_splits=k_folds, shuffle=shuffle,
                              random_state=seed)
        skf.get_n_splits(images, labels)

        for i, indices in enumerate(skf.split(images, labels)):
            print('Training Fold Number {}'.format(i))

            # Get testing and training split for this fold
            train_index, test_index = indices
            train_images, test_images = images[train_index], images[test_index]
            train_labels, test_labels = labels[train_index], labels[test_index]
            self.train_indices.append(train_index)
            self.test_indices.append(test_index)

            # Convert labels to one-hot and apply smoothing
            train_labels = to_categorical(train_labels, self.n_classes)
            if smooth_factor:
                train_labels = self._smooth_labels(train_labels, smooth_factor)
            test_labels = to_categorical(test_labels, self.n_classes)

            # Setup checkpointer
            checkpointer = None
            if check_point:
                filepath = os.path.join(
                        output_dir, 'best_checkpointed_model_{}.h5'.format(i))
                checkpointer = ModelCheckpoint(
                                filepath, monitor='val_loss', verbose=0,
                                save_weights_only=check_point_weights_only,
                                save_best_only=True, mode='auto')

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
            hist = self.model.fit_generator(datagen.flow(
                        train_images,
                        train_labels,
                        batch_size=batch_size,
                        seed=seed,
                        shuffle=shuffle),
                        steps_per_epoch=train_images.shape[0]//batch_size,
                        epochs=epochs,
                        class_weight=self.class_weights,
                        validation_data=(test_images, test_labels),
                        callbacks=[checkpointer],
                        initial_epoch=initial_epoch)

            # Evaluate  model
            score = self.model.evaluate(test_images, test_labels, verbose=0)
            print('Final test loss:', score[0])
            print('Final test accuracy:', score[1])

            # Save model, weights, history
            if save_model:
                split_str = save_model.split('.')
                self.model.save('{}/{}_{}.{}'.format(output_dir,
                                                     split_str[0], i,
                                                     split_str[-1]))
            if save_weights:
                split_str = save_weights.split('.')
                self.model.save_weights('{}/{}_{}.{}'.format(output_dir,
                                                             split_str[0],
                                                             i, split_str[-1]))
            if save_history:
                split_str = save_history.split('.')
                hist_vals = np.array([hist.history['acc'],
                                      hist.history['val_acc'],
                                      hist.history['loss'],
                                      hist.history['val_loss']])
                np.savetxt('{}/{}_{}.{}'.format(output_dir,
                                                split_str[0], i,
                                                split_str[-1]),
                           np.transpose(hist_vals),
                           delimiter=',',
                           header='acc,val_acc,loss,val_loss')

            self.model.set_weights(init_weights)

        self.train_indices = np.asarray(self.train_indices)
        self.test_indices = np.asarray(self.test_indices)
