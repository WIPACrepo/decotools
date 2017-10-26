
import numpy as np
from skimage.io import imsave
import py


def save_test_images(tmpdir, n_images, shape=(10, 10)):
    '''Function to generate and save fake test images

    Parameters
    ----------
    tmpdir : pytest tmpdir
        A tmpdir fixture from pytest. See
        https://docs.pytest.org/en/latest/tmpdir.html for more information.
    n_images : int
        Number of test images to create.
    shape : array-like, shape=(2, ), optional
        Shape of test image. The first and second elements of shape will be
        the number of x and y pixels in the test image (default is (10, 10)).

    Returns
    -------
    files : list
        List of saved test image paths.
    '''
    if not isinstance(tmpdir, py._path.local.LocalPath):
        raise TypeError('tmpdir should be a tmpdir fixture from pytest')
    if not len(shape) == 2:
        raise ValueError(
            'Shape should only have two items, found {}'.format(len(shape)))
    # Create and save a test image to a temporary file
    nx, ny = shape
    files = []
    for i in range(n_images):
        tmpfile = tmpdir.join('test_image_{}.png'.format(i))
        imsave(str(tmpfile), np.random.random((nx, ny, 4)))
        files.append(str(tmpfile))

    return files
