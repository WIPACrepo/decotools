
import numpy as np
from PIL import Image
from skimage import io
from skimage import measure


def get_image_array(image_file, as_grey=True):

    image = io.imread(image_file)
    if as_grey:
        image = image[:,:,:3].sum(axis=2)

    # img = Image.open(image_file)
    # if as_grey:
    #     img = img.convert('L')
    # image = []
    # pix = img.load()
    #
    # # Stuff image values into a 2D array called 'image'
    # nx, ny = img.size[0], img.size[1]
    # x0, y0, x1, y1 = (0, 0, nx, ny)
    # for y in xrange(ny):
    #     image.append([pix[x, y] for x in xrange(nx)])
    #
    # image = np.asarray(image, dtype=float)

    return image


class Blob(object):
    '''Class that defines a 'blob' in an image: the contour of a set of pixels
       with values above a given threshold.'''

    def __init__(self, x, y):
        '''Define a counter by its contour lines (an list of points in the xy
           plane), the contour centroid, and its enclosed area.'''
        self.x = np.array(x)
        self.y = np.array(y)
        self.xc = np.mean(x)
        self.yc = np.mean(y)
        self._length = x.shape[0]

        # Find the area inside the contour
        area = 0
        for i in range(self._length):
            area += 0.5*(y[i]+y[i-1])*(x[i]-x[i-1])
        self.area = area

    def length(self):
        ''' Find the approx length of the blob from the max points of the
            contour. '''
        xMin = self.x.min()
        xMax = self.x.max()
        yMin = self.y.min()
        yMax = self.y.max()
        len_ = np.sqrt( (xMin - xMax)**2 + (yMin - yMax)**2 )
        return len_

    def distance(self, blob):
        '''Calculate the distance between the centroid of this blob contour and
           another one in the xy plane.'''
        return np.sqrt((self.xc - blob.xc)**2 + (self.yc-blob.yc)**2)


def findblobs(image, threshold, min_area=2., max_area=1000.):
    '''Pass through an image and find a set of blobs/contours above a set
       threshold value.  The min_area parameter is used to exclude blobs with an
       area below this value.'''
    blobs = []
    ny, nx = image.shape

    # Find contours using the Marching Squares algorithm in the scikit package.
    contours = measure.find_contours(image, threshold)
    for contour in contours:
        x = contour[:,1]
        y = ny - contour[:,0]
        blob = Blob(x, y)
        if blob.area >= min_area and blob.area <= max_area:
            blobs.append(blob)

    return blobs


def extract_blobs(image_file, threshold=20., min_area=10., max_area=1000.,
                  max_dist=150.):

    image = get_image_array(image_file, as_grey=True)

    # Calculate contours using the scikit-image marching squares algorithm,
    # store as Blobs, and group the Blobs into associated clusters
    blobs = findblobs(image, threshold=threshold,
                      min_area=min_area, max_area=max_area)
    groups = groupBlobs(blobs, max_dist=max_dist)

    return blobs
