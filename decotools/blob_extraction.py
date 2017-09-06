from __future__ import division
import numpy as np
import pandas as pd
from skimage import io, measure
from PIL import Image
from collections import Counter, Iterable
import dask
from dask import delayed, multiprocessing
from dask.diagnostics import ProgressBar


def get_image_array(image_file, rgb_sum=False):
    '''Returns an image array

    Parameters
    ----------
    image_file : str
        Path to image file.
    rgb_sum : bool, optional
        Whether to use a simple RGB sum to convert to image to grayscale, or
        to use a weighted RGB sum (default: False).

    Returns
    -------
    numpy.ndarray
        Grayscale image array

    '''

    if rgb_sum:
        image = io.imread(image_file)
        # Convert RGB image to R+G+B grayscale
        image = np.sum(image[:, :, :-1], axis=2)

    else:
        img = Image.open(image_file)
        # Convert grayscale using a weighted RGB sum
        # From PIL documentation the weighted sum is given by
        # grayscale = R * 299/1000 + G * 587/1000 + B * 114/1000
        img = img.convert('L')
        # Convert img to a numpy array
        image = np.asarray(img, dtype=float).T

    return image


class Blob(object):
    '''Class that defines a 'blob' in an image: the contour of a set of pixels
       with values above a given threshold.
    '''

    def __init__(self, x, y):
        '''Define a counter by its contour lines (an list of points in the xy
           plane), the contour centroid, and its enclosed area.
        '''
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

    def __repr__(self):
        str_rep = 'Blob(xc={}, yc={}, area={})'.format(self.xc, self.yc, self.area)
        return str_rep

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
    nx, ny = image.shape

    # Find contours using the Marching Squares algorithm in the scikit package.
    contours = measure.find_contours(image, threshold)
    for contour in contours:
        y = contour[:,1]
        x = contour[:,0]
        blob = Blob(x, y)
        if blob.area >= min_area and blob.area <= max_area:
            blobs.append(blob)

    return blobs


class BlobGroup(object):
    '''A list of blobs that is grouped or associated in some way, i.e., if
       their contour centroids are relatively close together.'''

    def __init__(self, image):
        '''Initialize a list of stored blobs and the bounding rectangle which
        defines the group.'''
        self.blobs = []
        self.xmin =  1e10
        self.xmax = -1e10
        self.ymin =  1e10
        self.ymax = -1e10
        self.image = image
        self.xc = None
        self.yc = None

    def __repr__(self):
        str_rep = 'BlobGroup(n_blobs={}, xc={}, yc={})'.format(len(self.blobs), self.xc, self.yc)
        return str_rep

    def add_blob(self, blob):
        '''Add a blob to the group and enlarge the bounding rectangle of the
           group.'''
        self.blobs.append(blob)
        self.xmin = min(self.xmin, blob.x.min())
        self.xmax = max(self.xmax, blob.x.max())
        self.ymin = min(self.ymin, blob.y.min())
        self.ymax = max(self.ymax, blob.y.max())
        self.xc = np.mean([blob.xc for blob in self.blobs])
        self.yc = np.mean([blob.yc for blob in self.blobs])

    def get_bounding_box(self):
        '''Get the bounding rectangle of the group.'''
        return (self.xmin, self.xmax, self.ymin, self.ymax)

    def get_square_bounding_box(self):
        '''Get the bounding rectangle, redefined to give it a square aspect
           ratio.'''
        xmin, xmax, ymin, ymax = (self.xmin, self.xmax, self.ymin, self.ymax)
        xL = np.abs(xmax - xmin)
        yL = np.abs(ymax - ymin)
        if xL > yL:
            ymin -= 0.5*(xL-yL)
            ymax += 0.5*(xL-yL)
        else:
            xmin -= 0.5*(yL-xL)
            xmax += 0.5*(yL-xL)
        return (xmin, xmax, ymin, ymax)

    def get_sub_image(self, image=None, size=None):
        '''Given an image, extract the section of the image corresponding to
           the bounding box of the blob group.'''

        if image is None:
            image = self.image.copy()

        nx, ny = image.shape
        if size is None:
            x0,x1,y0,y1 = self.get_square_bounding_box()
        else:
            xc, yc = self.xc, self.yc
            if isinstance(size, Iterable):
                size_x, size_y = size
            else:
                size_x = size_y = size
            x0, x1 = xc - size_x, xc + size_x
            y0, y1 = yc - size_y, yc + size_y

        # Account for all the weird row/column magic in the image table...
        i0, i1 = int(x0), int(x1)
        j0, j1 = int(y0), int(y1)

        # Add a pixel buffer around the bounds, and check the ranges
        buf = 1
        i0 = 0 if i0-buf < 0 else i0-buf
        i1 = nx-1 if i1 > nx-1 else i1+buf
        j0 = 0 if j0-buf < 0 else j0-buf
        j1 = ny-1 if j1 > ny-1 else j1+buf

        return image[i0:i1, j0:j1]


    def get_region_props(self, threshold, size=None):

        subimage = self.get_sub_image(size=size)
        labeled_image = subimage >= threshold
        region_properties = measure.regionprops(labeled_image.astype(int), subimage)

        if len(region_properties) == 0:
            return {}
        elif len(region_properties) > 1:
            raise ValueError('Image has more than one region!')

        return region_properties[0]


def group_blobs(image, blobs, max_dist):
    '''Given a list of blobs, group them by distance between the centroids of
       any two blobs.  If the centroids are more distant than max_dist, create a
       new blob group.'''
    n = len(blobs)
    groups = []
    if n >= 1:
        # Single-pass clustering algorithm: make the first blob the nucleus of
        # a blob group.  Then loop through each blob and add either add it to
        # this group (depending on the distance measure) or make it the
        # nucleus of a new blob group
        bg = BlobGroup(image=image)
        bg.add_blob(blobs[0])
        groups.append(bg)

        for i in range(1, n):
            bi = blobs[i]
            is_grouped = False
            for group in groups:
                # Calculate distance measure for a blob and a blob group:
                # blob just has to be < max_dist from any other blob in the group
                for bj in group.blobs:
                    if bi.distance(bj) < max_dist:
                        group.add_blob(bi)
                        is_grouped = True
                        break
            if not is_grouped:
                bg = BlobGroup(image=image)
                bg.add_blob(bi)
                groups.append(bg)

    return np.asarray(groups, dtype=object)


def extract_blobs(image_file, threshold=20., rgb_sum=False, min_area=10.,
                  max_area=1000., max_dist=5., group_max_area=None, size=None):
    '''Function to perform blob detection on an input image

    Blobs are found using the marching squares algorithm implemented in
    scikit-image.

    Parameters
    ----------
    image_file : str
        Path to image file.
    threshold : float, optional
        Threshold for blob detection. Only pixels with an intensity above
        this threshold will be used in blob detection (default: 20).
    rgb_sum : bool, optional
        Whether to use a simple RGB sum to convert to image to grayscale, or
        to use a weighted RGB sum (default: False).
    min_area : float, optional
        Minimum area for a blob to be kept. This helps get rid of noise in
        an image (default: 10).
    max_area : float, optional
        Maximum area for a blob to be kept. This helps get rid of pathological
        events in an image (default: 1000).
    max_dist : float, optional
        Distance scale for grouping close by blobs. If two blobs are separated
        by less than max_dist, they are grouped together as a single blob
        (defualt: 5).
    group_max_area : float, optional
        Maximum area for a blob group to be kept. This helps get rid of pathological
        events in an image (default: None).
    size : {None, int, array-like of shape=(2,)}, optional
        Size of zoomed image of extracted blobs. If an integer is provided, the
        zoomed image will be a square box of size 2*size in each dimension. If
        an array-like object (of shape=(2,)) is provided, then the zoomed image
        will be of size 2*size[0] by 2*size[1]. Otherwise, the default behavior
        is to return a square image of size twice the equivalent diameter of
        the blob.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing information about the found blobs is returned.
        Each row in the DataFrame corresponds to a blob group, while each
        column corresponds to a pertinent quanitity (area, eccentricity,
        zoomed image array, etc.).

    '''
    image = get_image_array(image_file, rgb_sum=rgb_sum)
    if image.ndim != 2:
        return pd.DataFrame()

    # Calculate contours using the scikit-image marching squares algorithm,
    # store as Blobs, and group the Blobs into associated clusters
    blobs = findblobs(image, threshold=threshold,
                      min_area=min_area, max_area=max_area)
    groups = group_blobs(image, blobs, max_dist=max_dist)

    group_properties = []
    for group_idx, group in enumerate(groups):
        region_props = group.get_region_props(threshold, size=size)
        prop_dict = {property_:region_props[property_] for property_ in region_props}
        prop_dict['n_blobs'] = len(group.blobs)
        prop_dict['n_groups'] = len(groups)
        prop_dict['blob_idx'] = group_idx
        prop_dict['xc'] = group.xc
        prop_dict['yc'] = group.yc
        if size is None:
            size = prop_dict['equivalent_diameter']
        prop_dict['image'] = group.get_sub_image(size=size)
        prop_dict['image_file'] = image_file

        if group_max_area and prop_dict['area'] > group_max_area:
            continue

        group_properties.append(prop_dict)

    region_prop_df = pd.DataFrame.from_records(group_properties)

    return region_prop_df


def is_hotspot(x_coords, y_coords, threshold=3, radius=4.0):
    '''Function to identify hot spot from a list of x-y coordinates

    Parameters
    ----------
    x_coords : array-like
        X-coordinates of blob groups. Note: x_coords and y_coords must have
        the same shape.
    y_coords : array-like
        Y-coordinates of blob groups. Note: y_coords and x_coords must have
        the same shape.
    threshold : int, optional
        Threshold number of counts to classify an x-y coordinate pair as a hot
        spot. If a (x, y) coordinate pair occurs threshold or more times,
        it is considered a hot spot (default is 3).
    radius : float, optional
        If an x-y pair is within radius number of pixels of a hot spot, it
        is also considered a hot spot.

    Returns
    -------
    is_hot_spot : numpy.ndarray
        Boolean array that specifies whether or not a blob group is a hot spot.

    '''

    # Cast to numpy arrays for vectorizing distance computation later on
    x_coords = np.asarray(x_coords)
    y_coords = np.asarray(y_coords)
    # Check that x_coords and y_coords are compatiable
    if not x_coords.shape == y_coords.shape:
        raise ValueError('x_coords and y_coords must have the same shape.')

    # Get number of times each x-y pixel combination occurs
    centers_list = [(int(x), int(y)) for x, y in zip(x_coords, y_coords)]
    coord_counter = Counter(centers_list)
    # Get hot spot coordinates based on number of times an x-y coordinate repeats
    hotspots_coords = []
    for coord, count in coord_counter.items():
        if count >= threshold:
            hotspots_coords.append(coord)

    def get_distances(x1, y1, x2, y2):
        return np.sqrt( (x1-x2)**2 + (y1-y2)**2 )

    # Get mask for events within radius of hot spot
    is_hot_spot = np.zeros(len(x_coords), dtype=bool)
    for x_hot, y_hot in hotspots_coords:
        distances = get_distances(x_coords, y_coords, x_hot, y_hot)
        is_hot_spot = np.logical_or(is_hot_spot, (distances <= radius))

    return is_hot_spot


def _get_image_intensity(image_file, rgb_sum=False):
    '''Function to calculate intensity metrics

    Parameters
    ----------
    image_file : str
        Image file to be analyzed.
    rgb_sum : bool, optional
        Option to use simple RGB sum for grayscale conversion (default is to
        use weighted RGB sum).

    Returns
    -------
    intensity_dict : dict
        Dictionary with intensity metrics

    '''
    image = get_image_array(image_file, rgb_sum=rgb_sum)
    intensity_dict = {'mean': image.mean(), 'max': image.max()}
    for percentile in [16, 50, 84]:
        key = 'percentile_{}'.format(percentile)
        intensity_dict[key] = np.percentile(image, percentile)

    return intensity_dict


def get_intensity_metrics(files, rgb_sum=False, n_jobs=1):
    '''Calculates various metrics related to the image intensity

    Parameters
    ----------
    files : str, sequence
        Image file path (or sequence of file paths) to be analyzed.
    rgb_sum : bool, optional
        Option to use simple RGB sum for grayscale conversion (default is to
        use weighted RGB sum).
    n_jobs : int, optional
        The number of jobs to run in parallel (default is 1).

    Returns
    -------
    image_intensities : pandas.DataFrame
        DataFrame with intensity metrics

    '''
    if isinstance(files, str):
        files = [files]

    image_intensities = [delayed(_get_image_intensity)(f) for f in files]
    image_intensities = delayed(pd.DataFrame.from_records)(image_intensities)

    with ProgressBar() as bar:
        get = dask.get if n_jobs == 1 else multiprocessing.get
        image_intensities = image_intensities.compute(get=get,
                                                      num_workers=n_jobs)

    return image_intensities
