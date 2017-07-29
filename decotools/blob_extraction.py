
import numpy as np
import pandas as pd
from skimage import io, measure


def get_image_array(image_file):

    image = io.imread(image_file)
    # Convert RGB image to R+G+B grayscale
    image = np.sum(image[:, :, :-1], axis=2)

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

    def get_sub_image(self, image=None, square=True, size=None):
        '''Given an image, extract the section of the image corresponding to
           the bounding box of the blob group.'''

        if image is None:
            image = self.image.copy()

        nx, ny = image.shape
        if square and not size:
            x0,x1,y0,y1 = self.get_square_bounding_box()
        if square and size:
            xc, yc = self.xc, self.yc
            x0, x1 = xc - size, xc + size
            y0, y1 = yc - size, yc + size
        else:
            x0,x1,y0,y1 = self.get_bounding_box()

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


    def get_region_props(self, threshold, square=True):

        subimage = self.get_sub_image(square=square)
        labeled_image = subimage >= threshold
        region_properties = measure.regionprops(labeled_image.astype(int), subimage)

        if len(region_properties) > 1:
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


def extract_blobs(image_file, threshold=20., min_area=10., max_area=1000.,
                  max_dist=5., group_max_area=None, square=True):
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
    square : bool, optional
        Whether or not the returned zoomed image on the blob is square or
        not (defualt: True).

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing information about the found blobs is returned.
        Each row in the DataFrame corresponds to a blob group, while each
        column corresponds to a pertinent quanitity (area, eccentricity,
        zoomed image array, etc.).

    '''

    image = get_image_array(image_file)
    if image.ndim != 2:
        return pd.DataFrame()

    # Calculate contours using the scikit-image marching squares algorithm,
    # store as Blobs, and group the Blobs into associated clusters
    blobs = findblobs(image, threshold=threshold,
                      min_area=min_area, max_area=max_area)
    groups = group_blobs(image, blobs, max_dist=max_dist)

    group_properties = []
    for group_idx, group in enumerate(groups):
        region_props = group.get_region_props(threshold, square=square)
        prop_dict = {property_:region_props[property_] for property_ in region_props}
        prop_dict['n_blobs'] = len(group.blobs)
        prop_dict['n_groups'] = len(groups)
        prop_dict['blob_idx'] = group_idx
        prop_dict['xc'] = group.xc
        prop_dict['yc'] = group.yc
        prop_dict['image'] = group.get_sub_image(square=square,
                                 size=prop_dict['equivalent_diameter'])
        prop_dict['image_file'] = image_file

        if group_max_area and prop_dict['area'] > group_max_area:
            continue

        group_properties.append(prop_dict)

    region_prop_df = pd.DataFrame.from_records(group_properties)

    return region_prop_df
