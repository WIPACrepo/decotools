
import numpy as np
import pandas as pd
from PIL import Image
from skimage import measure


def get_image_array(image_file, greyscale=True):

    img = Image.open(image_file)
    if greyscale:
        img = img.convert('L')
    image = []
    pix = img.load()

    # Stuff image values into a 2D array called 'image'
    nx, ny = img.size[0], img.size[1]
    x0, y0, x1, y1 = (0, 0, nx, ny)
    for y in xrange(ny):
        if y != 0:
            y = -y+ny
        image.append([pix[x, y] for x in xrange(nx)])

    image = np.asarray(image, dtype=float)

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

    def __repr__(self):
        str_rep = 'Blob(x={}, y={}, area={})'.format(self.xc, self.yc, self.area)
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

    def __repr__(self):
        str_rep = 'BlobGroup(n_blobs={})'.format(len(self.blobs))
        return str_rep

    def add_blob(self, blob):
        '''Add a blob to the group and enlarge the bounding rectangle of the
           group.'''
        self.blobs.append(blob)
        self.xmin = min(self.xmin, blob.x.min())
        self.xmax = max(self.xmax, blob.x.max())
        self.ymin = min(self.ymin, blob.y.min())
        self.ymax = max(self.ymax, blob.y.max())
        self.cov  = None

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

    def get_sub_image(self, image=None, square=True):
        '''Given an image, extract the section of the image corresponding to
           the bounding box of the blob group.'''

        if image is None:
            image = self.image.copy()
        ny,nx = image.shape
        if square:
            x0,x1,y0,y1 = self.get_square_bounding_box()
        else:
            x0,x1,y0,y1 = self.get_bounding_box()

        # Account for all the weird row/column magic in the image table...
        i0,i1 = [ny - int(t) for t in (y1,y0)]
        j0,j1 = [int(t) for t in (x0,x1)]

        # Add a pixel buffer around the bounds, and check the ranges
        buf = 1
        i0 = 0 if i0-buf < 0 else i0-buf
        i1 = ny-1 if i1 > ny-1 else i1+buf
        j0 = 0 if j0-buf < 0 else j0-buf
        j1 = nx-1 if j1 > nx-1 else j1+buf

        return image[i0:i1, j0:j1]

    def get_raw_moment(self, image, p, q):
        '''Calculate the image moment given by
           M_{ij}=\sum_x\sum_y x^p y^q I(x,y)
           where I(x,y) is the image intensity at location x,y.'''
        nx,ny = image.shape
        Mpq = 0.
        if p == 0 and q == 0:
            Mpq = np.sum(image)
        else:
            for i in range(0,nx):
                x = 0.5 + i
                for j in range(0,ny):
                    y = 0.5 + j
                    Mpq += x**p * y**q * image[i,j]
        return Mpq

    def get_hu_moments(self, image):
        """Calcuate the Hu moments, which are translation, scale, and
        rotation invariant using skimage.measure.

        Additionally calculate an 8th moment to complete the set as
        described here:
        https://en.wikipedia.org/wiki/Image_moment#Raw_moments
        """

        subImage = self.get_sub_image(image).transpose()
        moments = measure.moments(subImage)
        centroid_row = moments[0, 1] / moments[0, 0]
        centroid_column = moments[1, 0] / moments[0, 0]
        moments_central = measure.moments_central(image,
                          centroid_row, centroid_column)
        moments_normalized = measure.moments_normalized(moments_central)
        moments_hu = measure.moments_hu(moments_normalized)

        # Calculate the 8th Hu moment
        mn = moments_normalized
        hu_8 = mn[1][1]*((mn[3][0]+mn[1][2])**2-(mn[0][3]+mn[2][1])**2)\
               - (mn[2][0]-mn[0][2])*(mn[3][0]+mn[1][2])*(mn[0][3]+mn[2][1])
        moments_hu = np.append(moments_hu,hu_8)
        return moments_hu

    def get_region_props(self, threshold, square=True):

        subimage = self.get_sub_image(square=square)
        labeled_image = subimage >= threshold
        region_properties = measure.regionprops(labeled_image.astype(int), subimage)
        if len(region_properties) > 1:
            raise ValueError('Image has more than one region!')

        return region_properties[0]

    def get_max_intensity(self, image):
        '''Find the maximum intensity within the blob
           where I(x,y) is the image intensity at location x,y.'''
        nx,ny = image.shape
        maxI = 0.
        for i in range(0,nx):
            for j in range(0,ny):
                if image[i,j] > maxI: maxI = image[i,j]
        return maxI

    # Not sure if this works
    def get_area(self, image):
        '''Calculate image area from image moment 00.'''
        M00 = -999
        subImage = self.get_sub_image(image).transpose()
        M00 = self.get_raw_moment(subImage, 0, 0)
        return M00

    def get_covariance(self, image):
        '''Get the raw moments of the image region inside the bounding box
           defined by this blob group and calculate the image covariance
           matrix.'''
        if self.cov == None:
            subImage = self.get_sub_image(image).transpose()
            M00 = self.get_raw_moment(subImage, 0, 0)
            M10 = self.get_raw_moment(subImage, 1, 0)
            M01 = self.get_raw_moment(subImage, 0, 1)
            M11 = self.get_raw_moment(subImage, 1, 1)
            M20 = self.get_raw_moment(subImage, 2, 0)
            M02 = self.get_raw_moment(subImage, 0, 2)
            xbar = M10/M00
            ybar = M01/M00
            self.cov = np.vstack([[M20/M00 - xbar*xbar, M11/M00 - xbar*ybar],
                                  [M11/M00 - xbar*ybar, M02/M00 - ybar*ybar]])
        return self.cov

    def get_principal_moments(self, image):
        '''Return the maximum and minimum eigenvalues of the covariance matrix,
           as well as the angle theta between the maximum eigenvector and the
           x-axis.'''
        cov = self.get_covariance(image)
        u20 = cov[0,0]
        u11 = cov[0,1]
        u02 = cov[1,1]

        theta = 0.5 * np.arctan2(2*u11, u20-u02)
        l1 = 0.5*(u20+u02) + 0.5*np.sqrt(4*u11**2 + (u20-u02)**2)
        l2 = 0.5*(u20+u02) - 0.5*np.sqrt(4*u11**2 + (u20-u02)**2)
        return l1, l2, theta


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
            isGrouped = False
            for group in groups:
                # Calculate distance measure for a blob and a blob group:
                # blob just has to be < max_dist from any other blob in the group
                for bj in group.blobs:
                    if bi.distance(bj) < max_dist:
                        group.add_blob(bi)
                        isGrouped = True
                        break
            if not isGrouped:
                bg = BlobGroup(image=image)
                bg.add_blob(bi)
                groups.append(bg)

    return np.asarray(groups, dtype=object)


def extract_blobs(image_file, threshold=20., min_area=10., max_area=200.,
                  max_dist=20., greyscale=True, square=True):

    image = get_image_array(image_file, greyscale=greyscale)

    # Calculate contours using the scikit-image marching squares algorithm,
    # store as Blobs, and group the Blobs into associated clusters
    blobs = findblobs(image, threshold=threshold,
                      min_area=min_area, max_area=max_area)
    groups = group_blobs(image, blobs, max_dist=max_dist)

    data = []
    for group in groups:
        region_props = group.get_region_props(threshold, square=square)
        prop_dict = {property_:region_props[property_] for property_ in region_props}
        prop_dict['n_blobs'] = len(group.blobs)
        prop_dict['image'] = group.get_sub_image(square=square).T
        data.append(prop_dict)

    region_prop_df = pd.DataFrame.from_records(data)

    return region_prop_df
