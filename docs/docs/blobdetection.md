# Blob detection

```python
def extract_blobs(image_file, threshold=20., min_area=10., max_area=200.,
                  max_dist=150., square=True):
```

Function to perform blob detection on an input image.

**Parameters**

* `image_file` : str

    Path to image file.

* `threshold` : float, optional

    Threshold for blob detection. Only pixels with an intensity above
    this threshold will be used in blob detection (default: 20).

* `min_area` : float, optional

    Minimum area for a blob to be kept. This helps get rid of noise in
    an image (default: 10).

* `max_area` : float, optional

    Maximum area for a blob to be kept. This helps get rid of pathological
    events in an image (default: 200).

* `max_dist` : float, optional

    Distance scale for grouping close by blobs. If two blobs are separated
    by less than max_dist, they are grouped together as a single blob
    (defualt: 150).

* `square` : bool, optional

    Whether or not the returned zoomed image on the blob is square or
    not (defualt: True).

**Returns**

* `pandas.DataFrame`

    A DataFrame containing information about the found blobs is returned.
    Each row in the DataFrame corresponds to a blob group, while each
    column corresponds to a pertinent quanitity (area, eccentricity,
    zoomed image array, etc.).
