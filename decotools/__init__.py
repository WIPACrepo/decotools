
import sys

# Check to see if decotools is being imported from the setup.py script
# If being loaded on setup, don't attempt to import full decotools package
# only the __version__ needs to be included in this case.
try:
    __DECOTOOLS_SETUP__
except NameError:
    __DECOTOOLS_SETUP__ = False

if __DECOTOOLS_SETUP__:
    sys.stderr.write(
        '\n***Partial import of decotools during the build process***\n')
else:
    from .fileio_iOS import get_iOS_files
    from .fileio_android import get_android_files
    from .blob_extraction import extract_blobs, is_hotspot
    from .metrics import get_intensity_metrics, get_rgb_hists

__version__ = '0.0.1'
