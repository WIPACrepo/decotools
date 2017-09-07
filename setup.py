'''
decotools
Author: James Bourbeau + Mr. Meehan
'''

import sys
if sys.version_info[0] < 3:
    import __builtin__ as builtins
else:
    import builtins
builtins.__DECOTOOLS_SETUP__ = True

from setuptools import setup, find_packages
import decotools

VERSION = decotools.__version__

setup(
    name='decotools',
    version=VERSION,
    description='Python tools for deco',
    author='James Bourbeau + Mr. Meehan',
    author_email='jbourbeau@wisc.edu',
    packages=find_packages(),
    install_requires=['numpy', 'pandas', 'scikit-image', 'Pillow',
                      'matplotlib', 'dask[complete]']
)
