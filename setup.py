'''
decotools
Author: James Bourbeau + Mr. Meehan
'''

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
    install_requires=['numpy', 'pandas']
)
