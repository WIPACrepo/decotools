#!/usr/bin/env python

DISTNAME = 'decotools'
DESCRIPTION = 'Python tools for analyzing DECO data'
MAINTAINER = 'DECO team'
URL = 'https://wipacrepo.github.io/decotools/'
LICENSE = 'MIT'
LONG_DESCRIPTION = '''Python tools for analyzing DECO data

Please refer to the online documentation at
https://wipacrepo.github.io/decotools/
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

with open('requirements/default.txt') as fid:
    INSTALL_REQUIRES = [l.strip() for l in fid.readlines() if l]

EXTRAS_REQUIRE = {
    'tf': ['tensorflow'],
    'tf-gpu': ['tensorflow-gpu'],
}

setup(
    name=DISTNAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    url=URL,
    author=MAINTAINER,
    license=LICENSE,
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE
)
