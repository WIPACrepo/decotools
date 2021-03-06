.. decotools documentation master file, created by
   sphinx-quickstart on Tue Sep 12 16:22:27 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

:github_url: https://github.com/WIPACrepo/decotools

.. toctree::
   :maxdepth: 3
   :caption: User Guide
   :hidden:

   api
   contributing

.. toctree::
  :maxdepth: 1
  :caption: Useful links
  :hidden:

  decotools @ GitHub <https://github.com/WIPACrepo/decotools>
  Issue tracker <https://github.com/WIPACrepo/decotools/issues>


Decotools
=========

.. image:: https://travis-ci.org/WIPACrepo/decotools.svg?branch=master
    :target: https://travis-ci.org/WIPACrepo/decotools

.. image:: https://img.shields.io/badge/python-2.7-blue.svg
    :target: https://github.com/WIPACrepo/decotools

.. image:: https://img.shields.io/badge/python-3.6-blue.svg
    :target: https://github.com/WIPACrepo/decotools


Decotools is a python package for analyzing data collected by the `Distributed Electronic Cosmic-ray Observatory (DECO) <https://wipac.wisc.edu/deco/home>`_.

DECO is a citizen science project that enables users around the world to detect cosmic rays and other energetic particles with their cell phones and tablets. The recorded events are automatically uploaded to a central database. In addition to detecting particle events, users can analyze the data produced by their own or other users’ phones.


============
Installation
============


The ``decotools`` Python package can be installed directly from GitHub. ``decotools`` is built on `Google's Tensorflow <https://www.tensorflow.org/>`_, which must be installed to use ``decotools``. If ``tensorflow`` is already installed, then ``decotools`` can be installed from GitHub via

.. code-block:: bash

    $ pip install git+https://github.com/WIPACrepo/decotools#egg=decotools

Alternatively, if ``tensorflow`` is not installed, then the following commands can be used to install ``tensorflow`` along with ``decotools``.

For installing the CPU version of ``tensorflow``:

.. code-block:: bash

    $ pip install git+https://github.com/WIPACrepo/decotools#egg=decotools[tf]

For installing the GPU version of ``tensorflow``:

.. code-block:: bash

    $ pip install git+https://github.com/WIPACrepo/decotools#egg=decotools[tf-gpu]


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
