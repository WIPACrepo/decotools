.. _api:

:github_url: https://github.com/WIPACrepo/decotools

*************
Decotools API
*************

========
File I/O
========

There are two functions used to retrieve DECO data: ``get_iOS_files`` and ``get_android_files``.

.. autofunction:: decotools.fileio_iOS.get_iOS_files

.. autofunction:: decotools.fileio_android.get_android_files


===============
Blob Extraction
===============
.. currentmodule:: decotools.blob_extraction

Finding interesting structure in images is done using the ``extract_blobs`` function.

.. autofunction:: extract_blobs


=======
Metrics
=======
.. currentmodule:: decotools.metrics

Functions to calculate intensity metrics and histogram images are in the ``metrics`` module.

.. autofunction:: get_rgb_hists

.. autofunction:: get_intensity_metrics
