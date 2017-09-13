.. _api:
.. module:: decotools

:github_url: https://github.com/WIPACrepo/decotools

*************
Decotools API
*************

========
File I/O
========

There are two functions used to retrieve DECO data: ``get_iOS_files`` and ``get_android_files``.

.. autofunction:: fileio_iOS.get_iOS_files

.. autofunction:: fileio_android.get_android_files


===============
Blob Extraction
===============

Finding interesting structure in images is done using the ``extract_blobs`` function.

.. autofunction:: blob_extraction.extract_blobs
