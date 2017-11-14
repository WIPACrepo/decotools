# decotools

[![Build Status](https://travis-ci.org/WIPACrepo/decotools.svg?branch=master)](https://travis-ci.org/WIPACrepo/decotools)
![Python 2.7](https://img.shields.io/badge/python-2.7-blue.svg)
![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)

A set of tools to help make DECO life easier :sparkles:

## Installation

The `decotools` Python package can be installed directly from GitHub via

```bash
$ pip install git+https://github.com/WIPACrepo/decotools#egg=decotools
```

`decotools` uses the Theano backend within the [Keras deep learning library](https://keras.io/). To configure this backend, the `KERAS_BACKEND` environment variable should be set to `theano`. For example,

```bash
export KERAS_BACKEND=theano
```

For more information on Keras backends see https://keras.io/backend/

## Documentation

The documentation for `decotools` is available at https://wipacrepo.github.io/decotools/.

## Contributing

Contributions to `decotools` are welcome! Please see the [contributing guide](https://wipacrepo.github.io/decotools/contributing.html) for more information.
