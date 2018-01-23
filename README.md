# decotools

[![Build Status](https://travis-ci.org/WIPACrepo/decotools.svg?branch=master)](https://travis-ci.org/WIPACrepo/decotools)
![Python 2.7](https://img.shields.io/badge/python-2.7-blue.svg)
![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)

A set of tools to help make DECO life easier :sparkles:

## Installation

The `decotools` Python package can be installed directly from GitHub. `decotools` is built on [Google's Tensorflow](https://www.tensorflow.org/), which must be installed to use `decotools`. If `tensorflow` is already installed, then `decotools` can be installed from GitHub via

```bash
$ pip install git+https://github.com/WIPACrepo/decotools#egg=decotools
```

Alternatively, if `tensorflow` is _not_ installed, then the following commands can be used to install `tensorflow` along with `decotools`.

For installing the CPU version of `tensorflow`:
```bash
$ pip install git+https://github.com/WIPACrepo/decotools#egg=decotools[tf]
```

For installing the GPU version of `tensorflow`:
```bash
$ pip install git+https://github.com/WIPACrepo/decotools#egg=decotools[tf-gpu]
```

## Documentation

The documentation for `decotools` is available at https://wipacrepo.github.io/decotools/.

## Contributing

Contributions to `decotools` are welcome! Please see the [contributing guide](https://wipacrepo.github.io/decotools/contributing.html) for more information.
