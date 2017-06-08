# Welcome to the decotools documentation!

**This documentation page is still under construction**.

Decotools is a python package for analyzing [DECO data](https://wipac.wisc.edu/deco/home).

DECO is a citizen science project that enables users around the world to detect cosmic rays and other energetic particles with their cell phones and tablets. The recorded events are automatically uploaded to a central database. In addition to detecting particle events, users can analyze the data produced by their own or other users’ phones.


## Useful links
* [DECO homepage](https://wipac.wisc.edu/deco/home)
* [GitHub repository](https://github.com/mattmeehan/decotools)
* [Contributing guide](contributing)


## Installation

The `decotools` Python package can be installed directly from GitHub via

```bash
$ pip install git+https://github.com/mattmeehan/decotools#egg=decotools
```

For more information on how `pip` interacts with version control systems (like GitHub) see [this section of the pip documentation](https://pip.pypa.io/en/stable/reference/pip_install/#vcs-support).


**Note for contributers:**

There are extra dependencies for developers (e.g. `pytest` for running test, `mkdocs` for building the documentation, etc.). These can be installed via

```bash
$ pip install -r requirements-dev.txt
```
