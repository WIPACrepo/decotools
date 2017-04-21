# decotools

A set of tools to help make DECO life easier :sparkles:

## Installation

The recommended installation command on cobalt machines is given by:

```bash
pip install --user -e /path/to/decotools
```

The `--user` option will install decotools into your .local python site-packages directory. The `-e` option tells pip to install decotools in "editable" mode. This will place a link to the decotools package in your .local python site-packages directory.

If you haven't already, make sure that your .local python site-packages directory is in your system `PYTHONPATH` environment variable. This can be done by adding the following lines to your shell .rc file (e.g. .bashrc for those using bash and .zshrc for those using zsh):
```bash
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$HOME/.local/lib/pythonX.Y/site-packages:$PYTHONPATH"
```
where here `pythonX.Y` should be replaced with whatever version of python you are using (e.g. python2.7, python3.6, etc).
