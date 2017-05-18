
import pytest
from decotools import fileio

def test_no_files():
    with pytest.raises(ValueError) as excinfo:
        fileio.get_iOS_files(include_events=False, include_min_bias=False)
    error = 'At least one of include_events or include_min_bias must be True.'
    assert error == str(excinfo.value)
