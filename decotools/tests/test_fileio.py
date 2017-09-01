
import pytest
from decotools.fileio_iOS import get_iOS_files


def test_no_events_or_min_bias_fail():
    with pytest.raises(ValueError) as excinfo:
        get_iOS_files(include_events=False, include_min_bias=False)
    error = 'At least one of include_events or include_min_bias must be True.'
    assert error == str(excinfo.value)


def test_invalid_start_date_fail():
    with pytest.raises(ValueError) as excinfo:
        get_iOS_files(start_date='jamesbond')
    error = 'Invalid start_date or end_date entered'
    assert error == str(excinfo.value)


def test_invalid_end_date_fail():
    with pytest.raises(ValueError) as excinfo:
        get_iOS_files(end_date='notadate')
    error = 'Invalid start_date or end_date entered'
    assert error == str(excinfo.value)


def test_verbose_not_int_fail():
    with pytest.raises(ValueError) as excinfo:
        verbose = '0'
        get_iOS_files(verbose=verbose)
    error = 'Expecting an int for verbose, got {}'.format(type(verbose))
    assert error == str(excinfo.value)
