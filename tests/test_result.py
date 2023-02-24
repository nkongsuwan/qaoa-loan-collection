import pytest
import numpy as np

from src.result import ResultQaoa

def test_result():
    result = ResultQaoa()
    result.append(1  , np.array([0.2, 0.4]))
    result.append(99 , np.array([0.9, 0.6]))
    result.append(0.2, np.array([1.1, 3.3]))
    result.append(0.5, np.array([1.0, 5  ]))
    
    assert result._ResultQaoa__len_result == len(result._ResultQaoa__log)
    assert result.get_len() == result._ResultQaoa__len_result
    assert result.get_len() == 4

    assert result.get_cost_with_index(0) == 1
    assert result.get_cost_with_index(1) == 99
    assert result.get_cost_with_index(2) == 0.2
    assert result.get_cost_with_index(3) == 0.5
    assert np.array_equal(result.get_params_with_index(0), np.array([0.2, 0.4]))
    assert np.array_equal(result.get_params_with_index(1), np.array([0.9, 0.6]))
    assert np.array_equal(result.get_params_with_index(2), np.array([1.1, 3.3]))
    assert np.array_equal(result.get_params_with_index(3), np.array([1.0, 5  ]))


def test_invalid_input():
    
    with pytest.raises(Exception):
        result = ResultQaoa()
        result.append(0.1, np.array([]))    
    
    with pytest.raises(Exception):
        result = ResultQaoa()
        result.append(0.1, [0.1, 0.2])

    with pytest.raises(Exception):
        result = ResultQaoa()
        result.append("abc", np.array([0.1, 0.2]))

    with pytest.raises(Exception):
        result = ResultQaoa()
        result.append("abc", np.array(["123", "456"]))

    with pytest.raises(Exception):
        result = ResultQaoa()
        result.append(0.1, np.array([0.2, 0.3]))
        result.append(0.1, np.array([0.2, 0.3, 0.4, 0.5]))

    with pytest.raises(Exception):
        result = ResultQaoa()
        result.append(0.1, np.array([0.2, 0.3, 0.5]))


def test_invalid_indices():
    result = ResultQaoa()
    result.append(1  , np.array([0.2, 0.4]))
    result.append(99 , np.array([0.9, 0.6]))

    with pytest.raises(Exception):
        result.get_cost_with_index(0.5)
        result.get_cost_with_index(1.0)
        result.get_cost_with_index(-1)
        result.get_cost_with_index(3)

    with pytest.raises(Exception):
        result.get_params_with_index(0.5)
        result.get_params_with_index(1.0)
        result.get_params_with_index(-1)
        result.get_params_with_index(3)