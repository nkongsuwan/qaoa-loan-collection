import pytest
import numpy as np
from src.loanee_graph import LoaneeGraph
from src.qaoa_analytics import QaoaAnalytics

rng = np.random.default_rng(12345)

e = np.array(
    [
        [0.01045035, 0.03135106, 0.02090071],
        [0.01045035, 0.03135106, 0.02090071]
    ]
)

a = np.array(
    [
        [0.        , 0.09405319],
        [0.09405319, 0.        ]
    ]
)

config = {
    "epsilon_constant": 0.5,
    "qaoa_repetition": 2,
    "numpy_seed": 12345
}

l = LoaneeGraph(e, a)

def test_invalid_input():
    with pytest.raises(Exception):
        qaoa_analytics = QaoaAnalytics({}, config)

    with pytest.raises(Exception):
        invalid_config = config.copy()
        invalid_config["epsilon_constant"] = -1
        qaoa_analytics = QaoaAnalytics({}, invalid_config)

    with pytest.raises(Exception):
        invalid_config = config.copy()
        invalid_config["qaoa_repetition"] = 0
        qaoa_analytics = QaoaAnalytics({}, invalid_config)

    with pytest.raises(Exception):
        invalid_config = config.copy()
        invalid_config["numpy_seed"] = "abc"
        qaoa_analytics = QaoaAnalytics({}, invalid_config)

def test_qaoa_analytics():
    qaoa_analytics = QaoaAnalytics(l, config)

def test_prepare_equal_superposition_of_valid_states():
    qaoa_analytics = QaoaAnalytics(l, config)
    wavefunction = qaoa_analytics._QaoaAnalytics__prepare_equal_superposition_of_valid_states()

    psi = np.array(
        [
            [1+0j, 1+0j, 1+0j],
            [1+0j, 1+0j, 1+0j],
            [1+0j, 1+0j, 1+0j]
        ]
    )
    psi = psi / np.sqrt(9)

    assert wavefunction.shape == psi.shape
    assert np.array_equal(wavefunction, psi)

def test_cost():
    qaoa_analytics = QaoaAnalytics(l, config)
    params = rng.random(2*config["qaoa_repetition"]) 
    assert qaoa_analytics._calculate_cost(params) == 0.0