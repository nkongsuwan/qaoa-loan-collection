import pytest
import numpy as np
from src.loanee_graph import LoaneeGraph
from src.qaoa_analytics import QaoaAnalytics
from src.helpers import generate_random_dataset


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

# 5 loanees, 4 actions
e_54, a_54 = generate_random_dataset(5,4)

config = {
    "epsilon_constant": 0.5,
    "qaoa_repetition": 2,
}

l = LoaneeGraph(e, a)
l_54 = LoaneeGraph(e_54, a_54)
qaoa_54 = QaoaAnalytics(l_54, config)


def test_qaoa_analytics():
    qaoa_analytics = QaoaAnalytics(l, config)
    qaoa_analytics.optimize_qaoa_params()


def test_invalid_input():
    with pytest.raises(Exception):
        qaoa_analytics = QaoaAnalytics({}, config)

    with pytest.raises(Exception):
        invalid_config = config.copy()
        invalid_config["epsilon_constant"] = -1
        qaoa_analytics = QaoaAnalytics(l, invalid_config)

    with pytest.raises(Exception):
        invalid_config = config.copy()
        invalid_config["qaoa_repetition"] = 0
        qaoa_analytics = QaoaAnalytics(l, invalid_config)

    with pytest.raises(Exception):
        invalid_config = config.copy()
        invalid_config["numpy_seed"] = -1
        qaoa_analytics = QaoaAnalytics(l, invalid_config)


def test_prepare_equal_superposition_of_valid_states():
    qaoa_analytics = QaoaAnalytics(l, config)
    wavefunction = qaoa_analytics._prepare_equal_superposition_of_valid_states()

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

    wavefunction_54 = qaoa_54._prepare_equal_superposition_of_valid_states()
    assert wavefunction_54.shape == tuple([4]*5) #(4,4,4,4,4)


def test_evolve():
    with pytest.raises(Exception):
        initial_qaoa_params = np.array([0.73, 0.33])
        qaoa_54._QaoaAnalytics__evolve_wavefunc(initial_qaoa_params)

    initial_qaoa_params = np.array([0.73, 0.33, 0.45, 0.23])
    qaoa_54._QaoaAnalytics__evolve_wavefunc(initial_qaoa_params)


def test_optimize():
    qaoa_analytics = QaoaAnalytics(l, config)
    initial_params = rng.random(2*config["qaoa_repetition"]) 
    result = qaoa_analytics.optimize_qaoa_params(initial_params)

    # Test if qaoa_analytics.__log is consistent with SciPy result
    assert qaoa_analytics.scipy_result.fun == result.get_optimized_cost()
    assert np.array_equal(qaoa_analytics.scipy_result.x, result.get_optimized_params())
