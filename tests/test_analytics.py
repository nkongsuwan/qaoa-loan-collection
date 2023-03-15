import pytest
import numpy as np
from src.loanee_graph import LoaneeGraph
from src.qaoa_analytics import QaoaAnalytics
from src.helpers import generate_random_dataset


qaoa_config = {
    "epsilon_constant": 0.5,
    "qaoa_repetition": 2,
}

e_23 = np.array(
    [
        [0.01045035, 0.03135106, 0.02090071],
        [0.01045035, 0.03135106, 0.02090071]
    ]
)

a_23 = np.array(
    [
        [0.        , 0.09405319],
        [0.09405319, 0.        ]
    ]
)

# 5 loanees, 4 actions
e_54, a_54 = generate_random_dataset(5,4)

l_23 = LoaneeGraph(e_23, a_23)
l_54 = LoaneeGraph(e_54, a_54)

qaoa_23 = QaoaAnalytics(l_23, qaoa_config)
qaoa_54 = QaoaAnalytics(l_54, qaoa_config)


def test_qaoa_analytics():
    qaoa_23.optimize_qaoa_params()
    qaoa_54.optimize_qaoa_params()


def test_prepare_initial_state():
    
    wavefunction = qaoa_23._prepare_initial_state()
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

    wavefunction_54 = qaoa_54._prepare_initial_state()
    assert wavefunction_54.shape == tuple([4]*5) #(4,4,4,4,4)


def test_evolve():
    with pytest.raises(Exception):
        initial_qaoa_params = np.array([0.73, 0.33])
        qaoa_54._QaoaAnalytics__evolve_wavefunc(initial_qaoa_params)

    initial_qaoa_params = np.array([0.73, 0.33, 0.45, 0.23])
    qaoa_54._QaoaAnalytics__evolve_wavefunc(initial_qaoa_params)


def test_optimize():
    rng = np.random.default_rng(12345)

    qaoa_analytics = QaoaAnalytics(l_23, qaoa_config)
    initial_params = rng.random(2*qaoa_config["qaoa_repetition"]) 
    result = qaoa_analytics.optimize_qaoa_params(initial_params)

    # Test if qaoa_analytics.__log is consistent with SciPy result
    assert qaoa_analytics.scipy_result.fun == result.get_optimized_cost()
    assert np.array_equal(qaoa_analytics.scipy_result.x, result.get_optimized_params())