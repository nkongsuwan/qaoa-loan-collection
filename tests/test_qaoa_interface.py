import pytest
import numpy as np
from src.loanee_graph import LoaneeGraph
from src.qaoa_interface import QaoaInterface
from src.enums import InitialState


class QaoaTest(QaoaInterface):
   
    def __init__(self, loanees: LoaneeGraph, qaoa_config: dict):
        super().__init__(loanees, qaoa_config)

    def _run_qaoa(self, initial_qaoa_params: np.ndarray):
        pass

    def _prepare_initial_state(self):
        pass


qaoa_config = {
    "epsilon_constant": 0.5,
    "qaoa_repetition": 2,
}

qiskit_config = {
    "simulator": "qasm_simulator"
}

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
l = LoaneeGraph(e, a)


def test_initializing():
    qaoa = QaoaTest(l, qaoa_config)
    qaoa._num_loanees == 2
    qaoa._num_actions == 3
    qaoa._num_qubits == 2 * 3
    qaoa._num_valid_states == 3 ** 2


def test_optimize():
    qaoa = QaoaTest(l, qaoa_config)
    qaoa.optimize_qaoa_params(np.array([1,2,3]))


def test_initial_state():
    config = qaoa_config.copy()
    config["initial_state"] = "ground"
    qaoa = QaoaTest(l, config)
    assert qaoa._initial_state == InitialState.GROUND

    config = qaoa_config.copy()
    config["initial_state"] = "default"
    qaoa = QaoaTest(l, config)
    assert qaoa._initial_state == InitialState.DEFAULT

    config = qaoa_config.copy()
    config["initial_state"] = "equal_superposition"
    qaoa = QaoaTest(l, config)
    assert qaoa._initial_state == InitialState.EQUAL_SUPERPOSITION

    config = qaoa_config.copy()
    config["initial_state"] = "equal_superposition_of_valid_states"
    qaoa = QaoaTest(l, config)
    assert qaoa._initial_state == InitialState.EQUAL_SUPERPOSITION_OF_VALID_STATES


def test_invalid_input():
    with pytest.raises(Exception):
        qaoa = QaoaTest({}, qaoa_config)

    with pytest.raises(Exception):
        invalid_config = qaoa_config.copy()
        invalid_config["epsilon_constant"] = -1
        qaoa = QaoaTest(l, invalid_config)

    with pytest.raises(Exception):
        invalid_config = qaoa_config.copy()
        invalid_config["qaoa_repetition"] = 0
        qaoa = QaoaTest(l, invalid_config)

    with pytest.raises(Exception):
        invalid_config = qaoa_config.copy()
        invalid_config["optimizer_method"] = 13
        qaoa = QaoaTest(l, invalid_config)

    with pytest.raises(Exception):
        invalid_config = qaoa_config.copy()
        invalid_config["optimizer_maxiter"] = "123"
        qaoa = QaoaTest(l, invalid_config)     
    
    with pytest.raises(Exception):
        invalid_config = qaoa_config.copy()
        invalid_config["optimizer_maxiter"] = 0
        qaoa = QaoaTest(l, invalid_config)   

    with pytest.raises(Exception):
        invalid_config = qaoa_config.copy()
        invalid_config["numpy_seed"] = -1
        qaoa = QaoaTest(l, invalid_config)

    with pytest.raises(Exception):
        invalid_config = qaoa_config.copy()
        invalid_config["initial_state"] = 1
        qaoa = QaoaTest(l, invalid_config)

    with pytest.raises(Exception):
        invalid_config = qaoa_config.copy()
        invalid_config["initial_state"] = "xyz"
        qaoa = QaoaTest(l, invalid_config)
