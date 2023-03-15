import pytest
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from src.loanee_graph import LoaneeGraph
from src.qaoa_qiskit import QaoaQiskit
from src.helpers import generate_random_dataset


qaoa_config = {
    "epsilon_constant": 0.5,
    "qaoa_repetition": 2,
}

qiskit_config = {
    "simulator": "qasm_simulator"
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

qaoa_23 = QaoaQiskit(l_23, qaoa_config, qiskit_config)
qaoa_54 = QaoaQiskit(l_54, qaoa_config, qiskit_config)


def test_qaoa_qiskit():
    qaoa_23.optimize_qaoa_params()
    qaoa_54.optimize_qaoa_params()

'''
def test_prepare_inital_state():
    #wavefunc = qaoa_23._prepare_initial_state()
    #assert isinstance(wavefunc, QuantumCircuit)
    
    #initial_op = Operator(wavefunc).to_matrix()

    #num = 2**6
    #qc = QuantumCircuit(num)
    #for i in range(num):
    #    qc.h(i)
    #test_op = Operator(qc).to_matrix()

    #assert initial_op.shape == test_op.shape
    #assert np.array_equal(initial_op, test_op)


    qc = QuantumCircuit(2)
    qc.h(0)
    qc.x(1)
    op = Operator(qc)
    m = op.to_matrix()
    assert True
'''

'''
def test_evolve():
    with pytest.raises(Exception):
        initial_qaoa_params = np.array([0.73, 0.33])
        qaoa_54._QaoaAnalytics__evolve_wavefunc(initial_qaoa_params)

    initial_qaoa_params = np.array([0.73, 0.33, 0.45, 0.23])
    qaoa_54._QaoaAnalytics__evolve_wavefunc(initial_qaoa_params)


def test_optimize():
    qaoa_analytics = QaoaAnalytics(l, qaoa_config)
    initial_params = rng.random(2*qaoa_config["qaoa_repetition"]) 
    result = qaoa_analytics.optimize_qaoa_params(initial_params)

    # Test if qaoa_analytics.__log is consistent with SciPy result
    assert qaoa_analytics.scipy_result.fun == result.get_optimized_cost()
    assert np.array_equal(qaoa_analytics.scipy_result.x, result.get_optimized_params())

'''