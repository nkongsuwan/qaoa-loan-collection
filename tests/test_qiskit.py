import pytest
import numpy as np
from src.loanee_graph import LoaneeGraph
from src.qaoa_analytics import QaoaAnalytics
from src.qaoa_qiskit import QaoaQiskit
from src.helpers import generate_random_dataset

'''
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
'''

def test_qaoa_analytics():
    #qaoa_analytics = QaoaQiskit()
    assert True

def test_consistency_with_QaoaAnalytics():
    assert True
