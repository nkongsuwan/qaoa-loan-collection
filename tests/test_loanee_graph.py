import pytest
import numpy as np
from src.loanee_graph import LoaneeGraph

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

def test_simple_instance():
    LoaneeGraph(e, a)

def test_invalid_instances():

    with pytest.raises(Exception):
        LoaneeGraph([[1,2],[3,4]], [[1,2],[3,4]])
    
    with pytest.raises(Exception):
        LoaneeGraph(e, [[1,2],[3,4]])
    
    with pytest.raises(Exception):
        LoaneeGraph([[1,2],[3,4]], a)
    
    with pytest.raises(Exception):
        LoaneeGraph(
            e,
            np.array(
                [
                    [
                        [1,0],
                        [0,1]
                    ],
                    [
                        [1,1],
                        [0,0]
                    ]
                ]
            ) 
        )
        
        with pytest.raises(Exception):
            LoaneeGraph(e, np.array([[1,1],[1,1]]))
