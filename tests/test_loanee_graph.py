from src.loanee_graph import LoaneeGraph

def test_loanee_graph():

    expected_net_profit_matrix = np.array(
        [
            [0.01045035, 0.03135106, 0.02090071],
            [0.01045035, 0.03135106, 0.02090071]
        ]
    )

    association_matrix = np.array(
        [
            [0.        , 0.09405319],
            [0.09405319, 0.        ]
        ]
    )

    loanees = LoaneeGraph(expected_net_profit_matrix, association_matrix)