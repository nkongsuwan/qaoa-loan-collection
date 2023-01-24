import numpy as np

class LoaneeGraph:
    def __init__(self, expected_net_profit_matrix, association_matrix):

        assert isinstance(expected_net_profit_matrix, np.ndarray)
        assert isinstance(association_matrix, np.ndarray)
        assert len(expected_net_profit_matrix.shape) == 2
        assert len(association_matrix.shape) == 2
        assert np.trace(association_matrix) == 0

        self.expected_net_profit_matrix = expected_net_profit_matrix
        self.association_matrix = association_matrix
        self.num_loanees = expected_net_profit_matrix.shape[0]
        self.num_actions = expected_net_profit_matrix.shape[1]
        self.num_qubits  = np.power(2, self.num_loanees*self.num_actions)
