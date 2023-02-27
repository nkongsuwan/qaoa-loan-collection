import numpy as np


class LoaneeGraph:

    def __init__(self, expected_net_profit_matrix: np.ndarray, association_matrix: np.ndarray):

        assert isinstance(expected_net_profit_matrix, np.ndarray)
        assert isinstance(association_matrix, np.ndarray)
        assert len(expected_net_profit_matrix.shape) == 2
        assert len(association_matrix.shape) == 2
        assert np.trace(association_matrix) == 0

        self.__expected_net_profit_matrix = expected_net_profit_matrix
        self.__association_matrix = association_matrix
        self.__num_loanees = expected_net_profit_matrix.shape[0]
        self.__num_actions = expected_net_profit_matrix.shape[1]
        #self.__num_qubits  = np.power(2, self.num_loanees*self.num_actions)


    def get_expected_net_profit_matrix(self):
        return self.__expected_net_profit_matrix
    

    def get_association_matrix(self):
        return self.__association_matrix


    def get_num_loanees(self):
        return self.__num_loanees


    def get_num_actions(self):
        return self.__num_actions