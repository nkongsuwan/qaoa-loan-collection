import numpy as np

class LoaneeGraph:
    def __init__(self, config):
        self.num_actions = config["number_of_actions"]
        self.num_loanees = config["number_of_loanees"]
        self.num_qubits  = np.power(2, self.num_actions * self.num_loanees)


    # "number_of_loanees": Qs.shape[0],
    # "number_of_actions": Qs.shape[1]

    def get_num_qubits(self):
        return self.num_qubits