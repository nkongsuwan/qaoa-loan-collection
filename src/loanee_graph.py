class LoaneeGraph:
    def __init__(self, config):
        self.num_actions = config["number_of_actions"]
        self.num_loanees = config["number_of_loanees"]
        self.num_qubits  = np.power(2, self.num_actions * self.num_loanees)