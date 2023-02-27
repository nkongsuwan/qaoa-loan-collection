import abc
import numpy as np

from src.loanee_graph import LoaneeGraph
from src.result import Result

# Abstract QAOA class
class ProblemInterface(metaclass=abc.ABCMeta):
    def __init__(self, loanees: LoaneeGraph, qaoa_config: dict):
        
        assert isinstance(loanees, LoaneeGraph)
        self._num_loanees = loanees.get_num_loanees()
        self._num_actions = loanees.get_num_actions()

        # set default values for instance variables
        self._epsilon = 0.1
        self._p       = 1
        self._optimizer_method  = "COBYLA"
        self._optimizer_maxiter = 50
        self.__initialize_instance_variables_with_config(qaoa_config)

    def __initialize_instance_variables_with_config(self, qaoa_config: dict):
        if "epsilon_constant" in qaoa_config:
            assert isinstance(qaoa_config["epsilon_constant"], float)
            assert qaoa_config["epsilon_constant"] >= 0
            self._epsilon = qaoa_config["epsilon_constant"]

        if "qaoa_repetition" in qaoa_config:
            assert isinstance(qaoa_config["qaoa_repetition"], int)
            assert qaoa_config["qaoa_repetition"] > 0
            self._p = qaoa_config["qaoa_repetition"]
        
        if "optimizer_method" in qaoa_config:
            assert isinstance(qaoa_config["optimizer_method"], str)
            self._optimizer_method = qaoa_config["optimizer_method"]

        if "optimizer_maxiter" in qaoa_config:
            assert isinstance(qaoa_config["optimizer_maxiter"], int)
            self._optimizer_maxiter = qaoa_config["optimizer_maxiter"]       

    @abc.abstractmethod
    def optimize_qaoa_params(self):
        pass

    @abc.abstractmethod   
    def _calculate_cost(self, qaoa_params: np.ndarray, result: Result):
        pass