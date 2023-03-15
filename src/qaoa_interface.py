import abc
import numpy as np

from src.loanee_graph import LoaneeGraph
from src.result import ResultQaoa
from src.enums import InitialState

# Abstract QAOA class
class QaoaInterface(metaclass=abc.ABCMeta):
    def __init__(self, loanees: LoaneeGraph, qaoa_config: dict):
        
        assert isinstance(loanees, LoaneeGraph)
        self._num_loanees = loanees.get_num_loanees()
        self._num_actions = loanees.get_num_actions()
        self._num_qubits = self._num_loanees * self._num_actions
        self._num_valid_states = self._num_actions**self._num_loanees

        # set default values for instance variables
        self._epsilon = 0.2
        self._p       = 1
        self._rng    = np.random.default_rng(12345)
        self._optimizer_method  = "COBYLA"
        self._optimizer_maxiter = 1000     
        self._initial_state = InitialState.EQUAL_SUPERPOSITION
        self.__initialize_instance_variables_with_qaoa_config(qaoa_config)

    def __initialize_instance_variables_with_qaoa_config(self, qaoa_config: dict):
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
            assert qaoa_config["optimizer_maxiter"] > 0
            self._optimizer_maxiter = qaoa_config["optimizer_maxiter"]       

        if "numpy_seed" in qaoa_config:
            assert isinstance(qaoa_config["numpy_seed"], int)
            assert qaoa_config["numpy_seed"] >= 0
            self._rng = np.random.default_rng(qaoa_config["numpy_seed"])

        if "initial_state" in qaoa_config:
            assert isinstance(qaoa_config["initial_state"], str)
            assert qaoa_config["initial_state"] in InitialState._value2member_map_
            for state in InitialState._value2member_map_:
                if qaoa_config["initial_state"] == state:
                    self._initial_state = InitialState._value2member_map_[state]


    def optimize_qaoa_params(
        self, 
        initial_qaoa_params: np.ndarray = None
    ) -> ResultQaoa:
    
        if initial_qaoa_params is None:
            initial_qaoa_params = self._rng.random(2*self._p)    
        
        return self._run_qaoa(initial_qaoa_params)


    @abc.abstractmethod
    def _run_qaoa(self, initial_qaoa_params: np.ndarray) -> ResultQaoa:
        pass


    @abc.abstractmethod
    def _prepare_initial_state(self):
        pass

    #@abc.abstractmethod   
    #def _calculate_cost(self, qaoa_params: np.ndarray, result: ResultQaoa):
    #    pass