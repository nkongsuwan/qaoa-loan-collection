import numpy as np
import scipy
from src.loanee_graph import LoaneeGraph
from src.result import ResultQaoa

# TO-DO 
#   from src.qaoa_interface import QaoaInterface
#   class QaoaAnalytics(QaoaInterface):
class QaoaAnalytics():

    def __init__(self, loanees: LoaneeGraph, qaoa_config: dict):

        assert isinstance(loanees, LoaneeGraph)
        self.__loanees = loanees

        # set default values for instance variables
        self.__epsilon = 0.1
        self.__p       = 1
        self.__rng     = np.random.default_rng(12345)
        self.__optimizer_method  = "COBYLA"
        self.__optimizer_maxiter = 50
        
        self.__initialize_instance_variables_with_config(qaoa_config)

        # Coefficients for calculating the cost function
        #self.h       = (1 - self.epsilon)*np.copy(self.loanees.get_expected_net_profit_matrix())
        #self.h[:,0] += - self.epsilon*np.sum(self.loanees.get_association_matrix(), axis=1)
        #self.J       = self.epsilon*np.copy(self.loanees.get_association_matrix())

    def __initialize_instance_variables_with_config(self, qaoa_config: dict):
        if "epsilon_constant" in qaoa_config:
            assert isinstance(qaoa_config["epsilon_constant"], float)
            assert qaoa_config["epsilon_constant"] >= 0
            self.__epsilon = qaoa_config["epsilon_constant"]

        if "qaoa_repetition" in qaoa_config:
            assert isinstance(qaoa_config["qaoa_repetition"], int)
            assert qaoa_config["qaoa_repetition"] > 0
            self.__p = qaoa_config["qaoa_repetition"]
        
        if "numpy_seed" in qaoa_config:
            assert isinstance(qaoa_config["numpy_seed"], int)
            assert qaoa_config["numpy_seed"] >= 0
            self.__rng = np.random.default_rng(qaoa_config["numpy_seed"])
        
        if "optimizer_method" in qaoa_config:
            assert isinstance(qaoa_config["optimizer_method"], str)
            self.__optimizer_method = qaoa_config["optimizer_method"]

        if "optimizer_maxiter" in qaoa_config:
            assert isinstance(qaoa_config["optimizer_maxiter"], str)
            self.__optimizer_maxiter = qaoa_config["optimizer_maxiter"]

    '''
    def optimize_qaoa_params(self, initial_qaoa_params: np.ndarray = None):
    
        if initial_qaoa_params is None:
            initial_qaoa_params = self.rng.random(2*self.p)    
        
        scipy_result = scipy.minimize(
            self._calculate_cost, 
            initial_qaoa_params, 
            method = self.__optimizer_method, 
            options = {
                "disp": False,
                "maxiter": self.__optimizer_maxiter
            }
        )

        result_qaoa = ResultQaoa()
        return result_qaoa
    '''

    def _calculate_cost(self, qaoa_params: np.ndarray):
        assert len(qaoa_params) == 2*self.__p

        cost = 0.0
        wavefunction = self.__evolve_wavefunction(qaoa_params)

        return cost
    '''  
        psi_bra = np.copy(np.conj(self.psi))
        c = self.c0

        for i in range(self.N): 
            psi_ket = np.copy(self.psi)
            c += self.inner_product(psi_bra, self._apply_h_B_onsite(psi_ket,i))                      
            for ii in range(i):
                if self.J[i,ii] != 0:
                    psi_ket = np.copy(self.psi)
                    c += self.inner_product(psi_bra, self._apply_h_B_coupling(psi_ket,i,ii))
        
        self.costs += [np.real(c)]
        return np.real(c)
    '''

    # Evolve a wavefunction using a QAOA circuit with the given QAOA variational parameters (betas, gammas)
    # Here, a reduced Hibert space is used to describe the wavefunction,
    # i.e. dimension of the wavefunction is num_actions**num_loanees
    # instead of 2**(num_actions*num_loanees).
    def __evolve_wavefunction(self, qaoa_params: np.ndarray):
        wavefunction = self.__prepare_equal_superposition_of_valid_states()
        
        for i in range(self.__p):
            wavefunction = self.__apply_U_A(wavefunction, qaoa_params[2*i])
            wavefunction = self.__apply_U_B(wavefunction, qaoa_params[2*i + 1])

        return wavefunction

    # A valid state is a state where only one action is taken for each loanee.
    # Returned wavefunction has a dimension of [[num_actions]*num_loanees
    # e.g. for 5 loanees and 3 actions
    # wavefunction,shape == (3, 3, 3, 3, 3)
    def __prepare_equal_superposition_of_valid_states(self):
        num_actions = self.__loanees.get_num_actions()
        num_loanees = self.__loanees.get_num_loanees()
        
        num_valid_states = num_actions**num_loanees
        wavefunction = np.ones(num_valid_states, dtype='complex') / np.sqrt(num_valid_states)
        wavefunction = np.reshape(wavefunction,[num_actions]*num_loanees)

        return wavefunction