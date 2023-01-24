import numpy as np
import scipy
from src.loanee_graph import LoaneeGraph

# TO-DO 
#   from src.qaoa_interface import QaoaInterface
#   class QaoaAnalytics(QaoaInterface):
class QaoaAnalytics():
    def __init__(self, loanee_graph, qaoa_config):

        assert isinstance(loanee_graph, LoaneeGraph)
        assert isinstance(qaoa_config["epsilon_constant"], float)
        assert isinstance(qaoa_config["qaoa_repetition"], int)
        assert isinstance(qaoa_config["numpy_seed"], int)
        assert qaoa_config["epsilon_constant"] >= 0
        assert qaoa_config["qaoa_repetition"] > 0
        assert qaoa_config["numpy_seed"] >= 0

        self.loanees = loanee_graph
        self.epsilon = qaoa_config["epsilon_constant"],
        self.p       = qaoa_config["qaoa_repetition"]
        self.rng     = np.random.default_rng(qaoa_config["numpy_seed"])

        # Coefficients for calculating the cost function
        #self.h       = (1 - self.epsilon)*np.copy(self.loanees.expected_net_profit_matrix)
        #self.h[:,0] += - self.epsilon*np.sum(self.loanees.association_matrix, axis=1)
        #self.J       = self.epsilon*np.copy(self.loanees.association_matrix)

        #self.final_state = 
        #self.qaoa_results   = []

    def optimize(self, method="COBYLA", maxiter=50):
        qaoa_variational_params = self.rng.random(2*self.p)    
        self.qaoa_results = scipy.minimize(
            self._calculate_cost, 
            qaoa_variational_params, 
            method=method, 
            options={
                "disp": False,
                "maxiter": maxiter
            }
        )

    def _calculate_cost(self, params):
        assert len(params) == 2*self.p

        wavefunction = self._evolve_wavefunction(params)
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
    # instead of 2**(num_actions*num_loanees)
    def _evolve_wavefunction(self, qaoa_variational_params):
        wavefunction = self.__prepare_equal_superposition_of_valid_states()
        
        '''
        for t in range(self.p):
            self._apply_U_A(params[2*t])
            self._apply_U_B(params[2*t+1])
        '''

        return wavefunction

    # A valid state is a state where only one action is taken for each loanee.
    '''
    def __prepare_equal_superposition_of_valid_states(self):
        num_valid_states = self.loanees.num_actions**self.loanees.num_loanees
        wavefunction = np.ones(num_valid_states,dtype='complex') / np.sqrt(num_valid_states)
        wavefunction = np.reshape(wavefunction,[self.loanees.num_actions]*self.loanees.num_loanees)
        return wavefunction
    '''