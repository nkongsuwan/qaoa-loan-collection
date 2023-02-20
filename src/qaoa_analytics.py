import numpy as np
import scipy
from functools import partial

from src.loanee_graph import LoaneeGraph
from src.result import ResultQaoa

# TO-DO 
#   from src.qaoa_interface import QaoaInterface
#   class QaoaAnalytics(QaoaInterface):
class QaoaAnalytics():

    def __init__(self, loanees: LoaneeGraph, qaoa_config: dict):

        assert isinstance(loanees, LoaneeGraph)
        self.__num_loanees = loanees.get_num_loanees()
        self.__num_actions = loanees.get_num_actions()       

        # set default values for instance variables
        self.__epsilon = 0.1
        self.__p       = 1
        self.__rng     = np.random.default_rng(12345)
        self.__optimizer_method  = "COBYLA"
        self.__optimizer_maxiter = 50
        self.__initialize_instance_variables_with_config(qaoa_config)

        # Initialize coefficients for problem and mixing hamiltonian
        self.__h       = (1 - self.__epsilon) * np.copy(loanees.get_expected_net_profit_matrix())
        self.__h[:,0] +=    - self.__epsilon  * np.sum (loanees.get_association_matrix(), axis=1)
        self.__J       =      self.__epsilon  * np.copy(loanees.get_association_matrix()) 

        # Initialize helper vectors for caluclating problem hamiltonian
        self.__vs = None
        self.__es = None
        self.__initialize_helper_vectors_for_problem_hamiltonian()


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


    # Initialize a tight-binding operator acting on each loanee
    def __initialize_helper_vectors_for_problem_hamiltonian(self):
        
        # Eigenstates  |E_k> = a^\dagger_k |0>
        self.__vs = np.exp(
            -1j * np.array(
                [
                    [
                        2 * np.pi * k * j / self.__num_actions for k in range(self.__num_actions)
                    ] for j in range(self.__num_actions)
                ]
            )
        )
        self.__vs = self.__vs / np.sqrt(self.__num_actions)
        
        # Eigenvalues e^{-iE_k} where E_k = 2*cos(k)
        self.__es = np.exp(
            -1j * 2 * np.cos(
                np.arange(self.__num_actions) * 2 * np.pi / self.__num_actions
            )
        )


    def optimize_qaoa_params(self, initial_qaoa_params: np.ndarray = None):
    
        if initial_qaoa_params is None:
            initial_qaoa_params = self.rng.random(2*self.p)    
        
        # To-do
        result_qaoa = ResultQaoa()

        scipy_result = scipy.minimize(
            partial(self._calculate_cost, result=result_qaoa), 
            initial_qaoa_params, 
            method = self.__optimizer_method, 
            options = {
                "disp": False,
                "maxiter": self.__optimizer_maxiter
            }
        )

        return result_qaoa


    def _calculate_cost(self, qaoa_params: np.ndarray, result: ResultQaoa):
        assert len(qaoa_params) == 2*self.__p
        wavefunc = self.__evolve_wavefunc(qaoa_params)
        cost = self.__calculate_cost_of_wavefunc(wavefunc)        
        result.append(cost, qaoa_params)
        return cost


    def __calculate_cost_of_wavefunc(self, wavefunc):   

        cost = 0.0
        wavefunc_bra = np.copy(np.conj(wavefunc))

        for i in range(self.__num_loanees): 
            wavefunc_ket = np.copy(wavefunc)
            cost += self.__inner_product(
                wavefunc_bra,
                self.__apply_h_B_onsite(wavefunc_ket, i)
            )                      
            
            for j in range(i):
                if self.__J[i,j] != 0:
                    wavefunc_ket = np.copy(wavefunc)
                    cost += self.inner_product(
                        wavefunc_bra,
                        self.__apply_h_B_coupling(wavefunc_ket, i, j)
                    )
        
        cost = np.real(cost)
        return cost

    def __inner_product(self, wavefunc_1, wavefunc_2):
        result = np.tensordot(
            wavefunc_1,
            wavefunc_2,
            axes=(
                np.arange(self.__num_loanees),
                np.arange(self.__num_loanees)
            )
        )
        return result


    # Evolve a wavefunc using a QAOA circuit with the given QAOA variational parameters (betas, gammas)
    # Here, a reduced Hibert space is used to describe the wavefunc,
    # i.e. dimension of the wavefunc is num_actions**num_loanees
    # instead of 2**(num_actions*num_loanees).
    def __evolve_wavefunc(self, qaoa_params: np.ndarray):
        assert qaoa_params.shape == (2*self.__p,)

        wavefunc = self.__prepare_equal_superposition_of_valid_states()
        for i in range(self.__p):
            wavefunc = self.__apply_U_problem(wavefunc, qaoa_params[2*i])
            wavefunc = self.__apply_U_mixing (wavefunc, qaoa_params[2*i + 1])

        return wavefunc


    # A valid state is a state where only one action is taken for each loanee.
    # Returned wavefunc has a dimension of [[num_actions]*num_loanees
    # e.g. for 3 loanees and 2 actions
    # wavefunc.shape == (2, 2, 2)
    # wavefunc = ( |10,10,10> + |10,10,01> + |10,01,10> + |10,01,01> + |01,10,10> + |01,10,01> + |01,01,10> + |01,01,01> ) / norm
    def __prepare_equal_superposition_of_valid_states(self):
        num_valid_states = self.__num_actions**self.__num_loanees
        wavefunc = np.ones(num_valid_states, dtype='complex') / np.sqrt(num_valid_states)
        wavefunc = np.reshape(wavefunc,[self.__num_actions]*self.__num_loanees)
        return wavefunc


    # U_problem  = exp(- i H_problem gamma)
    def __apply_U_problem(self, wavefunc, param_beta):
        assert isinstance(param_beta, float)
        assert np.shape(wavefunc) == tuple([self.__num_actions]*self.__num_loanees)

        arg = np.power(self.__es, param_beta)
        u_problem = np.conj(self.__vs.T).dot( arg[:,None] * self.__vs )
        
        for i in range(self.__num_loanees):
            wavefunc = np.tensordot(u_problem, wavefunc, axes=(1,i))
        
        return wavefunc


    # U_mixing  = exp(- i H_mixing beta)
    def __apply_U_mixing(self, wavefunc, param_gamma):
        assert isinstance(param_gamma, float)
        assert np.shape(wavefunc) == tuple([self.__num_actions]*self.__num_loanees)

        for i in range(self.__num_loanees):
            self.__apply_U_B_onsite(wavefunc, param_gamma, i)
            for j in range(i):
                if self.__J[i,j] != 0:
                    self.__apply_U_B_coupling(wavefunc, param_gamma, i, j)  

        return wavefunc


    # Apply an onsite term in U_B 
    def __apply_U_B_onsite(self, wavefunc, param_gamma, i):
        assert i < self.__num_loanees

        u = np.exp(
            -1j * (-self.__h[i,:]) * param_gamma
        )

        # Reshaping u
        # u = u[None, None, ..., :, None, ..., None]
        idx = '[' + 'None,'*i + ':' + ',None'*(self.__num_loanees-i-1) + ']'
        exec('u = u' + idx)
        
        wavefunc *= u
        return wavefunc
    

    # Apply a coupling term in U_B 
    def __apply_U_B_coupling(self, wavefunc, param_gamma, i, j):
        assert i>j
        
        # wavefunc = wavefunc[:, :, ..., 0, :, ..., 0, :, ..., :]
        idx = '['+':,'*j + '0,' + ':,'*(i-j-1) + '0' +',:'*(self.__num_loanees-i-1) + ']'
        exec('wavefunc = wavefunc' + idx)
        
        wavefunc *= np.exp(
            -1j * (-self.__J[i,j]) * param_gamma
        )
        return wavefunc

        
    # Apply an onsite term in H_B 
    def __apply_h_B_onsite(self, wavefunc, i):
        assert i < self.__num_loanees

        u = -self.h[i,:]

        # Reshaping u
        # u = u[None, None, ..., :, None, ..., None]
        idx = '[' +'None,'*i + ':' + ',None'*(self.__num_loanees-i-1) + ']'
        exec('u = u' + idx)
        
        wavefunc *= u
        return wavefunc

    
    # Apply the coupling term in H_B
    def __apply_h_B_coupling(self, wavefunc, i, j):
        assert i>j

        # -J * n_i * n_j
        h_B_coupling = np.zeros((self.__num_actions**2, self.__num_actions**2))
        h_B_coupling[0,0] = -self.__J[i,j]
        h_B_coupling = np.reshape(h_B_coupling, [self.__num_actions]*4)
    
        return np.tensordot(wavefunc, h_B_coupling, axes=([j,i],[0,1]))