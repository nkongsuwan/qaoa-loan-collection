''' TODO Use or delete
self.__rng     = np.random.default_rng(12345)
if "numpy_seed" in qaoa_config:
    assert isinstance(qaoa_config["numpy_seed"], int)
    assert qaoa_config["numpy_seed"] >= 0
    self.__rng = np.random.default_rng(qaoa_config["numpy_seed"])
'''

import numpy as np
import networkx as nx
#from itertools import product
from functools import partial

from qiskit import Aer #QuantumCircuit
#from qiskit.algorithms.optimizers.cobyla import COBYLA
from qiskit.opflow import PauliOp, I, X, Y, Z
#from qiskit.algorithms import QAOA
#from qiskit.opflow.evolutions import PauliTrotterEvolution, Suzuki
#from qiskit.circuit import Parameter

from src.loanee_graph import LoaneeGraph
from src.qaoa_interface import QaoaInterface
#from src.result import ResultQaoa


class QaoaQiskit(QaoaInterface):
   
    def __init__(self, loanees: LoaneeGraph, qaoa_config: dict, qiskit_config: dict):

        super().__init__(loanees, qaoa_config)

        self.__H_mixingenz = loanees.get_expected_net_profit_matrix()
        self.__G = nx.from_numpy_matrix(
            loanees.get_association_matrix()
        )

        self.__H_problem = 0
        self.__H_mixing = 0
        self.__initialize_hamiltonian()

        # Result
        self.candidate = None
        self.result = None
        self.xs = []
        self.ps = []
        
        # Qiskit simulator
        self.__simulator = Aer.get_backend(qiskit_config["simulator"])


    def __initialize_hamiltonian(self):
        X_op = partial(self.__generate_multi_pauliop, pauliOp=X)
        Y_op = partial(self.__generate_multi_pauliop, pauliOp=Y)
        Z_op = partial(self.__generate_multi_pauliop, pauliOp=Z)

        # construct Identity opertor as a tensor of num_term qubits
        I_op = I
        for _ in range(self._num_qubits - 1):
            I_op ^= I
        
        # Problem Hamiltonian (1st term)
        for i in np.arange(1, self._num_loanees + 1):
            for j in np.arange(1, self._num_actions + 1):
                self.__H_problem += (
                    - 0.5*(1 - self._epsilon) *
                    float( self.__H_mixingenz[i-1, j-1] ) *
                    ( I_op  - Z_op(i, j) )
                )

        # Problem Hamiltonian (2nd term)
        for _ , (node_1, node_2, data) in enumerate(self.__G.edges(data=True)):
            if node_1 != node_2:
                A = data['weight']
                self.__H_problem += - 0.25*self._epsilon*A*( 
                    ( I_op + Z_op(node_1+1, 1) ) @ 
                    ( I_op + Z_op(node_2+1, 1) ) 
                )

        # Mixing Hamiltonian
        for i in np.arange(1, self._num_loanees + 1):
            if self._num_actions == 2:
                self.__H_mixing += X_op(i, 1) @ X_op(i, 2)
                self.__H_mixing += Y_op(i, 2) @ Y_op(i, 1)
            else:
                for j in np.arange(1, self._num_actions + 1):
                    jp1 = j + 1
                    if jp1 == self._num_actions + 1:
                        jp1 = 1
                    self.__H_mixing += X_op(i, j  ) @ X_op(i, jp1) 
                    self.__H_mixing += Y_op(i, jp1) @ Y_op(i, j  )
        self.__H_mixing *= -1/2


    def __generate_multi_pauliop(
        self, 
        pauliOp: PauliOp,
        i: int,
        j: int
    ) -> PauliOp:

            index = (i - 1)*self._num_actions + j
            op = pauliOp if index == 1 else I
            
            for k in range(1, self._num_qubits):
                if (index - 1) == k:
                    op ^= pauliOp
                else:
                    op ^= I

            return op


    def optimize_qaoa_params(self, initial_qaoa_params: np.ndarray = None):
        pass

    '''
    def optimized(self):
        # Construct Hamiltonian
        self.__initialize_hamiltonian()
        # result including invalid state
        self.run_qaoa()
        # Eliminate invalid state
        self.valid_state()

        
    def run_qaoa(self):

        # Define initial state 
        initial_state = QuantumCircuit(self._num_qubits)
        initial_state.x(0)

        prev_q = 0
        for i in range(self._num_loanees-1):
            prev_q += self._num_actions
            initial_state.x(prev_q)

        # QAOA simulator
        qaoa_obj = QAOA(optimizer=COBYLA(maxiter=1000, disp=False), mixer=self.__H_mixing, quantum_instance=self.__simulator, initial_state=initial_state, reps=2)
        result = qaoa_obj.compute_minimum_eigenvalue(self.__H_problem)
        self.result = result

        # Create template result
        list_of_valid_action = []
        for i in range(self._num_actions):
            string = ['0']*self._num_actions
            string[i] = '1'
            string = ''.join(string)
            list_of_valid_action.append(string)

        template_result = {}
        for i in product(list_of_valid_action, repeat=self._num_loanees):
            # join string in tuple
            template_result[''.join(i)] = 0

        merged_result = {**template_result, **result.eigenstate}

        self.candidate = merged_result

        
    def valid_state(self):

        # Get valid index
        # Normalized Probability 
        for candidate_key, candidate_value in self.candidate.items():
            valid = if_valid_state(candidate_key, length_action=self._num_actions)
            if valid:
                self.xs.append(candidate_key)
                self.ps.append(np.conjugate(candidate_value)*candidate_value)        
    '''

'''
def if_valid_state(bitstring, length_action):
    assert(len(bitstring)%length_action == 0)
    for i in range(0, len(bitstring), length_action):
        valid = True
        summation = 0
        for s in bitstring[i:i+length_action]:
            summation += int(s)
        if summation != 1:
            valid = False
            break
    return valid
'''