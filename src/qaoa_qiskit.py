import numpy as np
import networkx as nx
#from itertools import product
from functools import partial

from qiskit import QuantumCircuit
from qiskit.opflow import PauliOp, I, X, Y, Z
from qiskit.algorithms import QAOA
from qiskit.algorithms.optimizers.cobyla import COBYLA
#from qiskit.opflow.evolutions import PauliTrotterEvolution, Suzuki
#from qiskit.circuit import Parameter

from src.loanee_graph import LoaneeGraph
from src.qaoa_interface import QaoaInterface
from src.enums import QiskitSimulator
from src.result import ResultQaoa


class QaoaQiskit(QaoaInterface):
   
    def __init__(self, loanees: LoaneeGraph, qaoa_config: dict, qiskit_config: dict):

        super().__init__(loanees, qaoa_config)

        self.__expected_net_profit = loanees.get_expected_net_profit_matrix()
        self.__association_graph = nx.from_numpy_array(
            loanees.get_association_matrix()
        )

        self.__H_problem = None
        self.__H_mixing = None
        self.__initialize_hamiltonian()

        # Result
        self.candidate = None
        self.result = None
        self.xs = []
        self.ps = []
        
        # Qiskit simulator
        match qiskit_config["simulator"]:
            case "qasm_simulator":
                self.__simulator = QiskitSimulator.QASM
            case "statevector_simulator":
                self.__simulator = QiskitSimulator.STATEVECTOR
            case _:
                assert False
        

    def __initialize_hamiltonian(self):
        self.__H_problem = 0
        self.__H_mixing = 0
        
        X_op = partial(self.__generate_multi_pauliop, pauliOp=X)
        Y_op = partial(self.__generate_multi_pauliop, pauliOp=Y)
        Z_op = partial(self.__generate_multi_pauliop, pauliOp=Z)

        # construct Identity opertor as a tensor of num_term qubits
        I_op = I
        for _ in range(self._num_qubits - 1):
            I_op ^= I
        
        # Problem Hamiltonian (1st term)
        for i in np.arange(self._num_loanees):
            for j in np.arange(self._num_actions):
                self.__H_problem += (
                    - 0.5*(1 - self._epsilon) *
                    float( self.__expected_net_profit[i, j] ) *
                    ( I_op  - Z_op(i, j) )
                )

        # Problem Hamiltonian (2nd term)
        for _ , (loanee_1, loanee_2, data) in enumerate(self.__association_graph.edges(data=True)):
            if loanee_1 != loanee_2:
                w = data['weight']
                self.__H_problem += - 0.25 * self._epsilon * w * ( 
                    ( I_op + Z_op(loanee_1, 0) ) @ 
                    ( I_op + Z_op(loanee_2, 0) ) 
                )

        # Mixing Hamiltonian
        for i in np.arange(self._num_loanees):
            if self._num_actions == 2:
                self.__H_mixing += X_op(i, 0) @ X_op(i, 1)
                self.__H_mixing += Y_op(i, 1) @ Y_op(i, 0)
            else:
                for j in np.arange(self._num_actions):
                    jp1 = j + 1
                    if jp1 == self._num_actions:
                        jp1 = 0
                    self.__H_mixing += X_op(i, j  ) @ X_op(i, jp1) 
                    self.__H_mixing += Y_op(i, jp1) @ Y_op(i, j  )
        self.__H_mixing *= -1/2


    def __generate_multi_pauliop(
        self, 
        i: int, # loanee index
        j: int, # action index
        pauliOp: PauliOp
    ) -> PauliOp:

            index = self._num_actions*i + j
            op = pauliOp if index == 0 else I
            
            for k in range(1, self._num_qubits):
                if index == k:
                    op ^= pauliOp
                else:
                    op ^= I

            return op

    
    def _run_qaoa(
        self, 
        initial_qaoa_params: np.ndarray
    ) -> ResultQaoa:

        result_qaoa = ResultQaoa()
        
        initial_state = self._prepare_initial_state()
        
        qaoa_qiskit = QAOA(
            optimizer = COBYLA(maxiter=self._optimizer_maxiter, disp=False),
            reps = self._p,
            initial_state = initial_state,
            mixer = self.__H_mixing, 
            initial_point = initial_qaoa_params,
            quantum_instance = self.__simulator.value
        )
        
        result = qaoa_qiskit.compute_minimum_eigenvalue(self.__H_problem)
        return result
        
        '''
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
        '''


    def _prepare_initial_state(self) -> QuantumCircuit:
        initial_state = QuantumCircuit(self._num_qubits)
        for i in range(self._num_qubits):
            initial_state.h(i)
        return initial_state
    

    '''   
    def eliminate_invalid_state(self):

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
