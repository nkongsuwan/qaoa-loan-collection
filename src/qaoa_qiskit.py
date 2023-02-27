''' TODO Use or delete
self.__rng     = np.random.default_rng(12345)
if "numpy_seed" in qaoa_config:
    assert isinstance(qaoa_config["numpy_seed"], int)
    assert qaoa_config["numpy_seed"] >= 0
    self.__rng = np.random.default_rng(qaoa_config["numpy_seed"])
'''

#import numpy as np
#import networkx as nx
#from itertools import product
#from functools import partial

#from qiskit import QuantumCircuit, Aer
#from qiskit.algorithms.optimizers.cobyla import COBYLA
#from qiskit.opflow import I, X, Y, Z
#from qiskit.algorithms import QAOA
#from qiskit.opflow.evolutions import PauliTrotterEvolution, Suzuki
#from qiskit.circuit import Parameter

#from src.loanee_graph import LoaneeGraph
#from src.qaoa_interface import QaoaInterface
#from src.result import ResultQaoa


#class QaoaQiskit(QaoaInterface):
#   def __init__(self) -> None:
#      super().__init__()

class QaoaQiskit():
   
    def __init__(self):
       pass

'''    
    def __init__(self,
            Qs: np.ndarray,
            As: np.ndarray,
            e: float=0.2,
            simulator=Aer.get_backend('aer_simulator')
        ):
        self.N = np.arange(1, Qs.shape[0]+1)
        self.M = np.arange(1, Qs.shape[1]+1)
        self.len_N = Qs.shape[0]
        self.len_M = Qs.shape[1]
        self.num_term = int(self.len_N*self.len_M)
        self.e = e
        self.h_benz = Qs
        self.h = (1-e)*np.copy(Qs)
        self.h[:,0] += -e*np.sum(As,axis=1)
        self.J = e*np.copy(As)
        self.G = nx.from_numpy_matrix(As)
        self.H_A = 0 # Problem Hamiltonian
        self.H_B = 0 # Mixer Hamiltonian
        self.draw = None
        self.result = None
        # Eigenstate
        self.candidate = None
        # QTFT notation
        self.xs = []
        self.ps = []
        self.cs = None

        # Qiskit simulator
        self.simulator = simulator
'''

'''
    def optimized(self):
        # Construct Hamiltonian
        self.construct_H()
        # result including invalid state
        self.run_qaoa()
        # Eliminate invalid state
        self.valid_state()

        
    def construct_H(self):

        ops = partial(operator, len_M=self.len_M, len_N=self.len_N)

        # construct Identity opertor as a tensor of num_term qubits
        I_op = I
        for i in range(1, self.num_term):
            I_op ^= I
        # First term

        for i in self.N:
            for j in self.M:
                # Qiskit does not support numpy, convert to python's float is necessary.
                self.H_A += (-(1-self.e)/2)*float(self.h_benz[i-1, j -1])*( I_op  - ops(Z, i, j ) )

        # Second term
        for _ , (node_1, node_2, data) in enumerate(self.G.edges(data=True)):
            if node_1 != node_2:
                A = data['weight']
                self.H_A += (-self.e/4)*A*((I_op + ops(Z, node_1+1, 1 ) ) @ (I_op + ops(Z, node_2+1, 1 ) ))

        for i in self.N:
            if self.len_M == 2:
                self.H_B += (ops(X, i, 1 ) @ ops(X, i, 2 ) ) + ( ops(Y, i, 2 ) @ ops(Y, i, 1 ) )    
            else:
                for j in self.M:
                    jp1 = j + 1
                    if jp1 == self.len_M + 1:
                        jp1 = 1
                    self.H_B += (ops(X, i, j ) @ ops(X, i, jp1 ) ) + ( ops(Y, i, jp1 ) @ ops(Y, i, j ) )
        self.H_B = (-1/2)*self.H_B

        
    def run_qaoa(self):

        # Define initial state 
        initial_state = QuantumCircuit(self.num_term)
        initial_state.x(0)

        prev_q = 0
        for i in range(self.len_N-1):
            prev_q += self.len_M
            initial_state.x(prev_q)

        # self.draw = initial_state.draw()
        # QAOA simulator
        qaoa_obj = QAOA(optimizer=COBYLA(maxiter=1000, disp=False), mixer=self.H_B, quantum_instance=self.simulator, initial_state=initial_state, reps=2)
        result = qaoa_obj.compute_minimum_eigenvalue(self.H_A)
        self.result = result

        # Create template result
        list_of_valid_action = []
        for i in range(self.len_M):
            string = ['0']*self.len_M
            string[i] = '1'
            string = ''.join(string)
            list_of_valid_action.append(string)

        template_result = {}
        for i in product(list_of_valid_action, repeat=self.len_N):
            # join string in tuple
            template_result[''.join(i)] = 0

        merged_result = {**template_result, **result.eigenstate}

        self.candidate = merged_result

        
    def valid_state(self):

        # Get valid index
        # Normalized Probability 
        for candidate_key, candidate_value in self.candidate.items():
            valid = if_valid_state(candidate_key, length_action=self.len_M)
            if valid:
                self.xs.append(candidate_key)
                #self.ps.append(candidate_value/norm)
                self.ps.append(np.conjugate(candidate_value)*candidate_value)

                
    def get_instruction(self, operator, parameter_name, order=1, reps=1):
        param = Parameter(parameter_name)
        ops_ = (param*operator).exp_i()
        return PauliTrotterEvolution(trotter_mode=Suzuki(order=order, reps=reps)).convert(ops_).to_instruction()

        
    def get_custom_circuit(self, p:int, order=1, reps=1):
        """Get custom circuit for QAOA.

        Args:
            p (int): Number of QAOA steps
            order (int, optional): Order of Trotterization. Defaults to 1.
            reps (int, optional): Number of Trotterization. Defaults to 1.

        Returns:
            QuantumCircuit: Custom circuit
        """

        qc = QuantumCircuit(self.num_term)
        qc.x(0)

        prev_q = 0
        for _ in range(self.len_N-1):
            prev_q += self.len_M
            qc.x(prev_q)

        for j in range(p):
            qc.append(self.get_instruction(self.H_A, f'alpha_{j}', order=order, reps=reps), range(self.num_term))
            qc.append(self.get_instruction(self.H_B, f'beta_{j}', order=order, reps=reps), range(self.num_term))

        return qc
   
    
    # Convert arg into element in Psi.    
    def _to_str(self, n):
        convertString = "0123456789ABCDEF"
        if n < self.len_M:
            return convertString[n]
        else:
            return self._to_str(n//self.len_M) + convertString[n%self.len_M]

            
    def get_str_from_index(self, n):
        Str = self._to_str(n)
        return "0"*(self.len_N - len(Str)) + Str

        
    # Get cost from bitstring.
    def get_cost_from_str(self, state):       
        c = 0       
        for i in range(len(state)):
            c -= self.h[i, int(state[i])]
            for ii in range(i):
                if (int(state[i]) == 0 and int(state[ii]) == 0):
                    c -= self.J[i,ii]
        return c


def operator(operator, i, j, len_N, len_M):
    num_term = int(len_N*len_M)

    index = (i - 1)*len_M + j

    op = operator if index == 1 else I
    for ii in range(1, num_term):
        if (index - 1) == ii:
            op ^= operator
        else:
            op ^= I
    return op

    
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

    
class QiskitVQE(QaoaInterface):
    def __init__(self,
            Qs: np.ndarray,
            As: np.ndarray,
            e: float=0.2,
            simulator=Aer.get_backend('aer_simulator')
        ):
        self.N = np.arange(1, Qs.shape[0]+1)
        self.M = np.arange(1, Qs.shape[1]+1)
        self.len_N = Qs.shape[0]
        self.len_M = Qs.shape[1]
        self.num_term = int(self.len_N*self.len_M)
        self.e = e
        self.h_benz = Qs
        self.h = (1-e)*np.copy(Qs)
        self.h[:,0] += -e*np.sum(As,axis=1)
        self.J = e*np.copy(As)
        self.G = nx.from_numpy_matrix(As)
        self.H_A = 0 # Problem Hamiltonian
        self.H_B = 0 # Mixer Hamiltonian
        self.draw = None
        self.result = None
        # Eigenstate
        self.candidate = None
        # QTFT notation
        self.xs = []
        self.ps = []
        self.cs = None

        # Qiskit simulator
        self.simulator = simulator

        
    def _construct_H(self):

        ops = partial(operator, len_M=self.len_M, len_N=self.len_N)

        # construct Identity opertor as a tensor of num_term qubits
        I_op = I
        for i in range(1, self.num_term):
            I_op ^= I
        # First term

        for i in self.N:
            for j in self.M:
                # Qiskit does not support numpy, convert to python's float is necessary.
                self.H_A += (-(1-self.e)/2)*float(self.h_benz[i-1, j -1])*( I_op  - ops(Z, i, j ) )

        # Second term
        for _ , (node_1, node_2, data) in enumerate(self.G.edges(data=True)):
            if node_1 != node_2:
                A = data['weight']
                self.H_A += (-self.e/4)*A*((I_op + ops(Z, node_1+1, 1 ) ) @ (I_op + ops(Z, node_2+1, 1 ) ))

        for i in self.N:
            if self.len_M == 2:
                self.H_B += (ops(X, i, 1 ) @ ops(X, i, 2 ) ) + ( ops(Y, i, 2 ) @ ops(Y, i, 1 ) )    
            else:
                for j in self.M:
                    jp1 = j + 1
                    if jp1 == self.len_M + 1:
                        jp1 = 1
                    self.H_B += (ops(X, i, j ) @ ops(X, i, jp1 ) ) + ( ops(Y, i, jp1 ) @ ops(Y, i, j ) )
        self.H_B = (-1/2)*self.H_B

        # Define initial state 
        initial_state = QuantumCircuit(self.num_term)
        initial_state.x(0)

        prev_q = 0
        for i in range(self.len_N-1):
            prev_q += self.len_M
            initial_state.x(prev_q)

        # self.draw = initial_state.draw()
        # QAOA simulator
        qaoa_obj = QAOA(optimizer=COBYLA(maxiter=1000, disp=False), mixer=self.H_B, quantum_instance=self.simulator, initial_state=initial_state, reps=2)
        result = qaoa_obj.compute_minimum_eigenvalue(self.H_A)
        self.result = result

        # Create template result
        list_of_valid_action = []
        for i in range(self.len_M):
            string = ['0']*self.len_M
            string[i] = '1'
            string = ''.join(string)
            list_of_valid_action.append(string)

        template_result = {}
        for i in product(list_of_valid_action, repeat=self.len_N):
            # join string in tuple
            template_result[''.join(i)] = 0

        merged_result = {**template_result, **result.eigenstate}

        self.candidate = merged_result

        
    def _valid_state(self):

        # Get valid index
        # Normalized Probability 
        for candidate_key, candidate_value in self.candidate.items():
            valid = if_valid_state(candidate_key, length_action=self.len_M)
            if valid:
                self.xs.append(candidate_key)
                #self.ps.append(candidate_value/norm)
                self.ps.append(np.conjugate(candidate_value)*candidate_value)

                
    def _get_instruction(self, operator, parameter_name, order=1, reps=1):
        param = Parameter(parameter_name)
        ops_ = (param*operator).exp_i()
        return PauliTrotterEvolution(trotter_mode=Suzuki(order=order, reps=reps)).convert(ops_).to_instruction()

        
    def runtime_qaoa(self, provider, backend_name:str, p:int=2, shots:int=4096):
        """Execute QAOA using VQE runtime.

        Args:
            provider (_type_): IBMQ Provider
                IBMQ.load_account()
                provider = IBMQ.get_provider(hub='ibm-q-startup', group='qtft', project='main')
            backend_name (str): String of backend name
            p (int, optional): Number of QAOA steps. Defaults to 2.
            shots (int, optional): Number of shots. Defaults to 4096.
        """

        self._construct_H()

        intermediate_info = {"nfev": [], "parameters": [], "energy": [], "stddev": []}

        def raw_callback(*args):
            # check if interim results, since both interim results (list) and final results (dict) are returned
            if type(args[1]) is list:
                job_id, (nfev, parameters, energy, stddev) = args
                intermediate_info["nfev"].append(nfev)
                intermediate_info["parameters"].append(parameters)
                intermediate_info["energy"].append(energy)
                intermediate_info["stddev"].append(stddev)

        qc = self._get_custom_circuit(p)
        vqe_inputs = {
            "ansatz": qc,
            "operator": self.H_A,
            "optimizer": COBYLA(),
            "initial_parameters": 'random',
            "measurement_error_mitigation": True,
            "shots": shots,
        }

        options = {
            'backend_name': backend_name
        }

        job = provider.runtime.run(
            program_id='vqe',
            options=options,
            inputs=vqe_inputs,
            callback=raw_callback
        )

        self.job_id = job.job_id()


    def get_results(self, provider):

        # Create template result
        list_of_valid_action = []
        for i in range(self.len_M):
            string = ['0']*self.len_M
            string[i] = '1'
            string = ''.join(string)
            list_of_valid_action.append(string)

        template_result = {}
        for i in product(list_of_valid_action, repeat=self.len_N):
            # join string in tuple
            template_result[''.join(i)] = 0
        
        job = provider.runtime.job(self.job_id)

        result = job.result()
        self.result = result

        merged_result = {**template_result, **result['eigenstate']}

        self.candidate = merged_result

        self._valid_state()


    def _get_custom_circuit(self, p:int, order=1, reps=1):
        """Get custom circuit for QAOA.

        Args:
            p (int): Number of QAOA steps
            order (int, optional): Order of Trotterization. Defaults to 1.
            reps (int, optional): Number of Trotterization. Defaults to 1.

        Returns:
            QuantumCircuit: Custom circuit
        """

        qc = QuantumCircuit(self.num_term)
        qc.x(0)

        prev_q = 0
        for _ in range(self.len_N-1):
            prev_q += self.len_M
            qc.x(prev_q)

        for j in range(p):
            qc.append(self._get_instruction(self.H_A, f'alpha_{j}', order=order, reps=reps), range(self.num_term))
            qc.append(self._get_instruction(self.H_B, f'beta_{j}', order=order, reps=reps), range(self.num_term))

        return qc

        
    # Convert arg into element in Psi.    
    def _to_str(self, n):
        convertString = "0123456789ABCDEF"
        if n < self.len_M:
            return convertString[n]
        else:
            return self._to_str(n//self.len_M) + convertString[n%self.len_M]

            
    def get_str_from_index(self, n):
        Str = self._to_str(n)
        return "0"*(self.len_N - len(Str)) + Str

        
    # Get cost from bitstring.
    def get_cost_from_str(self, state):       
        c = 0       
        for i in range(len(state)):
            c -= self.h[i, int(state[i])]
            for ii in range(i):
                if (int(state[i]) == 0 and int(state[ii]) == 0):
                    c -= self.J[i,ii]
        return c
'''
