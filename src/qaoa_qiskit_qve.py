class QiskitVQE(QaoaInterface):
    def __init__(self,
            Qs: np.ndarray,
            As: np.ndarray,
            e: float=0.2,
            simulator=Aer.get_backend('aer_simulator')
        ):
        self.N = np.arange(1, Qs.shape[0]+1)
        self.M = np.arange(1, Qs.shape[1]+1)
        self._num_loanees = Qs.shape[0]
        self._num_actions = Qs.shape[1]
        self._num_qubits = int(self._num_loanees*self._num_actions)
        self._epsilon = e
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

        
    def __initialize_hamiltonian(self):

        ops = partial(operator, len_M=self._num_actions, len_N=self._num_loanees)

        # construct Identity opertor as a tensor of num_term qubits
        I_op = I
        for i in range(1, self._num_qubits):
            I_op ^= I
        # First term

        for i in self.N:
            for j in self.M:
                # Qiskit does not support numpy, convert to python's float is necessary.
                self.H_A += (-(1-self._epsilon)/2)*float(self.h_benz[i-1, j -1])*( I_op  - ops(Z, i, j ) )

        # Second term
        for _ , (node_1, node_2, data) in enumerate(self.G.edges(data=True)):
            if node_1 != node_2:
                A = data['weight']
                self.H_A += (-self._epsilon/4)*A*((I_op + ops(Z, node_1+1, 1 ) ) @ (I_op + ops(Z, node_2+1, 1 ) ))

        for i in self.N:
            if self._num_actions == 2:
                self.H_B += (ops(X, i, 1 ) @ ops(X, i, 2 ) ) + ( ops(Y, i, 2 ) @ ops(Y, i, 1 ) )    
            else:
                for j in self.M:
                    jp1 = j + 1
                    if jp1 == self._num_actions + 1:
                        jp1 = 1
                    self.H_B += (ops(X, i, j ) @ ops(X, i, jp1 ) ) + ( ops(Y, i, jp1 ) @ ops(Y, i, j ) )
        self.H_B = (-1/2)*self.H_B

        # Define initial state 
        initial_state = QuantumCircuit(self._num_qubits)
        initial_state.x(0)

        prev_q = 0
        for i in range(self._num_loanees-1):
            prev_q += self._num_actions
            initial_state.x(prev_q)

        # self.draw = initial_state.draw()
        # QAOA simulator
        qaoa_obj = QAOA(optimizer=COBYLA(maxiter=1000, disp=False), mixer=self.H_B, quantum_instance=self.simulator, initial_state=initial_state, reps=2)
        result = qaoa_obj.compute_minimum_eigenvalue(self.H_A)
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

        
    def _valid_state(self):

        # Get valid index
        # Normalized Probability 
        for candidate_key, candidate_value in self.candidate.items():
            valid = if_valid_state(candidate_key, length_action=self._num_actions)
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

        self.__initialize_hamiltonian()

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
        for i in range(self._num_actions):
            string = ['0']*self._num_actions
            string[i] = '1'
            string = ''.join(string)
            list_of_valid_action.append(string)

        template_result = {}
        for i in product(list_of_valid_action, repeat=self._num_loanees):
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

        qc = QuantumCircuit(self._num_qubits)
        qc.x(0)

        prev_q = 0
        for _ in range(self._num_loanees-1):
            prev_q += self._num_actions
            qc.x(prev_q)

        for j in range(p):
            qc.append(self._get_instruction(self.H_A, f'alpha_{j}', order=order, reps=reps), range(self._num_qubits))
            qc.append(self._get_instruction(self.H_B, f'beta_{j}', order=order, reps=reps), range(self._num_qubits))

        return qc

        
    # Convert arg into element in Psi.    
    def _to_str(self, n):
        convertString = "0123456789ABCDEF"
        if n < self._num_actions:
            return convertString[n]
        else:
            return self._to_str(n//self._num_actions) + convertString[n%self._num_actions]

            
    def get_str_from_index(self, n):
        Str = self._to_str(n)
        return "0"*(self._num_loanees - len(Str)) + Str

        
    # Get cost from bitstring.
    def get_cost_from_str(self, state):       
        c = 0       
        for i in range(len(state)):
            c -= self.h[i, int(state[i])]
            for ii in range(i):
                if (int(state[i]) == 0 and int(state[ii]) == 0):
                    c -= self.J[i,ii]
        return c