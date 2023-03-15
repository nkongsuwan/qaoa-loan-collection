from enum import Enum
from qiskit import Aer

class QiskitSimulator(Enum):
    QASM = Aer.get_backend("qasm_simulator")
    STATEVECTOR = Aer.get_backend("statevector_simulator")
    
# Valid states are states with 1 action per loanee
class InitialState(Enum):
    GROUND = "ground"
    DEFAULT = "default"
    EQUAL_SUPERPOSITION = "equal_superposition"
    EQUAL_SUPERPOSITION_OF_VALID_STATES = "equal_superposition_of_valid_states"

