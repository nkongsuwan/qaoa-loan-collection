from enum import Enum
from qiskit import Aer

class QiskitSimulator(Enum):
    QASM = Aer.get_backend("qasm_simulator")
    STATEVECTOR = Aer.get_backend("statevector_simulator")
    