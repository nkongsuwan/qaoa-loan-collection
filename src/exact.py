from src.loanee_graph import LoaneeGraph
from src.qaoa_interface import QaoaInterface
from src.result import ResultQaoa

class ExactSolver(QaoaInterface):

    def __init__(self, loanees: LoaneeGraph, qaoa_config: dict):

        super().__init__(loanees, qaoa_config)