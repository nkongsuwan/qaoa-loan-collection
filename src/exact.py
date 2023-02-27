from src.loanee_graph import LoaneeGraph
from src.problem_interface import ProblemInterface
from src.result import ResultQaoa

class ExactSolver(ProblemInterface):

    def __init__(self, loanees: LoaneeGraph, qaoa_config: dict):

        super().__init__(loanees, qaoa_config)