# Loan Collection Optimization with QAOA

Python implementation of Tangpanitanon et. al., "Hybrid Quantum-Classical Algorithms for Loan Collection Optimization with Loan Loss Provisions", [arXiv:2110.15870](https://arxiv.org/abs/2110.15870).

Here, two Quantum Approximate Optimization Algorithm (QAOA) methods are used:

- An analytical approach where a reduced Hibert space is used for computational efficiency, as described in [arXiv:2110.15870](https://arxiv.org/abs/2110.15870).
- A circuit-based approach using [Qiskit QAOA library](https://qiskit.org/documentation/stubs/qiskit.algorithms.minimum_eigensolvers.QAOA.html).

## Installation

    pip install -r requirements.txt

## Testing
Install pytest with `pip install pytest`, then run the test suites:

    pytest tests

## Demos
Demo codes are given in [notebooks/](notebooks/), which can be run using Jupyter Notebook.