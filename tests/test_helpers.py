import numpy as np
from src.helpers import generate_random_dataset

def test_generate_random_dataset():

    exp, ass = generate_random_dataset(5, 4)
    assert ass.shape == (5, 5)
    assert exp.shape == (5, 4)
    assert np.trace(ass) == 0

    # test cutoff
    exp, ass = generate_random_dataset(5, 4, association_cutoff=0)
    assert np.sum(ass) == 0
    assert np.sum(exp) == 0

    # test seed
    _, ass1 = generate_random_dataset(5, 4, seed=999)
    _, ass2 = generate_random_dataset(5, 4, seed=999)
    _, ass3 = generate_random_dataset(5, 4, seed=111)
    assert np.array_equal(ass1, ass2)
    assert not np.array_equal(ass1, ass3)