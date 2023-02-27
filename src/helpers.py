import numpy as np


# method for generating expected_net_profit_matrix follows Appendix A in arXiv:2110.15870
def generate_random_dataset(num_loanees, num_actions, association_cutoff=0.4, seed=None):

    if seed is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(seed)

    # the association matrix element (weight) Ai,i′sampled independently 
    # from the uniform distribution between [0, 1]. 
    association_matrix = np.zeros((num_loanees, num_loanees))
    for i in range(1, num_loanees):
        for j in range(i):
            if rng.random() < association_cutoff:
                x = rng.random()
                association_matrix[i][j] = x
                association_matrix[j][i] = x
    
    # value of loanee i: V(i) = sum_j A(i,j)
    # fraction: j = 1.0 / num_actions
    # maximum net return of loanee i: h^max(i) = f * V(i)
    # expected net profit of loanee i: h^max(i) * [1.0, ..., 1.0/num_actions]
    #   which is randomly shuffle for each loanee
    #   so each action of each loanee has random weight
    expected_net_profit_matrix = np.zeros((num_loanees, num_actions))
    maximum_net_return = association_matrix.sum(axis=1) / num_actions
    action_weights = np.linspace(1.0, 1.0/num_actions, num_actions)
    for i in range(num_loanees):
        rng.shuffle(action_weights)
        expected_net_profit_matrix[i,:] =  maximum_net_return[i] * action_weights

    return expected_net_profit_matrix, association_matrix