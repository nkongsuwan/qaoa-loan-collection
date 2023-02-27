#  LEGACY CODE — TO BE DELETED LATER
#  ============================================================
#
#  Quantum algorithms for loan loss collection optimization
#  with a provision constraint.
#
#  Written by QTFT team. 7th Oct 2021.
#

import numpy as np
from numpy import tensordot, kron, ones
from tqdm import tqdm
from scipy.optimize import minimize, LinearConstraint, Bounds
import networkx as nx
import community
    

class QAOA():
    """

    Parameters

    ----------
        
    Qs : np.ndarray,
        A 2D numpy array containing Q-values.
    
    As : np.ndarray,
        A 2D numpy array containing association matrix.
        
    e : float, default=1/20
        The weight between the bank's revenue and the cash flow 
        among bank's customers in the cost function.
        
    p : int, default=2
        The number of driving cycles in QAOA.

    Attributes

    ----------
    
    N : int,
        The number of assets.
        
    M : int,
        The number of actions.
    
    h : np.ndarray,
        A 2D numpy array containing effective onsite "magnetic" fields.
        
    J : np.ndarray,
        A 2D numpy array containing "spin-spin "interactions between assets.
        
    psi : np.ndarray,
        A rank-N tensor containing a quantum state. Each leg has dimension M.
        
    costs : np.ndarray,
        Cost function during classical optimisation. 
        
    res : scipy.optimize.OptimizeResult,
        A result from scipy.optimize.minimize.
          
    """
    
    def __init__(self, Qs, As, e, p):
        
        assert Qs.ndim == 2
        assert As.ndim == 2
        assert As.shape[0] == As.shape[1] 
        assert As.shape[0] == Qs.shape[0]
        assert isinstance(e, float) and e >= 0
        assert isinstance(p, int) and p >= 1                

        self.N = Qs.shape[0]
        self.M = Qs.shape[1]
        self.Qs = np.copy(Qs)
        self.As = np.copy(As)
        self.h = (1-e)*np.copy(Qs)
        self.h[:,0] += -e*np.sum(As,axis=1)
        self.J = e*np.copy(As)
        self.e = e
        self.p = p
        self.c0 = -self.e * np.sum(As)/2
        self.psi = None
        self.costs = None
        self.res = None
        
        
    def cost(self, params):
        
        assert len(params) == 2 * self.p
        
        self.evolve(params)
        
        psi_bra = np.copy(np.conj(self.psi))
        
        c = self.c0
        
        for i in range(self.N): 
            
            psi_ket = np.copy(self.psi)
        
            c += self.inner_product(psi_bra, self._apply_h_B_onsite(psi_ket,i))                      
                
            for ii in range(i):
                
                if self.J[i,ii] != 0:
                    
                    psi_ket = np.copy(self.psi)
            
                    c += self.inner_product(psi_bra, self._apply_h_B_coupling(psi_ket,i,ii))
        
        self.costs += [np.real(c)]
        #Edited : print QAOA cost function
        #print(c)
        return np.real(c)
    
    def optimized(self, params=None, method='COBYLA', disp=False, maxiter=50):
    
        self.costs = []
        
        if params is None:
            params = np.random.rand(2*self.p)
        
        #Edited : parameter
        #params = np.array([0.8274463, 0.79026806])
        #print(params)
        self.res = minimize(self.cost,params, method=method, options={'disp':False,'maxiter':maxiter})
                
      
    def evolve(self, params):
        
        assert len(params) == 2 * self.p 
        #Edited : changing state from ones to zeros
        self.restart(state="ones")
        #print(self.psi)
        for t in range(self.p):
            self._apply_U_A(params[2*t])
            self._apply_U_B(params[2*t+1])
 

    def restart(self, state="zeros"):
        
        if state=="zeros":
            self.psi = np.zeros(self.M**self.N,dtype='complex')
            self.psi[0] = 1
            self.psi = np.reshape(self.psi,[self.M]*self.N) 
            
        if state=="ones":
            self.psi = np.ones(self.M**self.N,dtype='complex') / np.sqrt(self.M**self.N)
            self.psi = np.reshape(self.psi,[self.M]*self.N) 

        if state=="local":
            m = Exact(self.Qs, self.As, e=0) 
            m.optimized()
            self.psi = np.zeros(self.M**self.N,dtype='complex')
            self.psi[np.argmin(m.costs)] = 1
            self.psi = np.reshape(self.psi,[self.M]*self.N) 
        
    def inner_product(self,psi_1,psi_2):
        
        return np.tensordot(psi_1, psi_2, axes=(np.arange(self.N),np.arange(self.N)))
        
    
    # Apply the total U_A 
    def _apply_U_A(self, beta_t):
    
        u = self._get_u_A(beta_t)
        n = 0
        for i in range(self.N):
        
            assert np.shape(self.psi)[i] == self.M
        
            self.psi = np.tensordot(u,self.psi,axes=(1,i))  

    # Apply the total U_B
    def _apply_U_B(self, gamma_t):
    
        for i in range(self.N):
        
            assert np.shape(self.psi)[i] == self.M       
        
            self._apply_u_B_onsite(gamma_t,i)
                       
            for ii in range(i):
                
                if self.J[i,ii] != 0:
            
                    self._apply_u_B_coupling(gamma_t,i,ii)   
                
                
    # Get a tight-binding operator acting on one asset 
    def _get_u_A(self, beta_t):
    
        global vs, es
    
        # Eigenstates  |E_k> = a^\dagger_k |0>
        if 'vs' not in globals():
            vs = np.exp(-1j*np.array([[ 2*np.pi*k*j/self.M for k in range(self.M)] for j in range(self.M)]))
            vs = vs/ np.sqrt(self.M)

        # Eigenvalues e^{-iE_k} where E_k = 2*cos(k)
        if 'es' not in globals(): 
            es = np.exp(-1j*2*np.cos(np.arange(self.M)*2*np.pi/self.M))

        return np.conj(vs.T).dot(np.power(es, beta_t)[:,None]*vs)   
    
    
    # Apply an onsite term in U_B 
    def _apply_u_B_onsite(self, gamma_t,i):
    
        assert i < self.N
    
        u = np.exp(-1j*(-self.h[i,:]) * gamma_t)
    
        idx = '[' +'None,'*i + ':' + ',None'*(self.N-i-1) + ']'
    
        exec('self.psi *= u'+idx)


    # Apply a coupling term in U_B 
    def _apply_u_B_coupling(self, gamma_t,i,ii):
    
        assert i>ii
    
        idx = '['+':,'*ii + '0,' + ':,'*(i-ii-1) + '0' +',:'*(self.N-i-1) + ']'
    
        exec('self.psi' + idx +'*= np.exp(-1j*(-self.J[i,ii])*gamma_t)')
        
        
    # Apply an onsite term in H_B 
    def _apply_h_B_onsite(self, psi, i):
    
        assert i < self.N
    
        u = -self.h[i,:]
    
        idx = '[' +'None,'*i + ':' + ',None'*(self.N-i-1) + ']'
    
        exec('psi *= u'+idx)
        
        return psi

    # Apply the coupling term in H_B
    def _apply_h_B_coupling(self, psi,i,ii):
    
        assert i>ii
    
        # -J * n_i * n_j
        h_B_coupling = np.zeros((self.M**2,self.M**2))
        h_B_coupling[0,0] = -self.J[i,ii]
        h_B_coupling = np.reshape(h_B_coupling,[self.M]*4)
            
        return np.tensordot(psi, h_B_coupling, axes=([ii,i],[0,1]))
    

    # Convert arg into element in Psi.    
    def _to_str(self, n):
        convertString = "0123456789ABCDEF"
        if n < self.M:
            return convertString[n]
        else:
            return self._to_str(n//self.M) + convertString[n%self.M]

    def get_str_from_index(self, n):
        Str = self._to_str(n)
        return "0"*(self.N - len(Str)) + Str

    
    # Get cost from bitstring.
    def get_cost_from_str(self, state):       
        c = 0       
        for i in range(len(state)):
            c -= self.h[i, int(state[i])]
            for ii in range(i):
                if (int(state[i]) == 0 and int(state[ii]) == 0):
                    c -= self.J[i,ii]
        return c

    
    
class Exact():    
    """

    Parameters

    ----------
        
    Qs : np.ndarray,
        A 2D numpy array containing Q-values.
    
    As : np.ndarray,
        A 2D numpy array containing association matrix.
        
    e : float, default=1/20
        The weight between the bank's revenue and the cash flow 
        among bank's customers in the cost function.

    Attributes

    ----------
    
    N : int,
        The number of assets.
        
    M : int,
        The number of actions.
    
    h : np.ndarray,
        A 2D numpy array containing effective onsite "magnetic" fields.
        
    J : np.ndarray,
        A 2D numpy array containing "spin-spin "interactions between assets.
        
    costs : np.ndarray,
        Cost function during optimisation. 
        
    cost_min : float,
        The minimum cost function.
        
    x : string,
        A string containing the best bitstring.
          
    """
    
    def __init__(self, Qs, As, e):
        
        assert Qs.ndim == 2
        assert As.ndim == 2
        assert As.shape[0] == As.shape[1] 
        assert As.shape[0] == Qs.shape[0]
        assert isinstance(e, float) and e >= 0      
        
        self.N = Qs.shape[0]
        self.M = Qs.shape[1]
        self.h = (1-e)*np.copy(Qs)
        self.h[:,0] += -e*np.sum(As,axis=1)
        self.J = e*np.copy(As)
        self.e = e
        self.c0 = -e*np.sum(As)/2
        self.costs = None
        self.cost_min = None
        self.x = None
        
        
    def optimized(self):
        
        self._set_exact_energies()
        
        self.cost_min = min(self.costs)
        
        self.x = self._index2bitstring(np.argmin(self.costs))
        
        
    def _index2bitstring(self,idx):
        
        x = np.zeros(self.N)
        
        num = np.array([int(i) for i in np.base_repr(idx, base=self.M)])
        
        x[(self.N-len(num)):] = num
        
        x = [str(i) for i in x.astype('int')]
   
        new = ""
        for i in x:
            new += i
            
        return new
    
    
    def _set_exact_energies(self):
        
        
        if self.costs is None:
        
            self.costs = np.zeros(self.M**self.N) + self.c0
        
            coupling_node = np.zeros(self.M)
            coupling_node[0] = 1
            
            for i in range(self.N):
            
                self.costs += self._onsite_op(-self.h[i,:], i)
            
                for ii in range(i):
                    
                    if self.J[i,ii] != 0:
                    
                        self.costs += -self.J[i,ii] * self._twosite_op(coupling_node, coupling_node,ii,i)
            
            
    def _onsite_op(self,P,i):
        
        return kron(ones(self.M**i),kron(P,ones(self.M**(self.N-i-1))))

    
    def _twosite_op(self,P,Q,i,j):
        
        assert i < j
        
        return kron(ones(self.M**i),kron(P,kron(ones(self.M**(j-i-1)),kron(Q,ones(self.M**(self.N-j-1))))))
    

class Group:
    """

    Parameters

    ----------
        
    name : string,
        Group name.
    
    nodes : np.ndarray,
        A numpy array containing assets in the group.     
    
    
    Attributes
    
    -------
    
    size : int,
        The number of assets.
        
    xs : np.ndarray,
        A numpy array containing candidate bitstrings.
        
    ps : np.ndarray,
        A numpy array containing probabilities associated with xs.
        
    cs : np.ndarray,
        A numpy array containing cost functions associated with cs.
        
    x_exact : string,
        The exact best bitstring.
        
    c_exact : float,
        The exact minimum cost function.
        
    """
    
    def __init__(self, name, nodes):
        
        assert isinstance(name, str)
        assert isinstance(nodes, np.ndarray)
        
        self.id = name
        self.nodes = nodes
        
        self.size = len(nodes)
        self.xs = np.array([])
        self.ps = np.array([])
        self.cs = np.array([])
        self.x_exact = None
        self.c_exact = None
        
        
    def show(self):
        
        print("id:"+ self.id +", nodes:"+str(self.nodes)+", size:"+str(self.size))
        
        if self.xs:
            print("x_exact:"+self.x_exact+", c_exact:%.5f"%self.c_exact)
            for i in range(len(self.xs)):
                print("\tx:"+self.xs[i]+", p:%.5f"%self.ps[i] +", E:%.5f"%self.cs[i])
                

class DC_QAOA():
    """

    Parameters

    ----------
        
    Qs : np.ndarray,
        A 2D numpy array containing Q-values.
    
    As : np.ndarray,
        A 2D numpy array containing association matrix.
        
    e : float, default=1/20
        The weight between the bank's revenue and the cash flow 
        among bank's customers in the cost function.
        
    p : int, default=2
        The number of driving cycles in QAOA.
        
    n_candidates : int, default=10
        The maximum number of bitstrings that represent the solutions
        of a given problem instance.
        
    max_community_size : int, default=7
           
    
    Attributes
    
    -------
    
    N : int, 
        The number of assets.
        
    M : int, 
        The number of actions.
       
    groups : list,
        A list containing Group objects that represent asset communities.
        
    res : Group,
        The solution for all assets.
        
    x_best : string,
        The best found bitstring.
        
    """
    
    def __init__(self, Qs, As, e = 1/20, p = 2, n_candidates = 10, max_community_size = 7):
        
        assert Qs.ndim == 2
        assert As.ndim == 2
        assert As.shape[0] == As.shape[1] 
        assert As.shape[0] == Qs.shape[0]
        assert isinstance(e, float) and e >= 0
        assert isinstance(p, int) and p >= 1
        assert isinstance(n_candidates, int) and n_candidates > 0
        assert isinstance(max_community_size, int) and max_community_size > 0 
        
        self.Qs = Qs
        self.As = As
        self.e = e
        self.p = p
        self.n_candidates = n_candidates
        self.max_community_size = max_community_size
        
        self.N = Qs.shape[0]
        self.M = Qs.shape[1]
        self.groups = []
        self.res = None
        self.qtft_energy = []
       
    def optimized(self, maxiter=20, method='BFGS'):

        #if not self.groups:
        #    self.set_communities()
            
        for i in tqdm(range(len(self.groups))):
    
            # Perform QAOA
            Q = self.Qs[self.groups[i].nodes]
            A = self.As[self.groups[i].nodes][:, self.groups[i].nodes]    
            m = QAOA(Q, A, self.e, self.p)
            m.optimized(maxiter=maxiter,method=method)
            
            #*
            self.qtft_energy.append(m.res)

            # Output probabilities
            ps = np.abs(np.reshape(m.psi,-1))**2

            # Save Candidates
            inds = np.argsort(ps)[::-1][:self.n_candidates]
            self.groups[i].ps = ps[inds]
    
            # Save selected bitstrings
            for ind in inds:
                self.groups[i].xs = np.append(self.groups[i].xs, m.get_str_from_index(ind))
        
            # Save cost functions for selected bitstrings
            for bitstring in self.groups[i].xs:
                self.groups[i].cs = np.append(self.groups[i].cs, m.get_cost_from_str(bitstring))
        
            # Save exact ground state
            m_exact = Exact(Q, A, self.e)
            m_exact.optimized()
    
            self.groups[i].x_exact = m_exact.x
            self.groups[i].c_exact = m_exact.cost_min
            #self.groups[i].c_exact = m_exact.costs
        
            # State reconstruction
            if i == 0:
                self.res = self.groups[0]
            else:
                self.res = self._combine_groups(self.res, self.groups[i], ps, m)
 
        # Sorting
        ind = np.argsort(self.res.nodes)
        self.res.id = "final result"
        self.res.nodes = self.res.nodes[ind] 
        for i in range(len(self.res.xs)):
            self.res.xs[i] = "".join(list(np.array(list(self.res.xs[i]))[ind]))
          
        # Best solution
        self.x_best = self.res.xs[np.argmin(self.res.cs)]
    
    
    def set_communities(self):
    
        # Perform Greedy Modularity community finding.
        As_nx = nx.convert_matrix.from_numpy_array(self.As)
        G = nx.algorithms.community.modularity_max.greedy_modularity_communities(As_nx)

        for i in range(len(G)):
    
            # Perform iterative Louvain community finding with shared nodes.
            group = Group(str(i), np.array(list(G[i])))
            
            # Update groups
            self.groups += self._get_louvian_communities_with_shared_nodes(group)

        # Sort descendingly according to the size of the group.
        n_groups = np.array([group.size for group in self.groups])
        ind = np.argsort(n_groups)[::-1]
        self.groups = np.array(self.groups)[ind]
        
        
    def _combine_groups(self, L, R, ps, m):

        # Find the indices of the shared nodes
        _, ind_L, ind_R = np.intersect1d(L.nodes, R.nodes, return_indices = True)
        
        # Create a new combined group
        LR = Group(name = L.id+"|"+R.id, 
                   nodes = np.concatenate((L.nodes, np.delete(R.nodes, ind_R))))
               
        # Loop over all candidates
        for ii in range(len(L.xs)):
            for jj in range(len(R.xs)):
                LR = self._combine_bitstrings(L.xs[ii], R.xs[jj], ind_L, ind_R, LR)
   
        # Generate more candidates if no valid reconstruction
        count = 0     
        idx = np.argsort(ps)[::-1]
        while len(LR.xs)== 0:
            R_x = m.get_str_from_index(idx[self.n_candidates -1 + count])                            
            for ii in range(len(L.xs)):
                LR = self._combine_bitstrings(L.xs[ii], R_x, ind_L, ind_R, LR)                    
            count += 1
            
        # Evaluate bitstring. (QAOA is not performed.)
        m2 = QAOA(self.Qs[LR.nodes], self.As[LR.nodes][:,LR.nodes], self.e, self.p)
        for x in LR.xs:
            LR.cs = np.append(LR.cs, m2.get_cost_from_str(x))
                 
        # Truncate
        ind = np.argsort(LR.cs)[::-1][:self.n_candidates]    
        LR.xs = LR.xs[ind]
        LR.cs = LR.cs[ind]
        
        return LR
        
        
    def _combine_bitstrings(self, x_L, x_R, ind_L, ind_R, LR):
    
        # Check equality for the shared bits
        valid = [x_L[ind_L[k]] == x_R[ind_R[k]] for k in range(len(ind_L))]

        # Append x_R to x_L
        if np.all(valid):        
            x_R_new = ""
            for k in range(len(x_R)):
                if not k in ind_R:
                    x_R_new = x_R_new + x_R[k]
            x_new = x_L + x_R_new       
            LR.xs = np.append(LR.xs, x_new)
        
        return LR


    def _get_louvian_communities_with_shared_nodes(self, group):
    
        assert isinstance(group, Group)
    
        # NetworkX graph for the group
        G = nx.convert_matrix.from_numpy_array(self.As[group.nodes][:,group.nodes])

        # Find community groups
        mapping = community.best_partition(G)
    
        # Group labels
        labels = np.array(list(mapping.values()))

        # Unique labels
        unique_labels = np.unique(labels)
    
        # Subgroup including shared nodes 
        groups = []

        for i in range(len(unique_labels)):

            # New subgroup
            new_group = dict()
        
            # Indices of the assets in the subgroup i
            ind = np.where(labels==unique_labels[i])[0]
        
            # NetworkX subgraph
            g = G.subgraph(ind)
    
            # A set of shared nodes for the subgroup i
            shared_nodes_i = [] 
    
            # Check shared nodes
            for node in g.nodes:
            
                # Insert shared nodes
                if len(G.edges(node)) != len(g.edges(node)):
                
                    shared_nodes_i = np.array([int(v) for (u,v) in list(G.edges(node)) if (u,v) not in list(g.edges(node))])
                    ind = np.append(ind, shared_nodes_i)
            
            ind = np.unique(ind).astype('int')
            
            # Update group
            name = str(i) if group.id =="-1" else group.id + "_" + str(i)
            new_group = Group(name, group.nodes[ind])
        
            # Recursive reduction
            if new_group.size > self.max_community_size:
            
                smaller_groups = self._get_louvian_communities_with_shared_nodes(new_group)
            
                groups += smaller_groups 
                
            else:
                groups += [new_group]
            
        return groups
    
    
class GreedySearch():
    
    """

    Parameters

    ----------
        
    x0 : string,
        The initial bitstring.
        
    Qs : np.ndarray,
        A 2D numpy array containing Q-values.
        
    As : np.ndarray,
        A 2D numpy array containing association matrix.
        
    Ls : np.ndarray
        A 2D numpy array containing provisions.
        
    e : float, default=1/20
        The weight between the bank's revenue and the cash flow 
        among bank's customers in the cost function.

        
    Attributes
    
    ----------
    
    xs : np.ndarray,
        A numpy array containing optimized bitstrings.
            
    cs : np.ndarray,
        A numpy array containing cost functions associated 
        with xs.
        
    ls : np.ndarray,
        A numpy array containing provisions associated
        with xs.
        
    x_best : string,
        The best bitstring.
        
    c_best : float,
        The minimum cost function.
        
    l_best : float,
        The minimum combined provisios.
        
    """
    
    def __init__(self, x0, Qs, As, Ls, e = 1/20):
        
        assert isinstance(x0, str)
        assert len(x0) == Qs.shape[0]
        for char in x0:
            assert int(char) < Qs.shape[1]
        assert Qs.ndim == 2
        assert As.ndim == 2
        assert As.shape[0] == As.shape[1] 
        assert As.shape[0] == Qs.shape[0]
        assert isinstance(e, float) and e >= 0 
        assert Ls.shape[0] == Qs.shape[0]
        assert Ls.shape[1] == Qs.shape[1]
        
        self.x0 = x0
        self.Qs = np.copy(Qs)
        self.h = (1-e)*np.copy(Qs)
        self.h[:,0] += -e*np.sum(As,axis=1)
        self.J = e*np.copy(As)
        self.y0 = e * np.sum(As)/2
        self.e = e
        self.Ls = Ls
         
        self.N = Qs.shape[0]
        self.M = Qs.shape[1]
        self.x_best = np.zeros((self.N, self.M),dtype=int)
        for i in range(self.N):
            self.x_best[i][int(x0[i])] = 1
        self.y_best = self.get_objective_function(self.x_best)
        self.l_best = self.get_provision(self.x_best)       
        self.q_best = None
        self.ys = [self.y_best]
        self.ls = [self.l_best]
        
        
    def optimized(self, maxiter = 300):
        
        count = 0
        for _ in tqdm(range(maxiter)): 
            rs = np.zeros((self.N,self.M))
            for i in range(self.N):                  
                for j in range(self.M):
                    new_x = self.update_string(self.x_best, i, j)
                    y = self.get_objective_function(new_x)
                    l = self.get_provision(new_x)         
                    reward = self.l_best - l
                    penalty = self.y_best - y
                    
                    if reward <0:
                        rs[i,j] = -float('inf')
                    elif reward ==0 and penalty > 0:
                        rs[i,j] = -float('inf')
                    elif reward >=0 and penalty<=0:
                        rs[i,j] = reward
                    elif reward >0 and penalty >0:
                        rs[i,j] = -penalty/reward
             
            i, j = np.unravel_index(rs.argmax(), rs.shape)
            self.x_best = self.update_string(self.x_best, i, j) 
            self.y_best = self.get_objective_function(self.x_best)
            self.l_best = self.get_provision(self.x_best)
                
            self.ys += [self.y_best]
            self.ls += [self.l_best]   
                          
        self.q_best = np.sum(np.multiply(self.Qs, self.x_best)) 
         

    def update_string(self, bitstring, asset_i, action_j):
        x = np.copy(bitstring)
        x[asset_i] = 0
        x[asset_i][action_j] = 1
        return x    
        
    def get_objective_function(self, state):
        
        y =  np.sum(np.multiply(self.h, state))        
        ind = np.where(self.x_best[:,0]==1)[0]     
        y += 0.5 * np.sum(self.J[ind][:, ind])
        y += self.y0
        
        return y
    
    def get_provision(self, state):
        
        l = np.sum(np.multiply(self.Ls, state))
            
        return l
