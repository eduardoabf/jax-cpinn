import jax.numpy as jnp
import numpy as np
import scipy
from jax import random
from pyDOE import lhs

# This file generates training data
class ACDataSampler():
    def __init__(self, file_path : str, lb : np.array, ub : np.array, n_boundary : int, n_colloc : int, n_init : int, data_type : jnp.dtype):
        """
        1D Allen-Cahn equation Data sampler based on Latin Hypercube Sampling (LHS).

        Args:
            lb (np.array): Domain lower bounds i.e. [x, t] coordinates
            ub (np.array): Domain upper bounds i.e. [x, t] coordinates
            n_boundary (int): Number of boundary condition points (spatial)
            n_colloc (int): Number of collocation points
            n_init (int): number of initial condition points (t = 0)
        """

        self.data_path = file_path
        self.data_type = data_type
        self.n_boundary = n_boundary
        self.n_colloc = n_colloc
        self.n_init = n_init
        self.lb = jnp.array(lb, dtype=data_type)
        self.ub = jnp.array(ub, dtype=data_type)

        self.t_bc = None    	        # Time coordinate at the boundary points
        self.x_bc_ones = None           # x coordinate at the boundary points x = 1
        self.x_bc_minus_ones = None     # x coordinate at the boundary points x = -1
        self.X_f_train = None           # Collocation points [x, t]
        self.x_f = None                 # Collocation x coordinate only
        self.t_f = None                 # Collocation t cooridnate only
        self.X_star = None              # All available domain points
        self.u_star = None              # Solution value at the domain points above
        self.x_ic = None                # Initial condition (t=0) x cooridnates
        self.t_ic = None                # Corresponding t coordinate (i.e. t = 0) for each point above
        self.X_bc_ones = None           # Boundary condition coordinates in a (n_bound x 2) matrix form at x = 1
        self.X_bc_minus_ones = None     # Boundary condition coordinates in a (n_bound x 2) matrix form at x = -1
        self.u_ic = None                # Solution valua at the initial condition points
        self.X_ic = None                # Initial condition coordinates in a (n_init x 2) matrix form

        self.sample_data()
    
    def sample_data(self):
        
        # Load data
        data = scipy.io.loadmat(self.data_path)
        t = data['tt'].flatten()[:,None]
        x = data['x'].flatten()[:,None] 
        exact_sol = jnp.real(data['uu']).T

        # Domain
        X, T = jnp.meshgrid(x.flatten(),t.flatten())
        self.X_star = jnp.hstack((X.flatten()[:,None], T.flatten()[:,None]))
        self.u_star = exact_sol.flatten()[:,None]           

        # Initial condition points
        key_1, key_2 = random.split(random.key(64928345)) 
        self.x_ic = random.uniform(key_1, (self.n_init, 1)) * 2 - 1
        self.t_ic = jnp.zeros_like(self.x_ic)
        self.u_ic = self.x_ic ** 2 * jnp.cos(jnp.pi * self.x_ic)
        self.X_ic = jnp.hstack((self.x_ic, self.t_ic), dtype = self.data_type)

        # Boundary Condition points 
        self.t_bc = random.uniform(key_2, (self.n_boundary, 1), dtype = self.data_type)
        self.x_bc_ones = jnp.ones((self.n_boundary, 1), dtype = self.t_bc.dtype)
        self.x_bc_minus_ones = jnp.full((self.n_boundary, 1), -1, dtype = self.t_bc.dtype)
        self.X_bc_ones = jnp.hstack((self.x_bc_ones, self.t_bc))
        self.X_bc_minus_ones = jnp.hstack((self.x_bc_minus_ones, self.t_bc))

        # Collocation points
        self.X_f_train = self.lb + (self.ub - self.lb)*lhs(2, self.n_colloc)
        self.x_f = self.X_f_train[:,0:1]
        self.t_f = self.X_f_train[:,1:2]