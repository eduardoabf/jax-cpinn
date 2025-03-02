import jax.numpy as jnp
import scipy.io
from jax import random
from pyDOE import lhs

class BurgersDataSampler():
    def __init__(self, file_path : str, n_boundary : int, n_colloc : int, data_type : jnp.dtype):
        self.file_path = file_path
        self.n_boundary = n_boundary
        self.n_colloc = n_colloc
        self.dtype = data_type

        # Loaded data attributes
        self.lb = None          # Domain lower bound
        self.ub = None          # Domain upper bound
        self.X_u_train = None   # Boundary domain points
        self.u_train = None     # Boundary solution value
        self.X_f_train = None   # Collocation points
        self.x_f = None         # x-coordinate of collocation points
        self.t_f = None         # t-coordinate of collocation points
        self.X_star = None      # All available domain points
        self.u_star = None      # Solution value at each domain point
        self.xu = None          # x-coordinate of domain points
        self.tu = None          # t-coordinate of domain points

        self.sample_data_lhs()

    def sample_data_lhs(self):
        """
        Samples data based on the Latin Hypercube Sampling (LHS) method
        """
        data = scipy.io.loadmat(self.file_path)

        t = data['t'].flatten()[:,None]
        x = data['x'].flatten()[:,None] 
        Exact = jnp.real(data['usol']).T

        X, T = jnp.meshgrid(x.flatten(),t.flatten())

        self.X_star = jnp.hstack((X.flatten()[:,None], T.flatten()[:,None]), dtype=self.dtype)
        self.u_star = Exact.flatten()[:,None]              

        # Domain bounds
        self.lb = self.X_star.min(0)
        self.ub = self.X_star.max(0)    

        # Boundary points
        # Data at t = 0    
        xx1 = jnp.hstack((X[0:1,:].T, T[0:1,:].T))
        uu1 = Exact[0:1,:].T
        # Data at X = -1 for all t's
        xx2 = jnp.hstack((X[:,0:1], T[:,0:1]))
        uu2 = Exact[:,0:1]
        # Data at X = 1 for all t's
        xx3 = jnp.hstack((X[:,-1:], T[:,-1:]))
        uu3 = Exact[:,-1:]

        self.X_u_train = jnp.vstack([xx1, xx2, xx3])
        self.X_f_train = self.lb + (self.ub-self.lb)*lhs(2, self.n_colloc)
        self.u_train = jnp.vstack([uu1, uu2, uu3])

        idx = random.choice(random.key(0), jnp.arange(self.X_u_train.shape[0]), shape =(self.n_boundary,), replace=False)
        self.X_u_train = self.X_u_train[idx, :]
        self.u_train = self.u_train[idx,:]

        self.x_u = self.X_u_train[:,0:1]
        self.t_u = self.X_u_train[:,1:2]
        self.x_f = self.X_f_train[:,0:1]
        self.t_f = self.X_f_train[:,1:2]