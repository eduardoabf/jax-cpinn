import jax.numpy as jnp
import numpy as np
from jax import random
from pyDOE import lhs

# This file generates training data
class PoissonDataSampler():
    def __init__(self, lb : np.array, ub : np.array, n_boundary : int, n_colloc : int, n_mesh : int, data_type : jnp.dtype):
        """
        2 dimensional Poisson problem Data sampler based on Latin Hypercube Sampling (LHS).

        Args:
            lb (np.array): Domain lower bounds i.e. [x, y] coordinates
            ub (np.array): Domain upper bounds i.e. [x, y] coordinates
        """
        self.n_boundary = n_boundary
        self.n_colloc = n_colloc
        self.dtype = data_type
        
        # Train data
        self.all_xy_train = None 
        self.xy_bc = None 
        self.u_bc = None
        self.xy_inside = None
        self.f_xy = None
        # Test data
        self.x_test = None
        self.y_test = None
        self.xy_test = None
        self.u_test = None
        self.f_test = None
        self.X = None
        self.Y = None
        self.U = None

        self.generate_train_data(lb, ub, n_boundary, n_colloc, data_type)
        self.generate_test_data(lb, ub, n_mesh, data_type)

    
    def generate_train_data(self, lb, ub, num_bc, num_f, data_type = jnp.float64):
        '''
        @param lb: 1d array specifying the lower bound of x and y
        @param ub: 1d array specifying the upper bound of x and y
        @param num_bc: number of points on each side of training region (total number of boundary points = 4 * num_bc)
        @param num_f: number of non-boundary interior points
        @param u: a method that takes in a 2d ndarray as input and returns value of u with given inputs
        @param f: a method that takes in [n * 2]tensors x, y as input and returns value of u_xx+u_yy with given inputs

        @return: boundary xy points and inside xy points concatenated, boundary xy points, boundary u values, interior xy points, u_xx+u_yy labels of the interior points
        '''

        # Random key
        key = random.key(6854645324)
        subkeys = random.split(key, 4)

        f = lambda x, y: -2 * jnp.sin(x) * jnp.cos(y)
        u = lambda xy: jnp.sin(xy[:, [0]]) * jnp.cos(xy[:, [1]])


        # create edges on x={-2, 2}, y={-2, 2}
        leftedge_x_y = jnp.vstack((lb[0] * jnp.ones(num_bc), lb[1] + (ub[1] - lb[1]) * random.uniform(subkeys[0], (num_bc,)) ), dtype=data_type).T
        leftedge_u = u(leftedge_x_y)
        rightedge_x_y = jnp.vstack((ub[0] * jnp.ones(num_bc), lb[1] + (ub[1] - lb[1]) * random.uniform(subkeys[1], (num_bc,)) ), dtype=data_type).T
        rightedge_u = u(rightedge_x_y)
        topedge_x_y = jnp.vstack(( lb[0] + (ub[0] - lb[0]) * random.uniform(subkeys[2], (num_bc,)), ub[1] * jnp.ones(num_bc) ), dtype=data_type).T
        topedge_u = u(topedge_x_y)
        bottomedge_x_y = jnp.vstack((lb[0] + (ub[0] - lb[0]) * random.uniform(subkeys[3], (num_bc,)), lb[1] * jnp.ones(num_bc) ), dtype=data_type).T
        bottomedge_u = u(bottomedge_x_y)

        bc_x_y_train = jnp.vstack([leftedge_x_y, rightedge_x_y, bottomedge_x_y, topedge_x_y], dtype=data_type) #x,y pairs on boundaries
        bc_u_train = jnp.vstack([leftedge_u, rightedge_u, bottomedge_u, topedge_u], dtype=data_type) #u values on boundaries

        # Latin Hypercube sampling for collocation points
        # num_f sets of tuples(x,t)
        inside_xy = lb + (ub-lb) * lhs(2, num_f)

        # HERE we want code that also generates the training labels (values of f) for the interior points 
        all_xy_train = jnp.vstack((inside_xy, bc_x_y_train), dtype=data_type) # append training points to collocation points
        f_x_y = f(inside_xy[:, [0]], inside_xy[:, [1]])

        # Assign values
        self.all_xy_train = all_xy_train 
        self.xy_bc = bc_x_y_train 
        self.u_bc = bc_u_train 
        self.xy_inside = inside_xy 
        self.f_xy = f_x_y

    def generate_test_data(self, lb, ub, num, data_type = jnp.float64):

        u = lambda x, y: jnp.sin(x) * jnp.cos(y)
        f = lambda x, y: -2 * jnp.sin(x) * jnp.cos(y)

        X=jnp.linspace(lb[0], ub[0], num, dtype=data_type)
        Y=jnp.linspace(lb[1], ub[1], num, dtype=data_type)

        X, Y = jnp.meshgrid(X,Y) #X, Y are (num x num) matrices

        U = u(X,Y)
        u_test = U.flatten('F')[:,None]

        xy_test = jnp.hstack((X.flatten('F')[:,None], Y.flatten('F')[:,None]), dtype=data_type)
        f_test = f(xy_test[:,[0]], xy_test[:,[1]])

        x_test = xy_test[:,[0]]
        y_test = xy_test[:,[1]]

        self.x_test = x_test
        self.y_test = y_test
        self.xy_test = xy_test
        self.u_test = u_test
        self.f_test = f_test
        self.X = X
        self.Y = Y
        self.U = U