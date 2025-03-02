import jax
import jax.numpy as jnp
import numpy as np
import scipy
from jax import jit
from functools import partial
import jax.flatten_util as jfu
from jax import config
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg

class ACGD:
    def __init__ (self, min_params, max_params, minimax_f, lr=1e-3, beta=0.9, eps=1e-3, solver=None):
        '''
        JAX Implementation of the Adaptive Competitive Gradient Descent (ACGD) found in "Implicit competitive regularization in GANs" by F. Schäfer et al.

        This implementation makes use of scipy optimizers with JIT compiled operators from JAX
        Args:
            min_params: Parameters of the model (i.e. Neural Network) that MINIMIZES minimax_f.
            max_params: Parameters of the model (i.e. Neural Network) that MAXIMIZES minimax_f.
            minimax_f: minimax game objective function
            lr: Learning rate.
            beta: Exponential decay rate for the second moment estimates.
            eps: Small constant for numerical stability.
            solver: Linear system solver (so far only set up GMRES).
        '''

        # Define minimax game objecive function
        self.f = minimax_f

        # Store arguments in class
        self.min_params = min_params # list of tuples [(weights_layer1, bias_layer1), (weights_layer2, bias_layer2), ...]
        self.max_params = max_params

        # One can also define different learning rates for each agent. Here we work with the same learning rate for both
        self.eta = lr
        self.beta = beta
        self.eps = eps
        
        # Count number of parameters of each model
        self.n_min = sum([jnp.size(wx) + jnp.size(bx)  for wx, bx in self.min_params])
        self.n_max = sum([jnp.size(wy) + jnp.size(by) for wy, by in self.max_params])

        # First derivatives of the loss function in vectorized form
        self.df_dmin_func = jit(jax.grad(self.vectorized_f, argnums=0))
        self.df_dmax_func = jit(jax.grad(self.vectorized_f, argnums=1))
        
        # List to store the number of gmres iterations
        self.gmres_iterations = []

        # Define which linear system solver to use
        if solver is None:
            self.solver = jax.scipy.sparse.linalg.gmres
        else:
            self.solver = solver
        
        # Set overall dtype to be used thourout all computations (done based on the minimizing model's parameters)
        self.dtype = jnp.dtype(self.min_params[0][0])

        # Residual tolerance for the iterative solver (so far onyl using GMRES)
        self.gmres_rtol = 1e-10 if self.dtype == jnp.float64 else 1e-7
        
        # Initialize second moment estimates
        self.vx = jnp.zeros(self.n_min, dtype=self.dtype)
        self.vy = jnp.zeros(self.n_max, dtype=self.dtype)

        # Initialize iteration counter
        self.it = 0

    def vectorized_f(self, min_param_vec, max_param_vec):
        """
        Vectorized version of the minimax game objective function.
        Use this so that working with the hessians and derivatives becomes more intuitive.

        Args:
            min_param_vec: minimizing model parameters vector
            max_param_vec: maximizing model parameters vector

        Returns:
            The minimax objective function evaluated at f(min_params_vec, max_params_vec)
        """
        min_param = convert_array_to_tree_structure(self.min_params, min_param_vec)
        max_param = convert_array_to_tree_structure(self.max_params, max_param_vec)
        return self.f(min_param, max_param)

    @partial(jax.jit, static_argnums = 0)
    def hvp_df_dmin(self, min_params_vec, max_params_vec, v):
        isolated_df_dmin_func = lambda max_params : self.df_dmin_func(min_params_vec, max_params)
        return hvp(isolated_df_dmin_func, (max_params_vec,), (v,))
    
    @partial(jax.jit, static_argnums = 0)
    def hvp_df_dmax(self, max_params_vec, min_params_vec, v):
        isolated_df_dmax_func = lambda min_params : self.df_dmax_func(min_params, max_params_vec)
        return hvp(isolated_df_dmax_func, (min_params_vec,), (v,))

    def create_linear_system(self, min_params, max_params, vars_state_dict) -> None:
        '''
        Computes the input values to create the linear system to be solved during the ACGD

        Args: 
            min_params: Parameters of the model that minimizes f.
            max_params: Parameters of the model that maximizes f.
            vars_state_dict: dictionary with the current variable values (i.e state) of the optimizer
        '''

        # Increase iteration
        vars_state_dict['it'] += 1
        x_params_vec = convert_params_to_numpy(min_params)
        y_params_vec = convert_params_to_numpy(max_params)
        
        # jax.debug.print("Time step: {t}", t = time_step)
        # Compute first order partial derivatives
        df_dmin = self.df_dmin_func(x_params_vec, y_params_vec)
        df_dmax = self.df_dmax_func(x_params_vec, y_params_vec)

        # Update second moment estimates
        # Let's compute the variables below in the same tree strucutre as df_dx
        vx = self.beta * vars_state_dict['v_min'] + (1 - self.beta) * jnp.square(df_dmin)
        vy = self.beta * vars_state_dict['v_max'] + (1 - self.beta) * jnp.square(df_dmax)

        # Compute learning rates (with bias correction)
        bias_correction = 1 - self.beta**vars_state_dict['it']
        eta_min = jnp.sqrt(bias_correction) * self.eta / (jnp.sqrt(vx) + self.eps)
        eta_max = jnp.sqrt(bias_correction) * self.eta / (jnp.sqrt(vy) + self.eps)

        # Update Variables state dictionary
        vars_state_dict['v_min'] = vx
        vars_state_dict['v_max'] = vy
        vars_state_dict['eta_min'] = eta_min
        vars_state_dict['eta_max'] = eta_max
        vars_state_dict['df_dmin'] = df_dmin
        vars_state_dict['df_dmax'] = df_dmax
        
        return x_params_vec, y_params_vec, vars_state_dict

        
    def solve_gmres(self, min_params_vec, max_params_vec, vars_state_dict):
        
        # Unpack vars in dict
        eta_min = vars_state_dict['eta_min']
        eta_max = vars_state_dict['eta_max']

        # Initialize MatMul operator
        A1 = Operator(self, max_params_vec, min_params_vec, eta_max, eta_min, (self.n_min, self.n_min))

        # Here we only need to compute the paramter update of one agent. Since we already know what the minimizing model's "action" is,
        # one can simply compute the maximizing model's "reaction" to it.
        # Still, one can also have calculte delta_max using its own linear system as given below.
        #A2 = Operator(df_dy, df_dx, self.min_params, self.max_params, eta_min, eta_max)
        #b2 = eta_max.sqrt() * (df_dy - vectorize(autograd.grad(df_dx, self.max_params, eta_min * df_dx, retain_graph=True)))
        #dy = + eta_max.sqrt() * self.solver.solve(A2, b2).view(-1)

        # Compute b1 (i.e. right hand side of the system A1.x = b1)
        eta_max_df_dy_product = eta_max * vars_state_dict['df_dmax']
        df_dmaxdmin_vp = self.hvp_df_dmin(min_params_vec, max_params_vec, eta_max_df_dy_product)
        b1 = jnp.sqrt(eta_min) * (vars_state_dict['df_dmin'] + df_dmaxdmin_vp) 

        # Initialize empty gmres iteration counter
        counter = Gmres_it_counter()

        # Solve the linear system using given solver
        prev_sol, info = scipy.sparse.linalg.gmres(A1, b1, rtol=self.gmres_rtol, atol=1e-20, restart=1000, maxiter=1, callback=counter, callback_type='pr_norm')
        self.gmres_iterations.append(counter.iter)

        return prev_sol
    
    def update_params(self, min_params, max_params, min_params_vec, max_params_vec, prev_sol, vars_state_dict):
        
        # Compute Minimizing and Maximizing neural networks parameter update
        delta_min = -jnp.sqrt(vars_state_dict['eta_min']) * prev_sol
        hessian_vp = self.hvp_df_dmax(max_params_vec, min_params_vec, delta_min)
        delta_max = vars_state_dict['eta_max'] * (vars_state_dict['df_dmax'] + hessian_vp)
        
        # Update parameters
        update_param = lambda param, delta : param + delta
        delta_min = convert_array_to_tree_structure(min_params, delta_min)
        delta_max = convert_array_to_tree_structure(max_params, delta_max)

        return jax.tree.map(update_param, min_params, delta_min), jax.tree.map(update_param, max_params, delta_max)

class Operator(scipy.sparse.linalg.LinearOperator):
    def __init__ (self, acgd_instance, first_params, second_params, eta_first, eta_second, shape : tuple):
        '''
        ACGD matrix-vector multiplication operator. 
        This operator characterizes matrix vaector product of ONE OF the matrices below (see ACGD algorithm).
        The sub-index notation here is kept the same as in the ACGD article.

            Matrix 1: (I + A^{1/2}_{x,t} D^2_{xy}f A_{y,t} D^2_{yx}f A^{1/2}_{x,t})
            Matrix 2: (I + A^{1/2}_{y,t} D^2_{yx}f A_{x,t} D^2_{xy}f A^{1/2}_{y,t})
        
        where A is a diagonal matrix with eta_first or eta_second on its diagonal.

        Which of the paramters is the "first" or "second" depends on the chosen matrix above and its corresponding differential operator D.
        For instance, if one chooses to compute Matrix 1 the first differential operator that appears in its expression is 'D^2_{xy}'. This means
        that the first paramters needed to compute the second derivative, are the paramters corresponding to 'y'. Analogously, the second parameters
        should be the ones corresponding to model 'x', given the second differential operator 'D^2_{yx}'. 

        Args:
            acgd_instance: instance of the ACGD class
            first_params: The parameters to compute the second derivative of the first first derivative.
            second_params: The parameters to compute the second derivative of the second first derivative.
            eta_first: Learning rates of the parameters used for the first second derivative.
            eta_second: Learning rates of the parameters used for the second second derivative.
            shape: the shape/dimension of the multiplying matrix above.
        '''

        # Store arguments in class
        self.first_hvp_func = acgd_instance.hvp_df_dmin
        self.second_hvp_func = acgd_instance.hvp_df_dmax
        self.first_params = first_params
        self.second_params = second_params
        self.first_eta = eta_first
        self.second_eta = eta_second
        self.shape = shape
        self.dtype = jnp.dtype(first_params[0])

    def _matvec (self, v: jnp.array) -> jnp.array:
        '''
        Performs a matrix-vector product.
        '''

        # From right to left. The equations in the comments assume the matrix used in the pseudocode to calculate dx. 
        # A^{1/2}_{y,t} * v
        v0 = jnp.sqrt(self.second_eta) * v
        
        # First Hessian-vector product (invert the order from the pytorch code i.e. change dx_dy by dy_dx)
        # A_{y,t} D^2_{xy}f * v0
        first_hessian = self.second_hvp_func(self.first_params, self.second_params, v0)
        v1 = self.first_eta * first_hessian 

        # Second Hessian-vector product
        # A^{1/2}_{x,t} D^2{xy}f * v1
        second_hessian = self.first_hvp_func(self.second_params, self.first_params, v1) 
        v2 = jnp.sqrt(self.second_eta) * second_hessian
        
        # I v + v2
        result = v + v2
        return np.array(result) #return numpy array since this is the expected data type in SciPy

class Gmres_it_counter(object):
    def __init__(self, show=False):
        self._show = show
        self.iter = 0
    def __call__(self, rk=None):
        self.iter += 1
        if self._show:
            print('iter %3i\trk = %s' % (self.iter, str(rk)))

#Global functions
@partial(jit, static_argnums=(0,))    
def hvp(grad_func : jax.grad, primals, tangents):
    '''
    Efficiently computes the hessian vector product.
    '''
    # For information purposes. Usually, you don't want the hvp to be compiled more than once per hessian.
    print('Recompiling hvp')
    return jax.jvp(grad_func, primals, tangents)[1]

@jax.jit
def convert_params_to_numpy(params: list):
    x_0 = jax.tree_util.tree_flatten(params)
    return jnp.concatenate([wb.reshape(-1) for wb in x_0[0]], axis=0)

# def update_params(params: list, x_0: np.ndarray):
#     i = 0
#     for weights, biases in params:
#         k = weights.size
#         weights = jnp.array(x_0[i: i + k].reshape(weights.shape))
#         i += k
#         j = biases.size
#         biases = jnp.array(x_0[i: i + j].reshape(biases.shape))
#         i += j

def convert_array_to_tree_structure(params: list, x_0: np.ndarray):
    i = 0
    tree = []
    for weights, biases in params:
        k = weights.size
        w = jnp.array(x_0[i: i + k].reshape(weights.shape))
        i += k
        j = biases.size
        b = jnp.array(x_0[i: i + j].reshape(biases.shape))
        i += j
        tree.append((w, b))

    return tree