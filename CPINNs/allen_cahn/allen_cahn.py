# Set envirnonment variables
import sys
from pathlib import Path
sys.path.insert(1, './')

from CPINNs.JaxNeuralNetwork import JaxNeuralNetwork as JaxNN
import jax
import jax.numpy as jnp
import scipy
import matplotlib.pyplot as plt
import numpy as np
import CPINNs.utils as utils
import scipy.sparse.linalg

from ACGD import JACGD
from datetime import datetime
from pyDOE import lhs
from jax import random
from jax import config
from data_loader import ACDataSampler

#====================================================General parameters============================================================
# Here you find some very general paramters for running this model. Since this is built like a script for now, you can quickly 
# edit some paramters here or modify the script below to further change the model (like the used activation functions for example).

# Global data type to be used, 64bit recommended
data_type = jnp.float64                                         

# Input data
input_data_path = 'CPINNs/allen_cahn/data/AC.mat'
n_colloc = 10000
n_boundary = 256
n_init = 100

# Spatial domain upper and lower bounds ([x, y] coordinates)
lb = [-1, 0]
ub = [1, 1]
n_mesh = 256 # dimension of domain mesh point e.g. 256 x 256

# File suffix for loading/saving the data
save_folder = 'CPINNs/allen_cahn/'

# Where to save the iteration losses
path_save_losses = save_folder + 'losses/'      

# Where to save the NNs weights and biases checkpoint
path_save_checkpoint = save_folder + 'checkpoint_weights_biases/' 
          
# File suffix for saving weights and biases of the current NNs
file_suffix_save = 'allen_cahn'

# File suffix for loading previous weights and biases into the current NN. If if None, nothing is loaded
file_suffix_load = None #'allen_cahn'      

# Trainig parameters
training_its = 1000

# NN checkpoint saving training iterations interval 
save_checkpoint_its = 100
                                     
# ACGD system eigenvalues computation interval
compute_eigvals = False
compute_eigvals_its = 500

# Whether to include a Fourier Features layer in both NNs                  
fourier_features = False
#==================================================================================================================================

#==========================================================Script==================================================================
if data_type == jnp.float64:
    config.update("jax_enable_x64", True)

# First create a NN with all biases being intialized as 0
class AllenCahnNN(JaxNN):
        
    # Custom initializer to set the biases to 0
    def weight_biases_initializer(self, initializer : jax.nn.initializers, m, n, key):
        '''
        Initializes weights and biases based on given jax.nn.initializer
        
        Returns - a tuple (weights, biases) with the initialized weights and biases for the respective layer 
        '''
        w_key, b_key = random.split(key)
        return initializer(w_key, (n, m), self.nn_dtype), jnp.zeros(n, self.nn_dtype)

# Read input data
data = ACDataSampler(input_data_path, [-1, 0], [1, 1], n_boundary, n_colloc, n_init, data_type)

# Initialize the Neural Networks
layers = [2, 30, 30, 20, 20, 20, 1]
layers_d = [2, 30, 30, 20, 20, 20, 20, 4]
G = AllenCahnNN(layers, jnp.tanh, dtype=data_type)
D = AllenCahnNN(layers_d, jax.nn.gelu, dtype=data_type)

# Set data normalization function in both NNs
G.set_data_normalization_func(lambda data_point: 2.0*(data_point - data.lb)/(data.ub - data.lb) - 1.0)
D.set_data_normalization_func(lambda data_point: 2.0*(data_point - data.lb)/(data.ub - data.lb) - 1.0)

# Initialize default Fourier Features Kernel
if fourier_features:
    G.initialize_ff_kernel(32, 1.0, data.X_star, random.key(257589637))
    D.initialize_ff_kernel(32, 1.0, data.X_star, random.key(257589637))

# Build the neural networks
G.build(jax.nn.initializers.glorot_normal()) 
D.build(jax.nn.initializers.glorot_normal())

# Build the loss function
def u_sum(x, t, g_weights_biases):
    input_X = jnp.concat([x, t], axis=1)
    return jnp.sum(G.forward(g_weights_biases, input_X))

@jax.jit
def u_x_t(x, t, g_weights_biases):
    return jax.grad(u_sum, [0, 1])(x, t, g_weights_biases)

@jax.jit
def u_x_sum(x, t, g_weights_biases):
    return jnp.sum(u_x_t(x, t, g_weights_biases)[0])

@jax.jit
def u_xx(x, t, g_weights_biases):
    return jax.grad(u_x_sum, 0)(x, t, g_weights_biases)

@jax.jit
def PDE_residual(x, t, g_weights_biases):
    input_X = jnp.concat([x, t], axis=1)
    u = G.forward(g_weights_biases, input_X)
    u_t = u_x_t(x, t, g_weights_biases)[1]
    return u_t - 0.0001*u_xx(x, t, g_weights_biases) + 5*jnp.pow(u, 3) - 5*u 

def boundary_diff_u(g_weights_biases):
    '''
    u(t, -1) = u(t, 1) - note that here we reverse the inputs to (x, t)
    '''
    u_t_neg_ones = G.forward(g_weights_biases, data.X_bc_minus_ones)
    u_t_ones = G.forward(g_weights_biases, data.X_bc_ones)
    return (u_t_ones - u_t_neg_ones)

def boundary_diff_u_x(g_weights_biases):
    '''
    u_x(t, -1) = u_x(t, 1) - note that here we reverse the inputs to (x, t)
    '''
    ux_t_neg_ones = u_x_t(data.x_bc_minus_ones, data.t_bc, g_weights_biases)[0]
    ux_t_ones = u_x_t(data.x_bc_ones, data.t_bc, g_weights_biases)[0]
    return (ux_t_ones - ux_t_neg_ones)

def loss_boundary_u(g_weights_biases):
    return jnp.mean(jnp.square(boundary_diff_u(g_weights_biases)))

def loss_boundary_u_x(g_weights_biases):
    return jnp.mean(jnp.square(boundary_diff_u_x(g_weights_biases)))

def loss_pde(g_weights_biases): 
    f_pred = PDE_residual(data.x_f, data.t_f, g_weights_biases)   
    return jnp.mean(jnp.square(f_pred))

def initial_cond_diff(g_weights_biases):
    u_pred_ic = G.forward(g_weights_biases, data.X_ic) 
    return (data.u_ic - u_pred_ic)

@jax.jit
def loss_cpinn(g_weights_biases, d_weights_biases):
    d_output_boundary_u = D.forward(d_weights_biases, data.X_bc_ones)[:,0]
    d_output_boundary_u_x = D.forward(d_weights_biases, data.X_bc_ones)[:,1]
    d_output_ic = D.forward(d_weights_biases, data.X_ic)[:,2]
    d_output_pde = D.forward(d_weights_biases, data.X_f_train)[:,3]
    
    cpinn_loss_boundary_u = d_output_boundary_u * boundary_diff_u(g_weights_biases)[:,0]
    cpinn_loss_boundary_u_x = d_output_boundary_u_x * boundary_diff_u_x(g_weights_biases)[:,0]
    cpinn_loss_ic = d_output_ic * initial_cond_diff(g_weights_biases)[:,0]
    cpinn_loss_pde = d_output_pde * PDE_residual(data.x_f, data.t_f, g_weights_biases)[:,0]

    return jnp.mean(cpinn_loss_boundary_u) + jnp.mean(cpinn_loss_boundary_u_x) + jnp.mean(cpinn_loss_ic) + jnp.mean(cpinn_loss_pde)

# Training loop

# Optimizer
optimizer = JACGD.ACGD(G.weights_biases, D.weights_biases, loss_cpinn, lr = 1e-3, eps = 1e-6, beta = 0.99, solver=None)
create_system = jax.jit(optimizer.create_linear_system)
update_params = jax.jit(optimizer.update_params)
start_time = datetime.now()

# Initialize variable state dictionary
vars_state_dict = {
    'v_min' : optimizer.vx,
    'v_max' : optimizer.vy,
    'it' : optimizer.it,
    'eta_min' : None,
    'eta_max' : None,
    'df_dmin' : None,
    'df_dmax' : None
}

# Load previous NN weights and biases
if file_suffix_load != None:
    if not fourier_features:
        utils.load_weights_biases_in_nn(G, path_save_checkpoint + "/weights_gen_" + file_suffix_load + ".npz", path_save_checkpoint + "/biases_gen_" + file_suffix_load + ".npz")
        utils.load_weights_biases_in_nn(D, path_save_checkpoint + "/weights_dis_" + file_suffix_load + ".npz", path_save_checkpoint + "/biases_dis_" + file_suffix_load + ".npz")
    else:
        utils.load_weights_biases_in_nn(G, path_save_checkpoint + "/weights_gen_" + file_suffix_load + ".npz", path_save_checkpoint + "/biases_gen_" + file_suffix_load + ".npz", path_save_checkpoint + "/kernel_gen_" + file_suffix_load + ".npz")
        utils.load_weights_biases_in_nn(D, path_save_checkpoint + "/weights_dis_" + file_suffix_load + ".npz", path_save_checkpoint + "/biases_dis_" + file_suffix_load + ".npz", path_save_checkpoint + "/kernel_gen_" + file_suffix_load + ".npz")

# Initialize lists to save iteration results
l2_loss = []
cpinn_loss = []
pde_loss = []
boundary_loss = []

# Training loop
for i in range(training_its):

    g_params_vec, d_params_vec, vars_state_dict = create_system(G.weights_biases, D.weights_biases, vars_state_dict)
    prev_sol = optimizer.solve_gmres(g_params_vec, d_params_vec, vars_state_dict)

    # Generate and save the ACGD linear system to be solved with gmres every 1000 its
    if (compute_eigvals and ((i+1) % compute_eigvals_its == 0)):
        print("Compute ACGD linear system eigenvalues")
        t_start = datetime.now()
        
        # Build matrix-vector product oeprator based on the current system state
        OP = JACGD.Operator(optimizer, d_params_vec, g_params_vec, vars_state_dict['eta_max'], vars_state_dict['eta_min'], (optimizer.n_min, optimizer.n_min))
        
        # Save generated eigenvalues
        eigvals, eigvecs = scipy.sparse.linalg.eigs(OP, k = jnp.shape(vars_state_dict['eta_min'])[0] - 2, which='LM')
        np.savez_compressed("CPINNNs/allen_cahn/acgd_eigenvalues/eigvals_" + file_suffix_save + "_it" + str(i) + ".npz", eigvals = eigvals)
        
        print(f"Eigenvalues computation time: {datetime.now() - t_start}")

    G.weights_biases, D.weights_biases = update_params(G.weights_biases, D.weights_biases, g_params_vec, d_params_vec, prev_sol, vars_state_dict)

    # Print discriminator loss every 100 iterations
    if i % 100 == 0:
        d_output_boundary_u = D.forward(D.weights_biases, np.hstack((data.x_bc_ones, data.t_bc)))[:,0]
        d_output_pde = D.forward(D.weights_biases, data.X_f_train)[:,3]
        print(f"Max weight [boundary_u - PDE]: {jnp.max(d_output_boundary_u)} - {jnp.max(d_output_pde)}")
        print(f"Avg. weight [boundary_u - PDE]: {jnp.average(d_output_boundary_u)} - {jnp.average(d_output_pde)}")
        print(f"Median weight [boundary_u - PDE]: {jnp.median(d_output_boundary_u)} - {jnp.median(d_output_pde)}") 
    
    # Compute and save losses
    l2_loss.append(np.linalg.norm(G.forward(G.weights_biases, data.X_star) - data.u_star, 2) / np.linalg.norm(data.u_star, 2))
    cpinn_loss.append(loss_cpinn(G.weights_biases, D.weights_biases))
    pde_loss.append(loss_pde(G.weights_biases))
    boundary_loss.append(loss_boundary_u(G.weights_biases))

    #Save generator NN weights and biases every 1000 iterations
    if (i+1) % 100 == 0:
        utils.save_weights_biases_kernel(G.weights_biases, G.ff_kernel, path_save_checkpoint, "gen_" + file_suffix_save)
        utils.save_weights_biases_kernel(D.weights_biases, D.ff_kernel, path_save_checkpoint, "dis_" + file_suffix_save)
        utils.save_losses(l2_loss, cpinn_loss, pde_loss, boundary_loss, path_save_losses, file_suffix_save)
        utils.save_gmres_its(optimizer.gmres_iterations, path_save_checkpoint, file_suffix_save)
    
    print(f"{str(i):<6} Loss CPINN: {cpinn_loss[-1]:4.15f} - PDE loss: {pde_loss[-1]:4.15f} - BC loss: {boundary_loss[-1]:4.15f} - L2_Loss:  {l2_loss[-1]:4.15f}")

print("RUNTIME: ", datetime.now() - start_time)

# PLOTTING
path = 'CPINNs/allen_cahn/solution_plots'
filename_solution = 'allen_cahn.png'
filename_it_losses = 'ac_losses.png'
labels = ['Run 1']
colors = ['#a00000']

utils.plot_solution_ac(G, D, data.X_star, data.u_star, path, filename_solution)
utils.plot_iteration_losses(l2_loss, cpinn_loss, pde_loss, boundary_loss, labels, path, filename_it_losses, colors = colors)