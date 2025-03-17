import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import jax.numpy as jnp
import warnings
from CPINNs.JaxNeuralNetwork import JaxNeuralNetwork
from matplotlib.ticker import LogFormatterSciNotation, MaxNLocator
from scipy.spatial import cKDTree as KDTree

# Set global plotting parameters
matplotlib.rcParams['axes.linewidth'] = 0.3

# Set latex font
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
})

def save_weights_biases(nn_weights_biases, file_path, file_suffix):
    """
    Saves the NN weights and biases in an .npz file by using the numpy.savez function.
    - Weights file: weights.npz
    - Biases file: biases.npz

    """
    weights_list = []
    biases_list = []
    for weights, biases in nn_weights_biases:
        weights_list.append(weights)
        biases_list.append(biases)
    np.savez(file_path + "weights_" + file_suffix + ".npz", *weights_list)
    np.savez(file_path + "biases_" + file_suffix + ".npz", *biases_list)

def save_weights_biases_kernel(nn_weights_biases, ff_kernel, file_path, file_suffix):
    """
    Saves the NN weights, biases and fourier features Kernel (if available) in an .npz file by using the numpy.savez function.
    - Weights file: weights.npz
    - Biases file: biases.npz
    - Kernel file: kernel.npz
    """
    weights_list = []
    biases_list = []
    for weights, biases in nn_weights_biases:
        weights_list.append(weights)
        biases_list.append(biases)
    np.savez(file_path + "weights_" + file_suffix + ".npz", *weights_list)
    np.savez(file_path + "biases_" + file_suffix + ".npz", *biases_list)
    if ff_kernel is not None:
        np.savez(file_path + "kernel_" + file_suffix + ".npz", ff_kernel = ff_kernel)

def save_losses(l2_loss, cpinn_loss, pde_loss, boundary_loss, file_path, file_suffix):
    """
    Saves the NN losses in an .npz file by using the numpy.savez function.

    """
    np.savez(file_path + "l2_loss_" + file_suffix + ".npz", l2_loss = l2_loss)
    np.savez(file_path + "cpinn_loss_" + file_suffix + ".npz", cpinn_loss = cpinn_loss)
    np.savez(file_path + "pde_loss_" + file_suffix + ".npz", pde_loss = pde_loss)
    np.savez(file_path + "boundary_loss_" + file_suffix + ".npz", boundary_loss = boundary_loss)

def load_losses(file_path, file_suffix):
    """
    Load the losses arrays based on the file path and suffix 

    """
    l2 = np.load(file_path + "l2_loss_" + file_suffix + ".npz")['l2_loss']
    cpinn = np.load(file_path + "cpinn_loss_" + file_suffix + ".npz")['cpinn_loss']
    pde = np.load(file_path + "pde_loss_" + file_suffix + ".npz")['pde_loss']
    boundary = np.load(file_path + "boundary_loss_" + file_suffix + ".npz")['boundary_loss']
    try:
        gmres_its = np.load(file_path + "gmres_its_" + file_suffix + ".npz")['gmres_its']
    except:
        warnings.warn(("No GMRES iterations file found."))
        gmres_its = None

    return l2, cpinn, pde, boundary, gmres_its

def save_gmres_its(gmres_its, file_path, file_suffix):
    np.savez(file_path + "gmres_its_" + file_suffix + ".npz", gmres_its = gmres_its)

def load_weights_biases_in_nn(neural_net : JaxNeuralNetwork, weights_file_path, biases_file_path, ff_kernel_file_path = None):
    """
    Loads the weights and biases stored in a .npz file into the input neural network. 

    Args:
        neural_net (JaxNeuralNetwork): neural network to load the weights and biases into
        weights_file_path (string): file path of the weights to be loaded
        biases_file_path (string): file path of the biases to be loaded
        ff_kernel_file_path (string, optional): file path of the fourier features to be loaded
    """
    weights_file = np.load(weights_file_path)
    biases_file = np.load(biases_file_path)
    neural_net.weights_biases = [(weights_file[w_file].astype(neural_net.nn_dtype), biases_file[b_file].astype(neural_net.nn_dtype)) for w_file, b_file in zip(weights_file.files, biases_file.files)]
    if ff_kernel_file_path is not None:
        neural_net.ff_kernel = np.load(ff_kernel_file_path)['ff_kernel']

def plot_solution_burgers(generator_nn : JaxNeuralNetwork, discriminator_nn : JaxNeuralNetwork, input_data, input_targets, path, filename):
    """
    Generic plotter function.
    Generated Plots:
    1 - Generator Neural network solution
    2 - Solution error (i.e. target - computed value)
    3 - Assigned discriminator weights for each COLLOCATION point
    """

    font_size = 9

    fig, ax = plt.subplots(2, 2, figsize = (6, 6), layout='constrained')
    
    Y, X, Z = interpolate_to_regular_grid(input_data, input_targets, N = 1000, nnear = 8)
    
    # Reference solution
    sp1 = ax[0,0]
    scatter_1 = sp1.contourf(X, Y, Z[:,:,0], levels = 100, vmin = -1, vmax = 1)
    sp1.set_xlabel(r'$t$', size = font_size)
    sp1.set_ylabel(r'$x$', size = font_size)
    sp1.tick_params(axis='both', which='major', labelsize=font_size)
    cbar = fig.colorbar(scatter_1, ax = sp1, location='top', orientation='horizontal', pad  = 0.02)
    cbar.ax.set_xlabel(r"Exact $u(x, t)$ - $u$", size = font_size) 
    cbar.set_ticks([-1, 0, 1])
    cbar.ax.tick_params(axis='both', which='major', labelsize=font_size)
    sp1.text(-0.1, -1.75, '(a)', fontsize=font_size, va='bottom')

    # CPINN solution 
    Y, X, Z = interpolate_to_regular_grid(input_data, generator_nn.forward(generator_nn.weights_biases, input_data), N = 1000, nnear = 8)
    sp1 = ax[0,1]
    scatter_1 = sp1.contourf(X, Y, Z[:,:,0], levels = 100)
    sp1.set_xlabel(r'$t$', size = font_size)
    sp1.set_ylabel(r'$x$', size = font_size)
    sp1.tick_params(axis='both', which='major', labelsize=font_size)
    cbar = fig.colorbar(scatter_1, ax = sp1, location='top', orientation='horizontal', pad  = 0.02)
    cbar.ax.set_xlabel(r"CPINN $u(x, t)$ - $\mathcal{P}$", size = font_size) 
    cbar.locator = MaxNLocator(nbins=3)
    cbar.update_ticks()
    cbar.ax.tick_params(axis='both', which='major', labelsize=font_size)
    sp1.text(-0.1, -1.75, '(b)', fontsize=font_size, va='bottom')
    
    # Solution error
    sol_diff = input_targets - generator_nn.forward(generator_nn.weights_biases, input_data)
    Y, X, Z = interpolate_to_regular_grid(input_data, sol_diff, N = 1000, nnear = 8)
    # Adjust values data range
    min_val, max_val = adjust_min_max_range(np.min(sol_diff), np.max(sol_diff))
    sp1 = ax[1,0]
    lvls = np.linspace(min_val, max_val, 100)
    scatter_2 = sp1.contourf(X, Y, Z[:,:,0], levels = lvls, cmap = 'seismic')
    sp1.set_xlabel(r'$t$', size = font_size)
    sp1.set_ylabel(r'$x$', size = font_size)
    sp1.tick_params(axis='both', which='major', labelsize=font_size)
    cbar = fig.colorbar(scatter_2, ax = sp1, location='top', orientation='horizontal', pad  = 0.02)
    cbar.ax.set_xlabel(r"Solution error $(\mathcal{P} - u)$", size = font_size) 
    cbar.set_ticks([min_val, 0, max_val])
    cbar.ax.tick_params(axis='both', which='major', labelsize=font_size)
    cbar.ax.get_xaxis().set_major_formatter(LogFormatterSciNotation(base=10, minor_thresholds=(np.inf,np.inf), labelOnlyBase=False))
    sp1.text(-0.1, -1.75, '(c)', fontsize=font_size, va='bottom')
    
    # ATTENTION: careful with the discriminator output index. It should match the desired value
    # Assigned discriminator weights
    sp1 = ax[1,1]
    dis_weights = discriminator_nn.forward(discriminator_nn.weights_biases, input_data)[:, 1, None]
    Y, X, Z = interpolate_to_regular_grid(input_data, dis_weights, N = 1000, nnear = 8)
    min_val, max_val = adjust_min_max_range(np.min(dis_weights), np.max(dis_weights))
    lvls = np.linspace(min_val, max_val, 100)
    scatter_3 = sp1.contourf(X, Y, Z[:,:,0], levels = lvls, cmap = 'RdBu')
    sp1.set_xlabel(r'$t$', size = font_size)
    sp1.set_ylabel(r'$x$', size = font_size)
    sp1.tick_params(axis='both', which='major', labelsize=font_size)
    cbar = fig.colorbar(scatter_3, ax = sp1, location='top', orientation='horizontal', pad  = 0.02)
    cbar.ax.set_xlabel(r"Discriminator weights - PDE", size = font_size) 
    cbar.set_ticks([np.round(min_val,1), 0, np.round(max_val,1)])
    cbar.ax.tick_params(axis='both', which='major', labelsize=font_size)
    cbar.ax.get_xaxis().set_major_formatter(LogFormatterSciNotation(base=10, minor_thresholds=(np.inf,np.inf), labelOnlyBase=False))
    sp1.text(-0.1, -1.75, '(d)', fontsize=font_size, va='bottom')

    plt.savefig(path + '/' + filename, bbox_inches='tight', dpi = 300)
    plt.show()

def plot_solution_poisson(generator_nn : JaxNeuralNetwork, discriminator_nn : JaxNeuralNetwork, input_data, input_targets, path, filename):
    """
    Generic plotter function.
    Generated Plots:
    1 - Generator Neural network solution
    2 - Solution error (i.e. target - computed value)
    3 - Assigned discriminator weights for each COLLOCATION point
    """
    font_size = 9
    tx, ty = -2.3, -3.5

    fig, ax = plt.subplots(2, 2, figsize = (6, 6), layout='constrained')

    # Plot exact solution
    Y, X, Z = interpolate_to_regular_grid(input_data, input_targets, N = 1000, nnear = 8)
    sp1 = ax[0,0]
    scatter_1 = sp1.contourf(X, Y, Z[:,:,0], levels = 100, vmin = -1, vmax = 1, cmap = 'jet')
    sp1.set_xlabel(r'$x$', size = font_size)
    sp1.set_ylabel(r'$y$', size = font_size)
    sp1.tick_params(axis='both', which='major', labelsize=font_size)
    cbar = fig.colorbar(scatter_1, ax = sp1, location='top', orientation='horizontal', pad  = 0.02)
    cbar.ax.set_xlabel(r"Exact $u(x, y)$ - $u$", size = font_size) 
    cbar.set_ticks([-1, 0, 1])
    cbar.ax.tick_params(axis='both', which='major', labelsize=font_size)
    sp1.text(tx, ty, '(a)', fontsize=font_size, va='bottom')

    # Plot predicted solution
    Y, X, Z = interpolate_to_regular_grid(input_data, generator_nn.forward(generator_nn.weights_biases, input_data), N = 1000, nnear = 8)
    sp1 = ax[0,1]
    scatter_1 = sp1.contourf(X, Y, Z[:,:,0], levels = 100, cmap = 'jet')
    sp1.set_xlabel(r'$x$', size = font_size)
    sp1.set_ylabel(r'$y$', size = font_size)
    sp1.tick_params(axis='both', which='major', labelsize=font_size)
    cbar = fig.colorbar(scatter_1, ax = sp1, location='top', orientation='horizontal', pad  = 0.02)
    cbar.ax.set_xlabel(r"CPINN $u(x, t)$ - $\mathcal{P}$", size = font_size) 
    cbar.locator = MaxNLocator(nbins=3)
    cbar.update_ticks()
    cbar.ax.tick_params(axis='both', which='major', labelsize=font_size)
    sp1.text(tx, ty, '(b)', fontsize=font_size, va='bottom')
    
    # Plot solution error
    sol_diff = input_targets - generator_nn.forward(generator_nn.weights_biases, input_data)
    Y, X, Z = interpolate_to_regular_grid(input_data, sol_diff, N = 1000, nnear = 8)
    # Adjust values data range
    min_val, max_val = adjust_min_max_range(np.min(sol_diff), np.max(sol_diff))
    sp1 = ax[1,0]
    lvls = np.linspace(min_val, max_val, 100)
    scatter_2 = sp1.contourf(X, Y, Z[:,:,0], levels = lvls, cmap = 'seismic')
    sp1.set_xlabel(r'$x$', size = font_size)
    sp1.set_ylabel(r'$y$', size = font_size)
    sp1.tick_params(axis='both', which='major', labelsize=font_size)
    cbar = fig.colorbar(scatter_2, ax = sp1, location='top', orientation='horizontal', pad  = 0.02)
    cbar.ax.set_xlabel(r"Solution error $(\mathcal{P} - u)$", size = font_size) 
    cbar.set_ticks([min_val, 0, max_val])
    cbar.ax.tick_params(axis='both', which='major', labelsize=font_size)
    cbar.ax.get_xaxis().set_major_formatter(LogFormatterSciNotation(base=10, minor_thresholds=(np.inf,np.inf), labelOnlyBase=False))
    sp1.text(tx, ty, '(c)', fontsize=font_size, va='bottom')
    
    # Plot discriminator weights
    # ATTENTION: careful with the discriminator output index. It should match the desired value
    sp1 = ax[1,1]
    dis_weights = discriminator_nn.forward(discriminator_nn.weights_biases, input_data)[:, 1, None]
    Y, X, Z = interpolate_to_regular_grid(input_data, dis_weights, N = 1000, nnear = 8)
    min_val, max_val = adjust_min_max_range(np.min(dis_weights), np.max(dis_weights))
    lvls = np.linspace(np.round(min_val,1), np.round(max_val,1), 100)
    scatter_3 = sp1.contourf(X, Y, Z[:,:,0], levels = lvls, cmap = 'RdBu')
    sp1.set_xlabel(r'$x$', size = font_size)
    sp1.set_ylabel(r'$y$', size = font_size)
    sp1.tick_params(axis='both', which='major', labelsize=font_size)
    cbar = fig.colorbar(scatter_3, ax = sp1, location='top', orientation='horizontal', pad  = 0.02)
    cbar.ax.set_xlabel(r"Discriminator weights - PDE", size = font_size) 
    cbar.set_ticks([np.round(min_val,1), 0, np.round(max_val,1)])
    cbar.ax.tick_params(axis='both', which='major', labelsize=font_size)
    cbar.ax.get_xaxis().set_major_formatter(LogFormatterSciNotation(base=10, minor_thresholds=(np.inf,np.inf), labelOnlyBase=False))
    sp1.text(tx, ty, '(d)', fontsize=font_size, va='bottom')

    plt.savefig(path + '/' + filename + '.png', bbox_inches='tight', dpi = 300)
    plt.show()

def plot_solution_ac(generator_nn : JaxNeuralNetwork, discriminator_nn : JaxNeuralNetwork, input_data, input_targets, path, filename):
    """
    Plotter function created specifically to plot the results of the Allen-Cahn equation

    """

    font_size = 9
    tx, ty = -0.1, -1.75

    fig, ax = plt.subplots(2, 2, figsize = (6, 6), layout='constrained')
    
    # Plot exact solution
    Y, X, Z = interpolate_to_regular_grid(input_data, input_targets, N = 1000, nnear = 8)
    sp1 = ax[0,0]
    scatter_1 = sp1.contourf(X, Y, Z[:,:,0], levels = 100, vmin = -1, vmax = 1, cmap = 'jet')
    sp1.set_xlabel(r'$t$', size = font_size)
    sp1.set_ylabel(r'$x$', size = font_size)
    sp1.tick_params(axis='both', which='major', labelsize=font_size)
    # Colorbar
    cbar = fig.colorbar(scatter_1, ax = sp1, location='top', orientation='horizontal', pad  = 0.02)
    cbar.ax.set_xlabel(r"Exact $u(x, y)$ - $u$", size = font_size) 
    cbar.set_ticks([-1, 0, 1])
    cbar.ax.tick_params(axis='both', which='major', labelsize=font_size)
    sp1.text(tx, ty, '(a)', fontsize=font_size, va='bottom')

    # Plot predicted solution
    Y, X, Z = interpolate_to_regular_grid(input_data, generator_nn.forward(generator_nn.weights_biases, input_data), N = 1000, nnear = 8)
    sp1 = ax[0,1]
    scatter_1 = sp1.contourf(X, Y, Z[:,:,0], levels = 100, cmap = 'jet')
    sp1.set_xlabel(r'$t$', size = font_size)
    sp1.set_ylabel(r'$x$', size = font_size)
    sp1.tick_params(axis='both', which='major', labelsize=font_size)
    # Colorbar
    cbar = fig.colorbar(scatter_1, ax = sp1, location='top', orientation='horizontal', pad  = 0.02)
    cbar.ax.set_xlabel(r"CPINN $u(x, t)$ - $\mathcal{P}$", size = font_size) 
    cbar.locator = MaxNLocator(nbins=3)
    cbar.update_ticks()
    cbar.ax.tick_params(axis='both', which='major', labelsize=font_size)
    sp1.text(tx, ty, '(b)', fontsize=font_size, va='bottom')
    
    # Plot predicted solution error
    sol_diff = input_targets - generator_nn.forward(generator_nn.weights_biases, input_data)
    Y, X, Z = interpolate_to_regular_grid(input_data, sol_diff, N = 1000, nnear = 8)
    # Adjust values data range
    min_val, max_val = adjust_min_max_range(np.min(sol_diff), np.max(sol_diff))
    sp1 = ax[1,0]
    lvls = np.linspace(min_val, max_val, 100)
    scatter_2 = sp1.contourf(X, Y, Z[:,:,0], levels = lvls, cmap = 'seismic')
    sp1.set_xlabel(r'$t$', size = font_size)
    sp1.set_ylabel(r'$x$', size = font_size)
    sp1.tick_params(axis='both', which='major', labelsize=font_size)
    # Colorbar
    cbar = fig.colorbar(scatter_2, ax = sp1, location='top', orientation='horizontal', pad  = 0.02)
    cbar.ax.set_xlabel(r"Solution error $(\mathcal{P} - u)$", size = font_size) 
    cbar.set_ticks([min_val, 0, max_val])
    cbar.ax.tick_params(axis='both', which='major', labelsize=font_size)
    cbar.ax.get_xaxis().set_major_formatter(LogFormatterSciNotation(base=10, minor_thresholds=(np.inf,np.inf), labelOnlyBase=False))
    sp1.text(tx, ty, '(c)', fontsize=font_size, va='bottom')
    
    # ATTENTION: careful with the discriminator output index. It should match the desired value
    sp1 = ax[1,1]
    dis_weights = discriminator_nn.forward(discriminator_nn.weights_biases, input_data)[:, 3, None]
    Y, X, Z = interpolate_to_regular_grid(input_data, dis_weights, N = 1000, nnear = 8)
    min_val, max_val = adjust_min_max_range(np.min(dis_weights), np.max(dis_weights))
    lvls = np.linspace(np.round(min_val,1), np.round(max_val,1), 100)
    scatter_3 = sp1.contourf(X, Y, Z[:,:,0], levels = lvls, cmap = 'RdBu')
    sp1.set_xlabel(r'$t$', size = font_size)
    sp1.set_ylabel(r'$x$', size = font_size)
    sp1.tick_params(axis='both', which='major', labelsize=font_size)
    # Colorbar
    cbar = fig.colorbar(scatter_3, ax = sp1, location='top', orientation='horizontal', pad  = 0.02)
    cbar.ax.set_xlabel(r"Discriminator weights - PDE", size = font_size) 
    cbar.set_ticks([np.round(min_val,1), 0, np.round(max_val,1)])
    cbar.ax.tick_params(axis='both', which='major', labelsize=font_size)
    cbar.ax.get_xaxis().set_major_formatter(LogFormatterSciNotation(base=10, minor_thresholds=(np.inf,np.inf), labelOnlyBase=False))
    sp1.text(tx, ty, '(d)', fontsize=font_size, va='bottom')

    plt.savefig(path + '/' + filename, bbox_inches='tight', dpi = 300)
    plt.show()

def min_max(values):
    # Adjust values data range
    if np.abs(np.min(values)) > np.abs(np.min(values)):
        min_val = np.min(values)
        max_val = np.abs(min_val)
    else:
        max_val = np.max(values)
        min_val = -max_val
    min_val = 0 if np.min(values) >=0 else min_val
    return min_val, max_val


def adjust_min_max_range(min, max):
    if np.abs(min) > np.abs(np.min(max)):
        min_val = min
        max_val = np.abs(min)
    else:
        max_val = max
        min_val = -max
    
    abs_min = np.abs(min_val)
    if np.log10(abs_min) < 0:
        decimal_order = np.ceil(np.abs(np.log10(abs_min))) + 1
        rouding = np.ceil(abs_min * 10**(decimal_order))
        min_val = rouding/(10**(decimal_order)) * np.sign(min_val)
    else:
        decimal_order = np.floor(np.log10(abs_min)) - 1
        rouding = np.ceil(abs_min / 10**(decimal_order))
        min_val = rouding*(10**(decimal_order)) * np.sign(min_val)
    
    abs_max = np.abs(max_val)
    if np.log10(abs_max) < 0:
        decimal_order = np.ceil(np.abs(np.log10(abs_max))) + 1
        rouding = np.ceil(abs_max * 10**(decimal_order))
        max_val = rouding/(10**(decimal_order)) * np.sign(max_val)
    else:
        decimal_order = np.floor(np.log10(abs_max)) - 1
        rouding = np.ceil(abs_max / 10**(decimal_order))
        max_val = rouding*(10**(decimal_order)) * np.sign(max_val)

    return min_val, max_val


def plot_iteration_losses(l2_loss, cpinn_loss, pde_loss, boundary_loss, labels, path, filename, colors = None, legend = True):
    """
    Simply plots each loss value per iteration

    Args:
        l2_loss (_type_): _description_
        cpinn_loss (_type_): _description_
        pde_loss (_type_): _description_
        boundary_loss (_type_): _description_
    """

    # General plotting parameters
    font_size = 9

    if colors is None:
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

    fig, ax = plt.subplots(1,2, figsize = (6, 4.5))

    sp1 = ax[0]
    l = 0
    for l2 in l2_loss:
        sp1.plot([i for i in range(len(l2))], l2, label = labels[l] if len(labels) > 0 else None, c = colors[l])
        l += 1
    sp1.set_yscale('log')
    sp1.set_xlabel('Iteration', size = font_size)
    sp1.set_ylabel(r'$L_2$ Relative error', size = font_size)
    sp1.tick_params(axis='both', which='major', labelsize=font_size)
    sp1.set_box_aspect(aspect=1)
    sp1.ticklabel_format(axis='x', style='sci', scilimits=(3,3))

    sp3 = ax[1]
    l = 0
    for pde in pde_loss:
        sp3.plot([i for i in range(len(pde))], pde, label = labels[l] if len(labels) > 0 else None, c = colors[l])
        l += 1
    sp3.set_yscale('log')
    sp3.set_xlabel('Iteration', size = font_size)
    sp3.set_ylabel(f"PDE Loss", size = font_size)
    sp3.tick_params(axis='both', which='major', labelsize=font_size)
    sp3.set_box_aspect(aspect=1)
    sp3.ticklabel_format(axis='x', style='sci', scilimits=(3,3))

    if legend:
        plt.figlegend(labels, loc='upper center', ncols = 2, bbox_to_anchor=(0.55, 1 - 0.1), fontsize = font_size)
    plt.tight_layout()
    plt.savefig(path + '/' + filename + '_losses.png', bbox_inches='tight', dpi = 300)
    plt.show()

def plot_boundary_losses(boundary_loss, labels, path, filename, colors = None, legend = True):
    """Simply plots each loss value per iteration

    Args:
        l2_loss (_type_): _description_
        cpinn_loss (_type_): _description_
        pde_loss (_type_): _description_
        boundary_loss (_type_): _description_
    """

    # General plotting parameters
    font_size = 9

    if colors is None:
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

    fig, ax = plt.subplots(1,1, figsize = (3, 4.5))

    sp1 = ax
    l = 0
    for bc_loss in boundary_loss:
        sp1.plot([i for i in range(len(bc_loss))], bc_loss, label = labels[l] if len(labels) > 0 else None, c = colors[l])
        l += 1
    sp1.set_yscale('log')
    sp1.set_xlabel('Iteration', size = font_size)
    sp1.set_ylabel(r'Boundary Condition Loss', size = font_size)
    sp1.tick_params(axis='both', which='major', labelsize=font_size)
    sp1.set_box_aspect(aspect=1)
    sp1.ticklabel_format(axis='x', style='sci', scilimits=(3,3))

    if legend:
        plt.figlegend(labels, loc='upper center', ncols = 2, bbox_to_anchor=(0.6, 1 - 0.1), fontsize = font_size)
    plt.tight_layout()
    plt.savefig(path + '/' + filename + '_boundary.png', bbox_inches='tight', dpi = 300)
    plt.show()

def plot_gmres_its(gmres_its, labels, path, filename, colors = None, legend = True):
    """
    Simply plots the number of gmres iterations per ACGD iteration

    Args:
        l2_loss (_type_): _description_
        cpinn_loss (_type_): _description_
        pde_loss (_type_): _description_
        boundary_loss (_type_): _description_
    """

    # General plotting parameters
    font_size = 9

    if colors is None:
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

    fig, ax = plt.subplots(1,1, figsize = (3, 4.5))

    sp1 = ax
    l = 0
    for it in gmres_its:
        sp1.plot([i for i in range(len(it))], it, label = labels[l] if len(labels) > 0 else None, c = colors[l])
        l += 1
    sp1.set_xlabel('Iteration', size = font_size)
    sp1.set_ylabel('GMRES Iterations', size = font_size)
    sp1.tick_params(axis='both', which='major', labelsize=font_size)
    sp1.set_box_aspect(aspect=1)
    sp1.ticklabel_format(axis='x', style='sci', scilimits=(3,3))

    if legend:
        plt.figlegend(labels, loc='upper center', ncols = 2, bbox_to_anchor=(0.6, 1 - 0.1), fontsize = font_size)
    
    plt.tight_layout()
    plt.savefig(path + '/' + filename + '_gmres_its.png', bbox_inches='tight', dpi = 300)
    plt.show()

# Requries:
# http://docs.scipy.org/doc/scipy/reference/spatial.html
class Invdisttree:
    """inverse-distance-weighted interpolation using KDTree:
    invdisttree = Invdisttree( coordinates_base, solution_base )  -- data points, values
    interpol = invdisttree( q, nnear=3, eps=0, p=1, weights=None)
        interpolates z from the 3 points nearest each query point q;
        For example, interpol[ a query point q ]
        finds the 3 data points nearest q, at distances d1 d2 d3
        and returns the IDW average of the values z1 z2 z3
            (z1/d1 + z2/d2 + z3/d3)
            / (1/d1 + 1/d2 + 1/d3)
            = .55 z1 + .27 z2 + .18 z3  for distances 1 2 3
    Modified Code from https://stackoverflow.com/questions/3104781/inverse-distance-weighted-idw-interpolation-with-python
    """

    def __init__(self, coordinates_base, solution_base, leafsize=10):
        # build the tree
        self.tree = KDTree(coordinates_base, leafsize=leafsize)
        # if solution_base.ndim == 1:
        #     self.solution_base = solution_base
        # else:
        #     self.solution_base = solution_base
        self.solution_base = solution_base

    def __call__(self, q, nnear=10, eps=0, p=2, distance_equality=1e-6, posinf=0.0):

        assert nnear <= self.solution_base.shape[0], "Cannot use more points than are available in the data set"

        # nnear nearest neighbours of each query point --
        distances, ix = self.tree.query(q, k=nnear, eps=eps)
        interpol = np.empty(((q.shape[0],) + np.shape(self.solution_base[0])))

        solutions = self.solution_base[ix]
        weights = np.nan_to_num(1 / distances**p, posinf=posinf)

        if len(weights.shape) == 1:
            weights = weights.reshape(-1, 1)
            distances = distances.reshape(-1, 1)
            solutions = solutions.reshape(solutions.shape[0], -1, solutions.shape[1])

        # option to filter points which are close to itentical
        weights_sum = weights.sum(axis=1, keepdims=True)

        interpol = np.sum(weights[:, :, np.newaxis] * solutions, axis=1) / weights_sum * (distances[:, 0:1] > distance_equality) + solutions[:, 0, :] * (
            distances[:, 0:1] <= distance_equality
        )

        return interpol
    
def interpolate_to_regular_grid(coords, vals, N=200, nnear=8, plot_settings={}):

    if plot_settings.get("Coordinate_Limits", None) == None:
        x = np.linspace(np.min(coords[:, 0]), np.max(coords[:, 0]), N)
        y = np.linspace(np.min(coords[:, 1]), np.max(coords[:, 1]), N)
    else:
        limits = plot_settings["Coordinate_Limits"]
        x = np.linspace(limits[0][0], limits[0][1], N)
        y = np.linspace(limits[1][0], limits[1][1], N)

    X, Y = np.meshgrid(x, y)
    coords_plot = np.concatenate((X.reshape(-1, 1), Y.reshape(-1, 1)), axis=1)
    inter_res = Invdisttree(coords, vals)

    Z = inter_res(coords_plot, nnear=nnear).reshape(N, N, -1)

    return X, Y, Z