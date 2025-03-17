import jax
import jax.numpy as jnp
from jax import random
from functools import partial


class JaxNeuralNetwork:
    """
    Class for creating densely connected neural networks in Jax
    """

    def __init__(self, layers: list, activation_func, dtype: jnp.dtype = jnp.float32, output_modificator = None, random_key = None) -> None:

        self.layers = layers

        self.activation = activation_func

        # Function to be executed on each input point on each forward pass
        # Default function just returns the passed input
        self.data_normalization_func = lambda data: data

        # Weights and biases of each neural network layer
        # list of tuples [(W1, B1), (W2, B2), ...] where W1 represents the weights of the first nn layer and B1 its biases
        self.weights_biases: jnp.array

        self.output_modificator = output_modificator

        # Function that computes a forward pass of the neural network for batches of points
        self.forward: function

        # Function that computes a forwards pass for a single data point
        self.forward_single_data_point: function

        # For now, initialized as default values
        if not (dtype == jnp.float32 or dtype == jnp.float64):
            raise TypeError(
                f"Usupported dtype. Attribute must belong to the class jax.numpy. Supported options are jax.numpy.float32 or jax.numpy.float64 but got {dtype} instead."
            )
        self.nn_dtype = dtype

        self.random_key = random.key(56226487) if random_key == None else random_key

        self.step_size = 0.001

        # Fourier features embedding
        self.ff_kernel = None

    def __call__(self, input_data: jnp.array) -> jnp.array:
        return self.forward(self.weights_biases, input_data)

    def set_nn_initializer(self, initializer: jax.nn.initializers):
        self._initializer = initializer

    def set_data_normalization_func(self, func):
        """
        Function that will be applied to every data point that is being fed to the neural network.
        This means that before the data point reaches the first layer, it will be normalized by this function.

        Args:
            func (_function_): function should be in the following format: func(data_point) -> normalized_data_point
        """
        self.data_normalization_func = func

    def initialize_ff_kernel(self, n_features, std_dev, input_data, jax_random_key):
        """
        Initializes the fourier features kernel by sampling the entries b_ij of a matrix B (n_features x d) through a normal distribution
        N(0, std_dev)

        Params:
            n_features {integer} -- total number of Fourier Features to be created
            std_dev {float} -- standard deviation of the normal distribution N(0, std_dev)
            input_data {array} -- input dataset of shape n x d, where d is the dimension of each data point and n the number of datapoints
            jax_random_key {integer} -- number used as seed for the random normal distribution sampler
        """
        self.ff_kernel = jax.nn.initializers.normal(std_dev)(jax_random_key, (input_data[0].shape[-1], n_features))

    def build(self, initializer : jax.nn.initializers = None):
        """
        Builds the defined neural network, initializing the weights and biases based on the given initializer

        Args:
            initializer (jax.nn.initializers, optional): Weight and bias initializer function from JAX. 
            See documentation of the jax.nn.initializers.
            Defaults to None which invokes a plain random value generator for initializing the neural network weights and biases.
        """
        self.init_weights_and_biases(initializer)
        # Adjust nn forward pass to receive batches of data
        if self.ff_kernel is None:
            self.forward = jax.vmap(self._forward, in_axes=(None, 0))
            self.forward_single_data_point = self._forward
        else:
            self.forward = jax.vmap(self._forward_ff_features, in_axes=(None, 0))
            self.forward_single_data_point = self._forward_ff_features

        if self.output_modificator is not None:
            self.forward = jax.vmap(self._forward_mod, in_axes=(None, 0))
            self.forward_single_data_point = self._forward_mod

    # TODO: i don't like this... Improve?
    def parameters(self):
        """
        Returns all trainable parameters from the neural network
        """
        return jnp.concat([elem.flatten() for tup in self.weights_biases for elem in tup])

    def loss(self, nn_params, input_data, u_sol):
        """
        Builds the neural network loss
        """
        u_pred = self.forward(nn_params, input_data)
        return -jnp.mean(u_pred * u_sol)

    @partial(jax.jit, static_argnums=0)
    def update(self, nn_params, input_data, u_sol):
        grads = jax.grad(self.loss, argnums=0)(nn_params, input_data, u_sol)
        return [(W - self.step_size * grad_W, b - self.step_size * grad_b) for (W, b), (grad_W, grad_b) in zip(self.weights_biases, grads)]

    def _forward(self, nn_params, data_point):
        """
        Executes a forward pass in the neural network for a SINGLE data point
        """
        data_point = self.data_normalization_func(data_point)
        activations = data_point
        for w, b in nn_params[:-1]:
            outputs = jnp.dot(w, activations) + b
            activations = self.activation(outputs)

        final_w, final_b = nn_params[-1]

        return jnp.dot(final_w, activations) + final_b

    def _forward_ff_features(self, nn_params, data_point):
        """
        Executes a forward pass in the neural network for a SINGLE data point.
        Here, a layer of Fourier Features is added to the forward pass
        """
        data_point = self.data_normalization_func(data_point)

        # Apply FF features on the normalized data
        data_point = jnp.concat((jnp.cos(jnp.pi * jnp.dot(data_point, self.ff_kernel)), jnp.sin(jnp.pi * jnp.dot(data_point, self.ff_kernel))), axis=-1)

        activations = data_point
        for w, b in nn_params[:-1]:
            outputs = jnp.dot(w, activations) + b
            activations = self.activation(outputs)

        final_w, final_b = nn_params[-1]

        return jnp.dot(final_w, activations) + final_b

    def _forward_mod(self, nn_params, data_point):
        """
        Executes a modified forward pass in the neural network for a SINGLE data point
        """
        data_point = self.data_normalization_func(data_point)

        activations = data_point
        for w, b in nn_params[:-1]:
            outputs = jnp.dot(w, activations) + b
            activations = self.activation(outputs)

        final_w, final_b = nn_params[-1]
        pred_pre = jnp.dot(final_w, activations) + final_b

        return self.output_modificator(pred_pre)

    # Initialize all layers for a fully-connected neural network based on each layer size
    def init_weights_and_biases(self, initializer: jax.nn.initializers = None):
        """
        Initializes all weights and biases of the neural network based on a initialization function.
        If no initializer is passed, a default random initializer will be applied.

        Returns -- list of tuples [(W1, B1), (W2, B2), ...] where W1 represents the weights of the first nn layer and B1 its biases

        Params:
        initializer -- should be a function from the jax.nn.initializers module (see documentation)
        key         -- the random key to be utilized by the random number generator
        """

        # Check if Fourier Features have been initialized to adjust the NN layers
        if self.ff_kernel is not None:
            self.layers[0] = 2 * self.ff_kernel.shape[-1]

        keys = random.split(self.random_key, len(self.layers))
        if initializer is None:
            self.weights_biases = [self.random_layer_params(m, n, k) for m, n, k in zip(self.layers[:-1], self.layers[1:], keys)]
        else:
            self.weights_biases = [self.weight_biases_initializer(initializer, m, n, k) for m, n, k in zip(self.layers[:-1], self.layers[1:], keys)]

    # A helper function to randomly initialize weights and biases for a dense neural network layer
    def random_layer_params(self, m, n, key, scale=1):
        w_key, b_key = random.split(key)
        return scale * random.uniform(w_key, (n, m), self.nn_dtype, -jnp.sqrt(1 / m), jnp.sqrt(1 / m)), scale * random.uniform(
            b_key, (n,), self.nn_dtype, -jnp.sqrt(1 / m), jnp.sqrt(1 / m))

    def weight_biases_initializer(self, initializer: jax.nn.initializers, m, n, key):
        """
        Initializes weights and biases based on given jax.nn.initializer

        Returns a tuple (weights, biases) with the initialized weights and biases for the respective layer
        """
        w_key, b_key = random.split(key)
        return initializer(w_key, (n, m), self.nn_dtype), initializer(b_key, (1, n), self.nn_dtype).flatten()
