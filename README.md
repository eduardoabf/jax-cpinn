# jax-cpinn
This repository contains the Adaptive Comptetitive Gradient Descent (ACGD) algorithm implementation (see [[1]](https://arxiv.org/abs/1905.12103) and [[2]](https://arxiv.org/abs/1910.05852)) and a basic Competitive Physics Informed Neural Network (CPINN) implementation in JAX. For understanding the implemented CPINN, see its [original article](https://arxiv.org/abs/2204.11144).

The ACGD implementation is based on [this code](https://github.com/wagenaartje/torch-cgd) made for PyTorch and the CPINN one on this [repository/paper](https://github.com/comp-physics/CPINN).

## Features and usage
This project is built in a script form. To run it, simply execute the relevant script files for each desired PDE, preferably in an IDE. Parameter modification is currently only possible by modifying the script itself.

Additionally, this CPINN implementation includes **Fourier Features** into this competitive setting. This feature can be turned on and off as required and easily adjusted. The implemented Fourier Features are based on [this article](https://arxiv.org/abs/2006.10739).

## Note
As of 17/03/25, this repository is on a preliminary state and there will likely be many refactorings on the code.

In the CPINNs folder you can find working examples of CPINNs combined with the ACGD.

Some other PDE examples will be added over time.

## Cite
In case you use this code, here's the citation for it:

```
@misc{jax-cpinn,
  author = {Eduardo Ferreira},
  title = {jax-cpinn: A JAX implementation of Competitive Physics Informed Neural Networks based on the Adaptive Competitive Gradient Descent},
  year = {2025},
  url = {https://github.com/eduardoabf/jax-cpinn/}
}
```
