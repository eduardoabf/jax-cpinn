# jax-cpinn
This repository contains the Adaptive Comptetitive Gradient Descent (ACGD) algorithm implementation and a basic Competitive Physics Informed Neural Network (CPINN) implementation in JAX.
The ACGD implementation is based on [this code](https://github.com/wagenaartje/torch-cgd) made for PyTorch and the CPINN one on this [repository/paper](https://github.com/comp-physics/CPINN).

## Note
As of 02/03/25, this repository is on a preliminary state for now and there will be likely some refactorings on the code.

In the CPINNs/burgers folder you can find a working example of CPINNs combined with the ACGD.

Some other PDEs examples will be added over time.

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
