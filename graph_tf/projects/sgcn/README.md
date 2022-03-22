# Spectral Graph Convolution Network

Probably a misnomer, and if not it's an overloaded term, but we define a spectral graph convolution with a single input and single output channel as `V @ f(W) @ V.T @ X`, there `V` is the set of eigenvectors associated with the smallest eigenvalues of `L`, `W` is a diagonal matrix of those eigenvalues and `f` is a learnable transformation.

## Example Usage

```bash
python -m graph_tf gtf_config/build_and_fit_many.gin sgcn/config/cora.gin
```
