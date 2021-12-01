# Spectral Attention Graph Networks

For a connected graph, attention is interpretted as a learned mapping from node features to positive edge weights. These weights are used to compute a spectral embedding for each node based on the eigenvectors of the normalized weighted Laplacian.

## Implementation Efficiencies

Eigen-decomposition on sparse matrices of size `N >> 1` is potentially expensive. We make the computation viable in a deep learning setting by

- performing only a partial decomposition (i.e. only finding the eigenvectors associated with the smallest `M` eigenvalues, `M << N`);
- we use an iterative solver ([lobpcg][lobpcg]), using the decomposition from the previous training iteration as the initial estimate of the eigenvectors; and
- we compute derivatives based on [Steven Johnson method](https://math.mit.edu/~stevenj/18.336/adjoint.pdf).

Note layers and models here use `tf.Variable`s to track the previous decomposition. This makes them only applicable to datasets with a single large graph.

## TODO

- `jax` / `tf` implementation of `lobpcg`. Requires at least a generalized `eigh` implementation. See [jax issue](https://github.com/google/jax/issues/5461).

[lobpcg]: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lobpcg.html
