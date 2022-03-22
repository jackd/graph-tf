# Factorized GCN

GCN with adjacency matrix replaced by a low-rank factorization of the transformed adjacency matrix.

## Example Usage

```bash
python -m graph_tf gtf_config/build_and_fit_many.gin fgcn/config/cora.gin
```

## Thoughts

- Using Frobenius norm + gradient descent, `F = V @ sqrt(W)` looks to be about the best factorization such that `F @ F.T \approx A`.
- Results are poor except for very high rank, which defeats the purpose.
