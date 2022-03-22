# Learned Factorized GCN

Similar to FGCN, but using a learned `F`.

`lfgcn(A, x) = F @ F.T @ x`, with additional loss `|A @ x - lfgcn(A, x)|`.

## Usage

```bash
python -m graph_tf gtf_config/build_and_fit_many.gin lfgcn/config/cora.gin
```
