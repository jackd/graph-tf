# Graph Convolutional Networks

- [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907) (ICLR 2017)
- [Author's repository](https://github.com/tkipf/gcn)

```bibtex
@inproceedings{kipf2017semi,
  title={Semi-Supervised Classification with Graph Convolutional Networks},
  author={Kipf, Thomas N. and Welling, Max},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2017}
}
```

See [examples/gcn/citations.py](../../examples/gcn/citations.py) for training script.

## MultiGraphConv Benchmarks

Forward / Backward times for convolutions. See [ops_benchmark.py](ops_benchmark.py).

- WT: Walltime (ms)
- Mem: Memory (mb)
- F: forward
- B: Forward + backward pass

| Version | SparseImpl | WT F | WT B| Mem F| MeM B|
| ------- |:----------:| ----:|----:|-----:|-----:|
| 0       | coo        |   43 |  46 | 36.5 | 32.0|
| 0       | csr        |   43 |  47 | 34.0 | 32.0|
| 1       | coo        |   44 |  47 | 53.0 | 44.1|
| 1       | csr        |   45 |  47 | 53.0 | 44.1|
| 2       | coo        |  220 | 229 | 40.0 | 46.5|
| 2       | csr        |  213 | 224 | 40.0 | 44.1|
