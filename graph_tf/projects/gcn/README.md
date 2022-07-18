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

## Example Usage

```bash
python -m graph_tf gtf_config/build_and_fit_many.gin gcn/config/cora.gin
# Results for 10 runs
# test_acc           = 0.816100001335144 +- 0.005503628960682206
# test_cross_entropy = 0.7648167014122009 +- 0.01088103861096708
# test_loss          = 1.0232022404670715 +- 0.012148206116460194
# val_acc            = 0.7909999132156372 +- 0.004753927153450943
# val_cross_entropy  = 0.8101876616477967 +- 0.012065930356040111
# val_loss           = 1.0685733318328858 +- 0.012282761192363281
python -m graph_tf gtf_config/build_and_fit_many.gin gcn/config/citeseer.gin
# Results for 10 runs
# test_acc           = 0.7072999894618988 +- 0.006197591939338053
# test_cross_entropy = 1.030307936668396 +- 0.015719112612142254
# test_loss          = 1.305362069606781 +- 0.016864348541310442
# val_acc            = 0.708600002527237 +- 0.00774855726168702
# val_cross_entropy  = 1.0521578669548035 +- 0.016078652755661334
# val_loss           = 1.3272119879722595 +- 0.017350497404265613
python -m graph_tf gtf_config/build_and_fit_many.gin gcn/config/pubmed.gin
# Results for 10 runs
# test_acc           = 0.7905998885631561 +- 0.00280000141694776
# test_cross_entropy = 0.5838302612304688 +- 0.009914963454400681
# test_loss          = 0.7253390729427338 +- 0.013433205561354114
# val_acc            = 0.7948000431060791 +- 0.0032496190127991025
# val_cross_entropy  = 0.5708036661148072 +- 0.009337772212486481
# val_loss           = 0.7123124122619628 +- 0.013085460346977707
python -m graph_tf gtf_config/build_and_fit_many.gin gcn/config/ogbn-arxiv.gin
# Results for 10 runs
# test_acc           = 0.6511491239070892 +- 0.003143990184403487
# test_cross_entropy = 1.2278876304626465 +- 0.007369359510327262
# test_loss          = 1.2571795344352723 +- 0.011218531502535628
# val_acc            = 0.6576127171516418 +- 0.0006238069823802658
# val_cross_entropy  = 1.2056769728660583 +- 0.0066253748447447185
# val_loss           = 1.2349687576293946 +- 0.01078163032951462
```

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
