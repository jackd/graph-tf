# Deep Adaptive Graph Neural Networks

- [Original paper](https://arxiv.org/abs/2007.09296)
- [Repository](https://github.com/divelab/DeeperGNN)

```bibtex
@inproceedings{liu2020towards,
  title={Towards deeper graph neural networks},
  author={Liu, Meng and Gao, Hongyang and Ji, Shuiwang},
  booktitle={Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  pages={338--348},
  year={2020}
}
```

## Example Usage

```bash
python -m graph_tf gtf_config/build_and_fit_many.gin dagnn/config/cora.gin
# Results for 10 runs
# test_acc           = 0.8444000124931336 +- 0.007486009643831605
# test_cross_entropy = 0.5873610854148865 +- 0.007901289048583457
# test_loss          = 0.5873610854148865 +- 0.007901289048583457
# val_acc            = 0.8185998737812042 +- 0.0066962702391993485
# val_cross_entropy  = 0.6317150473594666 +- 0.008671557681520656
# val_loss           = 0.6317151069641114 +- 0.008671557681520656
python -m graph_tf gtf_config/build_and_fit_many.gin dagnn/config/ogbn-arxiv.gin
# Results for 10 runs
# test_acc           = 0.6246898651123047 +- 0.012437198327089652
# test_cross_entropy = 1.3348363518714905 +- 0.06424626698565031
# test_loss          = 1.3348363518714905 +- 0.06424626698565031
# val_acc            = 0.629608428478241 +- 0.015791071644523334
# val_cross_entropy  = 1.3252739667892457 +- 0.07164184400635916
# val_loss           = 1.3252738475799561 +- 0.07164184400635916
```
