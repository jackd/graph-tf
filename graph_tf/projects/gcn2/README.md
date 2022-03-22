# Simple and Deep Graph Convolutional Networks

- [Original repository](https://github.com/chennnM/GCNII)
- [Paper](https://arxiv.org/abs/2007.02133)

## Example Usage

```bash
python -m graph_tf gtf_config/build_and_fit_many.gin gcn2/config/cora.gin
# Results for 10 runs
# test_acc           = 0.848600047826767 +- 0.005276381887511761
# test_cross_entropy = 1.042220902442932 +- 0.01709224545993005
# test_loss          = 1.5192195177078247 +- 0.031249051267891813
# val_acc            = 0.8173998773097992 +- 0.005444248778520441
# val_cross_entropy  = 1.061059057712555 +- 0.01576617323817087
# val_loss           = 1.538057792186737 +- 0.030793525113766113
```
