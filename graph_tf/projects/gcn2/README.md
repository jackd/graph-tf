# Simple and Deep Graph Convolutional Networks

- [Original repository](https://github.com/chennnM/GCNII)
- [Paper](https://arxiv.org/abs/2007.02133)

```bib
@inproceedings{chen2020simple,
  title={Simple and deep graph convolutional networks},
  author={Chen, Ming and Wei, Zhewei and Huang, Zengfeng and Ding, Bolin and Li, Yaliang},
  booktitle={International Conference on Machine Learning},
  pages={1725--1735},
  year={2020},
  organization={PMLR}
}
```

## Example Usage

```bash
python -m graph_tf gtf_config/build_and_fit_many.gin gcn2/config/cora.gin
# Results for 10 runs
# test_acc           = 0.8545000076293945 +- 0.004341646017109917
# test_cross_entropy = 0.823865783214569 +- 0.024747363998443975
# test_loss          = 1.2384095072746277 +- 0.015054614122431285
# val_acc            = 0.8285999059677124 +- 0.005730632686449124
# val_cross_entropy  = 0.8536475956439972 +- 0.02475158411881512
# val_loss           = 1.268191432952881 +- 0.015248619789555107
python -m graph_tf gtf_config/build_and_fit_many.gin gcn2/config/cora.gin --bindings='simplified=True'
# Results for 10 runs
# test_acc           = 0.8547000110149383 +- 0.003287883729224653
# test_cross_entropy = 0.8241403043270111 +- 0.02036100917403657
# test_loss          = 1.2157257795333862 +- 0.00869156844737108
# val_acc            = 0.8263998985290527 +- 0.005919480694722561
# val_cross_entropy  = 0.8534304857254028 +- 0.020568948196874097
# val_loss           = 1.2450160741806031 +- 0.008460546101803574
python -m graph_tf gtf_config/build_and_fit_many.gin gcn2/config/cora.gin --bindings='
  simplified=True
  conv_dropout=False
'
# Results for 10 runs
# test_acc           = 0.8486999928951263 +- 0.003742979661631753
# test_cross_entropy = 0.6870098412036896 +- 0.02131706612788612
# test_loss          = 1.0851202249526977 +- 0.008258629235820113
# val_acc            = 0.8229999125003815 +- 0.006826423059466659
# val_cross_entropy  = 0.7214565575122833 +- 0.019256222366008744
# val_loss           = 1.1195670247077942 +- 0.0076674725392891895
python -m graph_tf gtf_config/build_and_fit_many.gin gcn2/config/cora.gin --bindings='
  simplified=True
  conv_dropout=False
  dropout_rate=0.7
'
# Results for 10 runs
# test_acc           = 0.852200037240982 +- 0.00331063034892334
# test_cross_entropy = 0.7384769976139068 +- 0.021446698405096744
# test_loss          = 1.1294996738433838 +- 0.00830149789506858
# val_acc            = 0.8259999215602875 +- 0.007589458204241094
# val_cross_entropy  = 0.7696070194244384 +- 0.02127185443782645
# val_loss           = 1.1606297969818116 +- 0.008615941273693263
python -m graph_tf gtf_config/build_and_fit_many.gin gcn2/config/cora.gin --bindings='
  simplified=True
  conv_dropout=False
  dropout_rate=0.75
'
# Results for 10 runs
# test_acc           = 0.8539000153541565 +- 0.004437357893413145
# test_cross_entropy = 0.7979124128818512 +- 0.024716058942872116
# test_loss          = 1.1721490263938903 +- 0.009692080408024257
# val_acc            = 0.8249999105930328 +- 0.0073348554730584235
# val_cross_entropy  = 0.8243353128433227 +- 0.02282343831634654
# val_loss           = 1.198572039604187 +- 0.008394771902370942
python -m graph_tf gtf_config/build_and_fit_many.gin gcn2/config/cora.gin --bindings='
  simplified=True
  conv_dropout=False
  dropout_rate=0.8
'
# Results for 10 runs
# test_acc           = 0.8487000048160553 +- 0.0034942824517043927
# test_cross_entropy = 0.8906352162361145 +- 0.02303830386772426
# test_loss          = 1.2403384566307067 +- 0.00966315728152276
# val_acc            = 0.8273998916149139 +- 0.00566038888321871
# val_cross_entropy  = 0.9120997726917267 +- 0.019529055776501613
# val_loss           = 1.261803114414215 +- 0.006758780876095281
python -m graph_tf gtf_config/build_and_fit_many.gin gcn2/config/pubmed.gin
python -m graph_tf gtf_config/build_and_fit_many.gin gcn2/config/pubmed.gin --bindings='simplified=True'
```
