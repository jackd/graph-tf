# Spectral Graph Autoencoder

Unpublished work in progress. The idea is to train an `MLP` that maps spectral properties to node embeddings `Z` such that `Z @ Z^T \approx A`.

## Usage

```bash
python -m graph_tf gtf_config/build_and_fit.gin \
    sgae/config/cora.gin \
    sgae/config/spectral_only.gin  # only spectral features
```

```bash
python -m graph_tf gtf_config/build_and_fit.gin \
    sgae/config/cora.gin
# Results on test data:
# auc_pr  : 0.9019970893859863
# auc_roc : 0.8846589922904968
# loss    : 0.5009334087371826
```

```bash
python -m graph_tf gtf_config/build_and_fit.gin     \
    sgae/config/cora.gin \
    --bindings='hidden_units=256'
# Results on test data:
# auc_pr  : 0.9115762114524841
# auc_roc : 0.8897521495819092
# loss    : 0.49455249309539795
python -m graph_tf gtf_config/build_and_fit.gin     \
    sgae/config/cora.gin \
    --bindings='
    hidden_units=256
    dropout_rate=0.5
    '
# Results on test data:
# auc_pr  : 0.9373770356178284
# auc_roc : 0.932487964630127
# loss    : 0.4775604009628296
python -m graph_tf gtf_config/build_and_fit.gin     \
    sgae/config/cora.gin \
    --bindings='
    hidden_units=256
    dropout_rate=0.5
    embedding_dim=64
    '
# Results on test data:
# auc_pr  : 0.9449596405029297
# auc_roc : 0.9381895065307617
# loss    : 0.46803468465805054
python -m graph_tf gtf_config/build_and_fit.gin     \
    sgae/config/cora.gin \
    --bindings='
    hidden_units=256
    dropout_rate=0.5
    embedding_dim=128
    '
# Results on test data:
# auc_pr  : 0.9467020630836487
# auc_roc : 0.9395939111709595
# loss    : 0.462597131729126
python -m graph_tf gtf_config/build_and_fit.gin     \
    sgae/config/cora.gin \
    --bindings='
    hidden_units=256
    dropout_rate=0.6
    embedding_dim=128
    '
# Results on test data:
# auc_pr  : 0.9476111531257629
# auc_roc : 0.9452251195907593
# loss    : 0.4711951017379761
python -m graph_tf gtf_config/build_and_fit.gin     \
    sgae/config/cora.gin \
    --bindings='
    hidden_units=256
    dropout_rate=0.6
    embedding_dim=16
    '
# Results on test data:
# auc_pr  : 0.9394339323043823
# auc_roc : 0.935258686542511
# loss    : 0.47687819600105286
```

```bash
python -m graph_tf gtf_config/build_and_fit_many.gin \
    sgae/config/cora.gin \
    --bindings='
      hidden_units=256
      dropout_rate=0.6
      embedding_dim=128
'
# Results for 10 runs
# auc_pr  = 0.9416941463947296 +- 0.005090645688478682
# auc_roc = 0.9392603993415832 +- 0.005630622663431584
# loss    = 0.4731168359518051 +- 0.007443133890974455
python -m graph_tf gtf_config/build_and_fit_many.gin \
    sgae/config/cora.gin \
    --bindings='
      hidden_units=256
      dropout_rate=0.6
      embedding_dim=128
      validation_edges_in_adj = True
'
# Results for 10 runs
# auc_pr  = 0.9429558753967285 +- 0.005996034836560965
# auc_roc = 0.9406437575817108 +- 0.00549132862827191
# loss    = 0.4713103950023651 +- 0.007990480919046184
```

### v2

```bash
python -m graph_tf gtf_config/build_and_fit.gin sgae/config/ca-AstroPh_v2.gin --bindings='embedding_dim=128'
# loss: 0.1053 - auc_roc: 0.9963 - auc_pr: 0.9960 - val_loss: 0.1882 - val_auc_roc: 0.9833 - val_auc_pr: 0.9842
python -m graph_tf gtf_config/build_and_fit.gin sgae/config/ca-AstroPh_v2.gin --bindings='embedding_dim=64'
# loss: 0.0967 - auc_roc: 0.9973 - auc_pr: 0.9970 - val_loss: 0.1610 - val_auc_roc: 0.9878 - val_auc_pr: 0.9877
python -m graph_tf gtf_config/build_and_fit.gin sgae/config/ca-AstroPh_v2.gin --bindings='embedding_dim=32'
# loss: 0.0962 - auc_roc: 0.9975 - auc_pr: 0.9973 - val_loss: 0.1947 - val_auc_roc: 0.9843 - val_auc_pr: 0.9846
python -m graph_tf gtf_config/build_and_fit.gin sgae/config/ca-AstroPh_v2.gin --bindings='embedding_dim=16'
# loss: 0.1101 - auc_roc: 0.9966 - auc_pr: 0.9965 - val_loss: 0.2166 - val_auc_roc: 0.9806 - val_auc_pr: 0.9798
python -m graph_tf gtf_config/build_and_fit.gin sgae/config/ca-AstroPh_v2.gin --bindings='embedding_dim=8'
# loss: 0.1698 - auc_roc: 0.9898 - auc_pr: 0.9900 - val_loss: 0.3651 - val_auc_roc: 0.9596 - val_auc_pr: 0.9602
```

v3:

```bash
python -m graph_tf gtf_config/build_and_fit.gin sgae/config/ca-AstroPh_v3.gin --bindings='embedding_dim=16'
# loss: 0.0013 - auc_roc: 0.9806 - auc_pr: 0.9792 - val_loss: 0.0013 - val_auc_roc: 0.9819 - val_auc_pr: 0.9808
python -m graph_tf gtf_config/build_and_fit.gin sgae/config/ca-AstroPh_v3.gin --bindings='embedding_dim=128'
# loss: 0.0013 - auc_roc: 0.9841 - auc_pr: 0.9831 - val_loss: 0.0013 - val_auc_roc: 0.9838 - val_auc_pr: 0.9834
```
