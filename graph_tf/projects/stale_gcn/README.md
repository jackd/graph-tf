# Stale GCN

Graph convolution networks that sub-batch using stale node features for non-batched nodes.

## Example Usage

```bash
python -m graph_tf stale_gcn/config/fit_and_test_many.gin stale_gcn/config/gcn/cora.gin
# Results for 10 runs
# test_acc           : 0.7616000056266785 +- 0.011534282749099066
# test_cross_entropy : 0.9842712879180908 +- 0.0549885906866896
# test_loss          : 0.9842712879180908 +- 0.0549885906866896
# val_acc            : 0.7547999024391174 +- 0.007547162607987161
# val_cross_entropy  : 0.9878926157951355 +- 0.051033062184595517
# val_loss           : 0.9878927290439605 +- 0.051033063771345984
python -m graph_tf stale_gcn/config/fit_and_test_many.gin stale_gcn/config/gcn/cite_seer.gin
# Results for 10 runs
# test_acc           : 0.6724999845027924 +- 0.008947083940708233
# test_cross_entropy : 1.0951887726783753 +- 0.03665642908056622
# test_loss          : 1.0951887726783753 +- 0.03665642908056622
# val_acc            : 0.6685999989509582 +- 0.013298131653806111
# val_cross_entropy  : 1.1299167633056642 +- 0.038982742639150886
# val_loss           : 1.1299167633056642 +- 0.038982742639150886
python -m graph_tf stale_gcn/config/fit_and_test_many.gin stale_gcn/config/gcn/pub_med.gin
# Results for 10 runs
# test_acc           : 0.7796998858451843 +- 0.005762816743594405
# test_cross_entropy : 0.5895079493522644 +- 0.014903778098054713
# test_loss          : 0.5895080089569091 +- 0.014903778098054713
# val_acc            : 0.8034000217914581 +- 0.0074323665476892144
# val_cross_entropy  : 0.5610684156417847 +- 0.01406437182554169
# val_loss           : 0.5610684156417847 +- 0.01406437182554169
python -m graph_tf stale_gcn/config/fit_and_test.gin stale_gcn/config/gcn/ogbn-arxiv.gin

```

The same models can also be trained without subgraph batching.

```bash
python -m graph_tf stale_gcn/config/baseline/build_and_fit_many.gin stale_gcn/config/gcn2/cora.gin
# Results for 10 runs
# test_acc           = 0.8415999948978424 +- 0.0029393778318850794
# test_cross_entropy = 0.5605585694313049 +- 0.011878842790894833
# test_loss          = 0.7766392588615417 +- 0.011830674182216071
# val_acc            = 0.8161999106407165 +- 0.006838133104470735
# val_cross_entropy  = 0.616016548871994 +- 0.010837771795063983
# val_loss           = 0.8320972800254822 +- 0.011010068957633332
```

## TODO

Implementations:

- Implement scalable cache output update

Extensions:

- make cache output values distributions
- can we somehow cache gradient information?
