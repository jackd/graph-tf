# MLP

Simple baseline implementation.

## Example Usage

```bash
python -m graph_tf gtf_config/build_and_fit.gin \
    config/cora.gin
# Final results
# test_acc           : 0.4809999465942383
# test_cross_entropy : 1.93781316280365
# test_loss          : 2.1298935413360596
# val_acc            : 0.5239999890327454
# val_cross_entropy  : 1.9382566213607788
# val_loss           : 2.1303369998931885
python -m graph_tf gtf_config/build_and_fit.gin \
    config/cora.gin \
    config/spectral.gin
# Final results
# test_acc           : 0.718999981880188
# test_cross_entropy : 1.7003529071807861
# test_loss          : 2.230227470397949
# val_acc            : 0.722000002861023
# val_cross_entropy  : 1.7080714702606201
# val_loss           : 2.237946033477783
python -m graph_tf gtf_config/build_and_fit.gin \
    config/ogbn-arxiv.gin
# Final results
# test_acc           : 0.5477234125137329
# test_cross_entropy : 1.5347206592559814
# test_loss          : 1.534720778465271
# val_acc            : 0.5715292692184448
# val_cross_entropy  : 1.48497474193573
# val_loss           : 1.4849746227264404
python -m graph_tf gtf_config/build_and_fit.gin \
    config/ogbn-arxiv.gin \
    config/spectral.gin
# Final results
# test_acc           : 0.6093450784683228
# test_cross_entropy : 1.296971321105957
# test_loss          : 1.296971321105957
# val_acc            : 0.6263633370399475
# val_cross_entropy  : 1.2771133184432983
# val_loss           : 1.2771131992340088
python -m graph_tf gtf_config/build_and_fit.gin \
   config/ogbn-arxiv.gin \
   config/page-rank.gin
# Final results
# test_acc           : 0.7021788954734802
# test_cross_entropy : 0.9461618661880493
# test_loss          : 0.9461618661880493
# val_acc            : 0.7152924537658691
# val_cross_entropy  : 0.9119006991386414
# val_loss           : 0.9119006395339966
python -m graph_tf gtf_config/build_and_fit_many.gin \
   config/ogbn-arxiv.gin \
   config/page-rank.gin \
# Results for 10 runs
# test_acc           = 0.6942184746265412 +- 0.004496337727582618
# test_cross_entropy = 0.9782700657844543 +- 0.01961790036160597
# test_loss          = 0.9782700598239898 +- 0.01961790574852028
# val_acc            = 0.7052653074264527 +- 0.005533138624581985
# val_cross_entropy  = 0.9455797135829925 +- 0.020290075153031193
# val_loss           = 0.9455796599388122 +- 0.02029006773100859
python -m graph_tf gtf_config/build_and_fit_many.gin \
   config/ogbn-arxiv.gin \
   config/page-rank.gin \
   --bindings='include_transformed_features=False'  # only use page-rank features

python -m graph_tf gtf_config/build_and_fit_many.gin \
    config/ogbn-arxiv.gin \
    config/page-rank.gin \
    igcn/config/losses/quad.gin
# Results for 10 runs
# test_acc                = 0.5515420973300934 +- 0.19170338595876152
# test_cross_entropy      = 2.762668490409851 +- 0.04226119911152964
# test_loss               = -6.893861734867096 +- 3.6577445709950704
# test_quad_cross_entropy = -6.893861734867096 +- 3.6577445709950704
# val_acc                 = 0.5655022293329239 +- 0.18760663171944575
# val_cross_entropy       = 2.753391981124878 +- 0.023721234756538038
# val_loss                = -7.127738726139069 +- 3.7778503030988353
# val_quad_cross_entropy  = -7.127739346027374 +- 3.7778506032121455
python -m graph_tf gtf_config/build_and_fit_many.gin \
    config/ogbn-arxiv.gin \
    config/page-rank.gin igcn/config/losses/quad.gin \
    --bindings="monitor='val_quad_cross_entropy'"
# Results for 10 runs
# test_acc                = 0.6886056661605835 +- 0.004370978787818776
# test_cross_entropy      = 3.0518511295318604 +- 0.04398409811811501
# test_loss               = -9.97844820022583 +- 0.06660351909784977
# test_quad_cross_entropy = -9.97844820022583 +- 0.06660351909784977
# val_acc                 = 0.6994798898696899 +- 0.0031420928092466575
# val_cross_entropy       = 3.015373182296753 +- 0.030108230313669056
# val_loss                = -10.248607158660889 +- 0.06030754986587472
# val_quad_cross_entropy  = -10.248608207702636 +- 0.060307483367854076
python -m graph_tf gtf_config/build_and_fit_many.gin \
    config/ogbn-arxiv.gin \
    config/page-rank.gin igcn/config/losses/quad.gin \
    --bindings="
monitor='val_acc'
mode='max'
"
# Results for 10 runs
# test_acc                = 0.6941258907318115 +- 0.0021165803618628663
# test_cross_entropy      = 2.8227633237838745 +- 0.034013222004606505
# test_loss               = -9.800707626342774 +- 0.08685750726054473
# test_quad_cross_entropy = -9.800707626342774 +- 0.08685750726054473
# val_acc                 = 0.7036142468452453 +- 0.0017190754197330639
# val_cross_entropy       = 2.809351372718811 +- 0.033015391510677554
# val_loss                = -10.074717235565185 +- 0.07860383131779795
# val_quad_cross_entropy  = -10.074718284606934 +- 0.0786036238248599
```
