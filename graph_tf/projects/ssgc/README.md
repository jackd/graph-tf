# Simple Spectral Graph Convolutions

- [Paper](https://openreview.net/pdf?id=CYO5T-YjWZV)
- [Original Repo](https://github.com/allenhaozhu/SSGC)

```bib
@inproceedings{zhu2020simple,
  title={Simple spectral graph convolution},
  author={Zhu, Hao and Koniusz, Piotr},
  booktitle={International Conference on Learning Representations},
  year={2020}
}
```

## Example Usage

```bash
python -m graph_tf gtf_config/build_and_fit_many.gin \
    ssgc/config/cora.gin
# Results for 10 runs
# test_acc           = 0.825599879026413 +- 0.0008000060918930662
# test_cross_entropy = 0.8213066816329956 +- 7.693677041106874e-05
# test_loss          = 1.0844363451004029 +- 7.932109806598289e-05
# val_acc            = 0.8061999320983887 +- 0.0005999922752380372
# val_cross_entropy  = 0.8533543884754181 +- 0.00015086776168993395
# val_loss           = 1.1164840579032898 +- 0.00015561454316054895
python -m graph_tf gtf_config/build_and_fit_many.gin \
    ssgc/config/citeseer.gin
# Results for 10 runs
# test_acc           = 0.731999933719635 +- 0.0
# test_cross_entropy = 1.4120635390281677 +- 2.163661741316672e-05
# test_loss          = 1.6688621640205383 +- 6.019360906659011e-06
# val_acc            = 0.7479999661445618 +- 0.0
# val_cross_entropy  = 1.4268400192260742 +- 2.2835297878128672e-05
# val_loss           = 1.6836386322975159 +- 7.163129384385299e-06
python -m graph_tf gtf_config/build_and_fit_many.gin \
    ssgc/config/pubmed.gin
# Results for 10 runs
# test_acc           = 0.798899906873703 +- 0.0002999961376190186
# test_cross_entropy = 0.6041425228118896 +- 7.010027848187513e-05
# test_loss          = 0.7609697580337524 +- 3.547014267784175e-05
# val_acc            = 0.8097999453544616 +- 0.0005999922752380372
# val_cross_entropy  = 0.5885282754898071 +- 3.547411883405347e-05
# val_loss           = 0.7453554987907409 +- 6.720446192202595e-05
```
