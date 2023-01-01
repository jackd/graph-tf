# Literature Notes

## General Notes

PageRank:

$$P_\alpha = \alpha\left[I - (1 - \alpha)A\right]^{-1}$$
$$P_\alpha = \alpha \sum_{k=0}^\infty(1 - \alpha)^k A^k$$
$$f(\lambda) = \frac{\alpha}{\alpha + (1 - \alpha)\lambda}$$

Heat Diffusion:

$$H_t = e^{-tL}$$
$$H_t = \sum_{k=0}^\infty\frac{t^k}{k!}(A - I)^k=e^{-t}\sum_{k=0}^\infty \frac{t^k}{k!}A^k$$
$$f(\lambda) = e^{-t\lambda}$$

## (A)PPNP

- [Paper](https://arxiv.org/pdf/1810.05997.pdf)
- [Original Repository](https://github.com/gasteigerjo/ppnp)
  - [Different repository](https://github.com/benedekrozemberczki/APPNP)
- Approximate implementation uses an iterative method to approximate the page rank vector
- Slow, memory intensive
- Does not take advantage of semi-supervised nature

$$ Z^{(k+1)} = (1 - \alpha)Z^{(k)} + \alpha H, \quad Z^{(0)} = H $$

## PPRGo

```bibtex
@inproceedings{bojchevski2020scaling,
  title={Scaling graph neural networks with approximate pagerank},
  author={Bojchevski, Aleksandar and Klicpera, Johannes and Perozzi, Bryan and Kapoor, Amol and Blais, Martin and R{\'o}zemberczki, Benedek and Lukasik, Michal and G{\"u}nnemann, Stephan},
  booktitle={Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  pages={2464--2473},
  year={2020}
}
```

- [Homepage](https://www.daml.in.tum.de/pprgo)
- [Paper](https://arxiv.org/pdf/2007.01570.pdf)
- Repositories:
  - [TF1](https://github.com/TUM-DAML/pprgo_tensorflow)
  - [pytorch](https://github.com/TUM-DAML/pprgo_pytorch)
- Approximates PageRank matrix with top-k row entries
- Nonstandard datasets

## JKNet: Jumping Knowledge Networks

```bibtex
@inproceedings{xu2018representation,
  title={Representation learning on graphs with jumping knowledge networks},
  author={Xu, Keyulu and Li, Chengtao and Tian, Yonglong and Sonobe, Tomohiro and Kawarabayashi, Ken-ichi and Jegelka, Stefanie},
  booktitle={International conference on machine learning},
  pages={5453--5462},
  year={2018},
  organization={PMLR}
}
```

- [Paper](http://proceedings.mlr.press/v80/xu18c/xu18c.pdf)
- [Original Repository](https://github.com/ShinKyuY/Representation_Learning_on_Graphs_with_Jumping_Knowledge_Networks)
  - [DGL implementation](https://github.com/mori97/JKNet-dgl)
- Generalizes multi-hop neighborhood aggregations
$$ Z = \text{Aggregate}([H^{(0)}, H^{(1)}, H^{(2)}, \cdots H^{(K)}])$$
$$ H^{(k+1)} = \text{GraphConv}(H^{(k)}, A; \theta^{(k)})$$
$$ H^{(0)} = X $$
- `Aggregate` is one of
  - `concat`
  - `max_pool`
  - `LSTM`

## GDC: Graph Diffusion Convolution

```bibtex
@article{klicpera2019diffusion,
  title={Diffusion improves graph learning},
  author={Klicpera, Johannes and Wei{\ss}enberger, Stefan and G{\"u}nnemann, Stephan},
  journal={arXiv preprint arXiv:1911.05485},
  year={2019}
}
```

- [Paper](https://arxiv.org/pdf/1911.05485.pdf)
- [Reference Implementation](https://github.com/gasteigerjo/gdc)
- Uses precomputed sparsified Heat / PPR matrix instead of normalized adjacency matrix in GCN framework.

## GPR-GNN: Generalized Page Rank GNN

```bibtex
@inproceedings{chien2021adaptive,
  title={Adaptive Universal Generalized PageRank Graph Neural Network},
  author={Eli Chien and Jianhao Peng and Pan Li and Olgica Milenkovic},
  booktitle={International Conference on Learning Representations},
  year={2021},
  url={https://openreview.net/forum?id=n6jl7fLxrP}
}
```

- [Paper](https://arxiv.org/pdf/2006.07988.pdf)
- [Repository](https://github.com/jianhao2016/GPRGNN)
$$ Z = \sum_{k=0}^{K} \gamma_k \tilde{A}^k \tilde{X}, \quad \tilde{X} = \text{MLP}(X; \theta) $$

## PPR-GNN: Infinite Depth GNN

```bibtex
@article{roth2022transforming,
  title={Transforming PageRank into an Infinite-Depth Graph Neural Network},
  author={Roth, Andreas and Liebig, Thomas},
  journal={arXiv preprint arXiv:2207.00684},
  year={2022}
}
```

- [Paper](https://arxiv.org/pdf/2207.00684.pdf)
- [Repository](https://github.com/roth-andreas/pprgnn)
- TODO

## (G)ADC: Adaptive Diffusion in Graph Neural Networks

```bibtex
@article{zhao2021adaptive,
  title={Adaptive Diffusion in Graph Neural Networks},
  author={Zhao, Jialin and Dong, Yuxiao and Ding, Ming and Kharlamov, Evgeny and Tang, Jie},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  pages={23321--23333},
  year={2021}
}
```

- [Paper](https://proceedings.neurips.cc/paper/2021/file/c42af2fa7356818e0389593714f59b52-Paper.pdf)
- [Repository](https://github.com/abcbdf/ADC)
$$Z^{(l+1)}_i = \sum_{k=0}^{\infty}\theta_{ik}^{(l+1)} T^k X^{(l)}_i$$
- 2 operations
  - ADC: adaptive
  - GADC: generalized + adaptive, i.e. $\theta_{ik}^{(l)}$ is learned
$$\text{ADC:} \quad \theta_{ik}^{(l)} = e^{-t_i^{(l)}}\frac{\left({t_i}^{(l)}\right)^k}{k!}$$
- GADC on input layer is just a factorized linear layer operating on $\text{concat}([X, TX, T^2X, T^3X, \cdots, T^{K-1}X])$
- Experiment configs have:
  - `use_lcc=True`: uses largest connected component
  - random train/validation/test splits
  - more validation examples than standard

## SSGC: Simple Spectral Graph Convolution

```bibtex
@inproceedings{zhu2020simple,
  title={Simple spectral graph convolution},
  author={Zhu, Hao and Koniusz, Piotr},
  booktitle={International Conference on Learning Representations},
  year={2020}
}
```

- [Paper](https://openreview.net/pdf?id=CYO5T-YjWZV)
- [Repository](https://github.com/allenhaozhu/SSGC)
- Linear model operating on transformed features
$$X^\prime = \alpha X + (1-\alpha)\frac{1}{K}\sum_{k=1}^K T^kX $$
- Looks similar to PPR, but note $1 - \alpha$ factor is not raised to the index power
- Cannot reproduce exact values from paper
  - consistently slightly worse
    - cora: 82.6 vs 83.5
    - citeseer: 73.0 vs 73.6
    - pubmed: 79.9 vs 80.2
  - Github [issue](https://github.com/allenhaozhu/SSGC/issues/15) and [PR](https://github.com/allenhaozhu/SSGC/pull/14)

## CGNN: Continuous Graph Neural Networks

- [Paper](http://proceedings.mlr.press/v119/xhonneux20a/xhonneux20a.pdf)
- [Repository](https://github.com/DeepGraphLearning/ContinuousGNN)
- Solves continuous heat diffusion equation using ODE solver
- Hyperparameters for citeseer/pubmed not given
  - [github issue](https://github.com/DeepGraphLearning/ContinuousGNN/issues/3) opened

## DGC: Decoupled Graph Convolution

```bibtex
@InProceedings{wang2021dgc,
  title={Dissecting the Diffusion Process in Linear Graph Convolutional Networks},
  author={Wang, Yifei and Wang, Yisen and Yang, Jiansheng and Lin, Zhouchen},
  booktitle={Proceedings of the 35th Conference on Neural Information Processing Systems (NeurIPS 2021)},
  year={2021}
}
```

- [Paper](https://arxiv.org/pdf/2102.10739.pdf)
- [Repository](https://github.com/yifeiwang77/DGC)
- Linear model with heat-diffused inputs
$$X^{(k+1)} = \left[\left(1 - \frac{T}{K}\right)I + \frac{T}{K}A\right]X^{(k)}$$
- Cannot reproduce results of hyperparameter search - [github issue](https://github.com/yifeiwang77/DGC/issues/1) raised

## DGMLP: Deep Graph MLP

```bibtex
@article{zhang2021evaluating,
  title={Evaluating deep graph neural networks},
  author={Zhang, Wentao and Sheng, Zeang and Jiang, Yuezihan and Xia, Yikuan and Gao, Jun and Yang, Zhi and Cui, Bin},
  journal={arXiv preprint arXiv:2108.00955},
  year={2021}
}
```

- [Paper](https://arxiv.org/pdf/2108.00955.pdf)
- [Repository](https://github.com/PKU-DAIR/DGMLP)
- Fancy initial propagation
  - TODO: understand better
- Linear model
- Cannot reproduce citation results
  - consistently comparable/worse than reported
  - cora: 84.4 vs 84.6 (comparable)
  - citeseer: 72.3 vs 73.4 (worse)
  - pubmed: 80.1 vs 81.2 (worse)
- author contacted

## AP-GCN: Adaptive Propagation GCN

```bibtex
@article{spinelli2020adaptive,
  title={Adaptive propagation graph convolutional network},
  author={Spinelli, Indro and Scardapane, Simone and Uncini, Aurelio},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  volume={32},
  number={10},
  pages={4755--4760},
  year={2020},
  publisher={IEEE}
}
```

- [Paper](https://arxiv.org/pdf/2002.10306.pdf)
- [Repository](https://github.com/spindro/AP-GCN)
- Node adaptive GCN
- Uses larger validation set than standard
  - 72.34% test accuracy for citeseer with 500 validation examples

## PinvGCN: PseudoInverse GCN

```bibtex
@article{alfke2021pseudoinverse,
  title={Pseudoinverse graph convolutional networks},
  author={Alfke, Dominik and Stoll, Martin},
  journal={Data Mining and Knowledge Discovery},
  pages={1--24},
  year={2021},
  publisher={Springer}
}
```

- [Paper](https://link.springer.com/article/10.1007/s10618-021-00752-w)
- [Repository](https://github.com/dominikbuenger/PinvGCN)
- Uses Laplacian pseudoinverse rather than PPR
  - Transforms projections onto nullspace separately
- Attributes poor performance on citation datasets to small eigengap
  - Empirically we show performance can be improved by using shallow architecture similar to DAGNN, GCNII etc.

## DAGNN: Deep Adaptive GNN

- [Paper](https://arxiv.org/pdf/2007.09296.pdf)
- [Repository](https://github.com/mengliu1998/DeeperGNN)
- Transform then propagate model
- Propagation adaptive
  - Empirical evidence suggests adaptive propagation factor not necessary
  - Without adaptive propagation factor, propagation is linear (i.e. shallow)

## GCNII

- [Paper](https://arxiv.org/pdf/2007.02133.pdf)
- [Repository](https://github.com/chennnM/GCNII)
- Adapts GCN with residual/initial skip connections
- The whole thing is almost linear, since:
  - High regularization applied for convolution kernel matrices
  - The kernel matrices are the only sources of negative values, hence ReLU almost always does nothing
- Model is essentially an MLP with PPR
  - Success is based on hyperparameters/regularization

## SGC

```bibtex
@inproceedings{wu2019simplifying,
  title={Simplifying graph convolutional networks},
  author={Wu, Felix and Souza, Amauri and Zhang, Tianyi and Fifty, Christopher and Yu, Tao and Weinberger, Kilian},
  booktitle={International Conference on Machine Learning},
  pages={6861--6871},
  year={2019},
  organization={PMLR}
}
```

- [Paper](https://arxiv.org/pdf/1902.07153.pdf)
- [Repository](https://github.com/Tiiiger/SGC)
- Linear classifier using $\hat{T}^K X$ inputs

## SIGN

```bibtex
@article{rossi2020sign,
  title={Sign: Scalable inception graph neural networks},
  author={Rossi, Emanuele and Frasca, Fabrizio and Chamberlain, Ben and Eynard, Davide and Bronstein, Michael and Monti, Federico},
  journal={arXiv preprint arXiv:2004.11198},
  year={2020}
}
```

- [Paper](https://arxiv.org/abs/2004.11198)
- [Repository](https://github.com/twitter-research/sign)
- MLP with inputs $\text{concat}([X, TX, T^2X, \cdots])$ with various transition matrices $T$ (row-normalized, column-normalized, symmetric-normalized).

## Factorized Neural Networks

- [Paper](https://www.microsoft.com/en-us/research/uploads/prod/2021/03/main_Initialization-Regularization-Factorized-Neural-Layers_final.pdf)
- Not really relevant, but leaving here for future reference

## Results

| Model    | Source   | Cora             | Citeseer         | Pubmed           |
| -------- | -------- | ---------------- | ---------------- | ---------------- |
| SGC      | Paper    | $80.1 \pm 0.0$   | $71.9 \pm 0.1$   | $78.9 \pm 0.0$   |
|          | Repo     | $80.89 \pm 0.08$ | $72.20 \pm 0.00$ | $78.80 \pm 0.00$ |
| SSGC     | Paper    | $83.5 \pm 0.02$  | $73.6 \pm 0.09$  | $80.2 \pm 0.02$  |
|          | **Repo** | $82.57 \pm 0.05$ | $73.00 \pm 0.00$ | $79.93 \pm 0.08$ |
|          | Repo opt | $81.12 \pm 0.12$ | $72.40 \pm 0.00$ | $78.71 \pm 0.09$ |
| DGC      | Paper    | $83.3 \pm 0.0$   | $73.3 \pm 0.1$   | $80.3 \pm 0.1$   |
|          | Repo     | $83.28 \pm 0.04$ | $73.28 \pm 0.08$ | $80.29 \pm 0.10$ |
|          | Repo opt | $80.94 \pm 0.11$ | $72.90 \pm 0.00$ | $78.52 \pm 0.22$ |
| DGMLP    | Paper    | $84.6 \pm 0.6$   | $73.4 \pm 0.5$   | $81.2 \pm 0.6$   |
|          | **Repo** | $84.41 \pm 0.91$ | $72.29 \pm 0.54$ | $80.07 \pm 0.62$ |
| AP-GCN*  | Paper    | -                | $76.12 \pm 0.22$ | $79.80 \pm 0.34$ |
| **       | Repo     | -                | $72.34 \pm 0.30$ | -                |
|          | Repo-mod | $83.55 \pm 0.79$ | $70.40 \pm 0.61$ | $78.57 \pm 0.51$ |
|
|
| ADC-GCN* | Paper    | $\sim 84.5$      | $\sim 74.3$      | $\sim 82.2 $
| *        | Repo-mod | $82.52 \pm 0.40$ | $70.51 \pm 0.80$ | $81.86 \pm 0.63$ |
|          | Repo-mod | $82.88 \pm 0.47$ | $69.99 \pm 0.78$ | $77.92 \pm 0.25$ |
| GADC*
| CGNN     | Paper    | $84.2 \pm 1.0$   | $72.6 \pm 0.6$   | $82.5 \pm 0.4$   |
|          | Repo     | $82.69 \pm 0.43$ | $71.46 \pm 1.14$ | $80.12 \pm 0.34$ |
| GDC      | Paper    | $<84$            | $\sim 73$        | $<80$            |
| GDC-None | Repo     | $81.12 \pm 0.63$ | $69.18 \pm 0.77$ | $78.40 \pm 1.36$ |
| GDC-Heat | Repo     | $81.39 \pm 0.66$ | $70.37 \pm 0.60$ | OOM              |
| GDC-PPR  | Repo     | $81.61 \pm 0.50$ | $70.29 \pm 0.65$ | OOM              |
|
| PinvGCN  | Paper    | $71.41$          | $61.62$          | $71.58$          |
|          | Repo     | $73.49 \pm 1.03$ | $55.75 \pm 1.12$ | $72.83 \pm 0.97$ |
| APPNP*   | Paper    | -                | $75.73 \pm 0.30$ | $79.73 \pm 0.31$ |
|          | APPNP rep| $82.84 \pm 0.50$ | $69.41 \pm 0.63$ | $80.34 \pm 0.08$ |
| PPRGo*   | Paper    | -                | -                | $75.2$           |
| GPR-GNN* | Paper    | $79.51 \pm 0.36$ | $67.63 \pm 0.38$ | $85.07 \pm 0.09$ |
|          | Repo     | $83.92 \pm 0.26$ | $71.62 \pm 0.71$ | $75.89 \pm 1.54$ |
|
|
| DAGNN    | Paper    | $84.4 \pm 0.5$   | $73.3 \pm 0.6$   | $80.5 \pm 0.5$   |
|          | Repo     | $84.15 \pm 0.56$ | $73.18 \pm 0.50$ | $80.62 \pm 0.49$ |
| SS-DAGNN | Repo-mod | $84.32 \pm 0.64$ | $73.08 \pm 0.51$ | $80.59 \pm 0.47$ |
| GCNII    | Paper    | $85.5 \pm 0.5$   | $73.4 \pm 0.2$   | $80.3 \pm 0.4$   |
|          | Repo     | $85.23 \pm 0.57$ | $73.14 \pm 0.40$ | $80.32 \pm 0.51$ |
| SS-GCNII | Repo-mod | $85.15 \pm 0.43$ | $72.61 \pm 1.17$ | $80.03 \pm 0.33$ |

*: non-standard train/validation/test splits used

**: non-standard train/validation/test splits, but same number of train/validation examples

Repo-opt: Original repository with hyperparameters from a custom hyperparameter search

General notes:

- PPR-GNN and SIGN don't give results on citation datasets
- SSGC and DGMLP repos gives inferior results to those reported in their papers
- ADC Pubmed results ~~look legit~~ TRAIN ON THE VALIDATION DATA
  - $82.17 \pm 0.64$ default
  - $81.86 \pm 0.43$ with denseT on second layer
  - $79.04 \pm 0.40$ with denseT on first layer
  - $81.63 \pm 0.33$ with $t \rightarrow \text{softplus}(t)$
  - $80.36 \pm 1.01$ with only 1st propagation layer

### SSGC vs Bugged SSGC

| Model            | Cora             | Citeseer         | Pubmed           |
| -----            | ----             | --------         | ------           |
| Reported (paper) | $83.50 \pm 0.02$ | $73.60 \pm 0.09$ | $80.20 \pm 0.02$ |
| Repo README      | $83.0$           | $73.6$           | $80.6$           |
| Master           | $82.57 \pm 0.05$ | $73.00 \pm 0.00$ | $79.93 \pm 0.08$ |
| Bugged           | $83.04 \pm 0.05$ | $73.40 \pm 0.00$ | $80.55 \pm 0.05$ |
