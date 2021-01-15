# DFNets

Implementation of _DFNets: Spectral CNNs for Graphs with
Feedback-Looped Filters_

```bibtex
@inproceedings{asiri2019dfnets,
  title={DFNets: Spectral CNNs for Graphs with Feedback-Looped Filters},
  author={Wijesinghe, Asiri and Wang, Qing},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2019}
}
```

- [Paper](https://arxiv.org/pdf/1910.10866.pdf)
- [Original Repository](https://github.com/wokas36/DFNets.git)

## Setup

```bash
pip install cvxpy
```

## Discreprencies

This implementation has the following discreprencies with the original:

- No class weights. The original repo uses 8 class weights (0 being 'non-training') but only infers 7 classes. This results in under-weighting of the original class 0 nodes.
- Different validation/test IDs. This makes a significant difference to results (81.4% vs 85.0%).
- `BatchNormalization` momentum: we use `0.8`, the original uses default (`0.99`).
- No activity regularizer on the final outputs (original uses `l2=1e-10`)
