"""
v0:
python -m graph_tf gtf_config/build_and_fit.gin igcn/config/cora/lp.gin \
    igcn/config/impl/batched-lp.gin --bindings='
always_include_train_ids=False
batch_size=512
'

v1:
python -m graph_tf gtf_config/build_and_fit.gin igcn/config/cora/lp.gin \
    igcn/config/impl/batched-lp.gin --bindings='
always_include_train_ids=True
batch_size=372  # 512 - 140
'

v2:
python -m graph_tf gtf_config/build_and_fit.gin igcn/config/cora/lp.gin \
    igcn/config/impl/batched-v2-lp.gin --bindings='
label_nodes_per_step = 64
label_nodes_per_step = 488  # 512 - 64
'

v3:
python -m graph_tf gtf_config/build_and_fit.gin igcn/config/cora/lp.gin \
    igcn/config/impl/batched-v3-lp.gin igcn/config/losses/quad.gin  --bindings='
temperature=0.01
batch_size=512
'

Timings via:
tensorboard --load_fast=false --logdir=/tmp/graph_tf/igcn
"""
import os
import typing as tp

import matplotlib.pyplot as plt


class Result(tp.NamedTuple):
    batch_size: int
    test_acc: float
    mean_step_time: float  # ms


v0_results = ()
v1_results = ()
v2_results = ()
v3_results = ()

# plt.show()
fname = os.path.expanduser("~/Pictures/batched.png")
plt.savefig(fname)
print(f"Figure saved to {fname}")
