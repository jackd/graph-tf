"""
python -m graph_tf gtf_config/build_and_fit.gin igcn/config/cora/lp.gin igcn/config/impl/low-rank.gin igcn/config/utils/tb.gin --bindings='rank=64'
"""
import os
import typing as tp

import matplotlib.pyplot as plt


class Result(tp.NamedTuple):
    rank: int
    test_acc: float
    mean_step_time: float  # ms


results = (
    Result(16, 0.15999998152256012, 4.8),
    Result(32, 0.5859999656677246, 4.8),
    Result(64, 0.7109999060630798, 4.9),
    Result(128, 0.7629998922348022, 4.9),
    Result(256, 0.809999942779541, 5.3),
    Result(512, 0.8219999074935913, 5.6),
    Result(1024, 0.8389999270439148, 7.0),
)

x = [r.rank for r in results]
y = [r.test_acc for r in results]
# fig, (ax0, ax1) = plt.subplots(1, 2)
ax0 = plt.gca()
ax1 = ax0.twinx()
lns0 = ax0.plot(x, y, color="blue", label="accuracy")
ax0.set_xscale("log")
ax0.set_ylabel("test accuracy")
ax0.set_xlabel("rank")
ax0.tick_params(axis="y", colors="blue")

y = [r.mean_step_time for r in results]
lns1 = ax1.plot(x, y, color="red", label="step time")
ax1.set_ylabel("train step time (ms)")
ax1.set_xlabel("rank")
ax1.set_yscale("log")
ax1.tick_params(axis="y", colors="red")

lns = lns0 + lns1
labs = [l.get_label() for l in lns]
ax0.legend(lns, labs, loc=0)

# plt.show()
fname = os.path.expanduser("~/Pictures/low-rank.png")
plt.savefig(fname)
print(f"Figure saved to {fname}")
