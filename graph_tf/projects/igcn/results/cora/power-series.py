"""
python -m graph_tf gtf_config/build_and_fit.gin igcn/config/cora/lp.gin igcn/config/impl/power-series.gin igcn/config/utils/tb.gin --bindings='num_terms=64'
"""
import os
import typing as tp

import matplotlib.pyplot as plt


class Result(tp.NamedTuple):
    num_terms: int
    test_acc: float
    mean_step_time: float  # ms
    peak_memory_usgae: float  # MiB


results = (
    Result(1, 0.41199991106987, 4.6, 50.43),
    Result(2, 0.5199999809265137, 4.7, 50.82),
    Result(4, 0.8129999041557312, 4.9, 51.64),
    Result(8, 0.8319998979568481, 5.0, 51.13),
    Result(16, 0.8349999189376831, 5.4, 51.12),
    Result(32, 0.8319998979568481, 6.1, 50.81),
    Result(64, 0.8459998369216919, 7.0, 50.82),
    Result(128, 0.8369998931884766, 9.7, 50.46),
    Result(256, 0.8369998931884766, 14.3, 50.63),
    Result(512, 0.8369998931884766, 24.6, 51.13),
)

x = [r.num_terms for r in results]
y = [r.test_acc for r in results]
# fig, (ax0, ax1) = plt.subplots(1, 2)
ax0 = plt.gca()
ax1 = ax0.twinx()
lns0 = ax0.plot(x, y, color="blue", label="accuracy")
ax0.set_xscale("log")
ax0.set_ylabel("test accuracy")
ax0.set_xlabel("number of terms")
ax0.tick_params(axis="y", colors="blue")

y = [r.mean_step_time for r in results]
lns1 = ax1.plot(x, y, color="red", label="step time")
ax1.set_ylabel("train step time (ms)")
ax1.set_xlabel("number of terms")
ax1.set_yscale("log")
ax1.tick_params(axis="y", colors="red")

lns = lns0 + lns1
labs = [l.get_label() for l in lns]
ax0.legend(lns, labs, loc=0)

# plt.show()
fname = os.path.expanduser("~/Pictures/power-series.png")
plt.savefig(fname)
print(f"Figure saved to {fname}")
