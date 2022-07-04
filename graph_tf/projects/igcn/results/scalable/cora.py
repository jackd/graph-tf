"""
python -m graph_tf mains/build-and-fit.gin cora.gin --bindings="
batch_size=0.03125
temperature=1e-2
"
"""

import matplotlib.pyplot as plt
import numpy as np

temperatures = [1e-2, 1e-1, 1e0, 1e1, np.inf]
batch_size = [0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0]

# fmt: off
# pylint: disable=line-too-long
accs = np.array([
    [0.2630000114440918, 0.5410000085830688, 0.8050000071525574, 0.8360000252723694, 0.8450000286102295, 0.8420000076293945],   # temperature=1e-2
    [0.39100000262260437, 0.6980000138282776, 0.8349999785423279, 0.8209999799728394, 0.8420000076293945, 0.8330000042915344],  # temperature=1e-1
    [0.20000000298023224, 0.37500000000000000, 0.7490000128746033, 0.8339999914169312, 0.8349999785423279, 0.8360000252723694],  # temperature=1e0
    [0.14900000393390656, 0.14900000393390656, 0.4449999928474426, 0.8379999995231628, 0.8460000157356262, 0.8349999785423279],  # temperature=1e1
    [0.14900000393390656, 0.3199999928474426, 0.33500000834465027, 0.7900000214576721, 0.8399999737739563, 0.8379999995231628],  # temperature=%INF
])
# pylint: enable=line-too-long
# fmt: on


def plot_many(x, ys, labels, ax=None, xlabel=None, ylabel=None, logx=True):
    if ax is None:
        ax = plt.gca()
    assert len(ys) == len(labels)
    for y, label in zip(ys, labels):
        assert len(y) == len(x)
        (line,) = ax.plot(x, y, linestyle="dotted")
        c = line._color  # pylint: disable=protected-access
        ax.scatter(x, y, color=c, label=label)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if logx:
        ax.set_xscale("log")
    ax.legend()


fig, (ax0, ax1) = plt.subplots(1, 2)
plot_many(
    batch_size,
    accs,
    [f"temp = {t}" for t in temperatures],
    xlabel="batch_size",
    ylabel="test accuracy",
    logx=True,
    ax=ax0,
)
plot_many(
    temperatures[:-1],
    accs[:-1].T,
    [f"batch_size = {bs}" for bs in batch_size],
    xlabel="temperature",
    ylabel="test accuracy",
    logx=True,
    ax=ax1,
)

plt.show()
