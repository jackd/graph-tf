"""
python -m graph_tf mains/build-and-fit.gin cora.gin --bindings="
batch_size=0.03125
temperature=1e-2
"
"""

import matplotlib.pyplot as plt
import numpy as np

temperatures = [1e-2, 1e-1, 1e0, 1e1, np.inf]
batch_size = [0.125]

# fmt: off
# pylint: disable=line-too-long
accs = np.array([
    [0.7760000228881836],   # temperature=1e-2
    [0.7329999804496765],  # temperature=1e-1
    [],  # temperature=1e0
    [],  # temperature=1e1
    [],  # temperature=%INF
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
    temperatures,
    xlabel="batch_size",
    ylabel="test accuracy",
    logx=True,
    ax=ax0,
)
plot_many(
    temperatures[:-1],
    accs[:-1].T,
    batch_size,
    xlabel="temperature",
    ylabel="test accuracy",
    logx=True,
    ax=ax1,
)

plt.show()
