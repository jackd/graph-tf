import matplotlib.pyplot as plt

from graph_tf.projects.igcn.scalable.jl import johnson_lindenstrauss_min_dim

num_labels = 90941
eig_ranks = [128, 256, 512, 1024]
jl_eps = [0.5, 0.2, 0.1]
jl_ranks = [johnson_lindenstrauss_min_dim(num_labels, eps=eps) for eps in jl_eps]

eig_accs = [
    0.5136308670043945,
    0.5916918516159058,
    0.6121021509170532,
    0.6250025629997253,
]
jl_accs = [
    0.6265662908554077,
    0.6652470231056213,
    0.6779828667640686,
]
jl_sample_accs = [
    0.6704524159431458,
    0.682488739490509,
    0.6874266862869263,
]

ip_acc = 0.7115198731422424
ip_acc_std = 0.001346684259970854

ax0 = plt.gca()

minx = min(min(eig_ranks), min(jl_ranks))
maxx = max(max(eig_ranks), max(jl_ranks))


def scatter_and_plot(ax: plt.Axes, x, y, label):
    (line,) = ax.plot(x, y, label=label, linestyle="dashed")
    color = line._color  # pylint: disable=protected-access
    ax.scatter(x, y, c=color)
    return color


def plot_constant(ax: plt.Axes, y: float, std: float, label: str):
    (line,) = ax.plot([minx, maxx], [y, y], linestyle="dashed", label=label)
    color = line._color  # pylint: disable=protected-access
    ax.fill_between([minx, maxx], y - std, y + std, alpha=0.25, color=color)
    return color


scatter_and_plot(ax0, eig_ranks, eig_accs, label="eigen approximation")
scatter_and_plot(ax0, jl_ranks, jl_accs, label="random projection")
scatter_and_plot(ax0, jl_ranks, jl_sample_accs, label="sampled random projection")

plot_constant(ax0, ip_acc, ip_acc_std, label="input propagated")
plot_constant(ax0, 0.7209, 0.0025, label="DAGNN")
plot_constant(ax0, 0.7195, 0.0011, label="SIGN")
ax0.set_xscale("log")
plt.legend()
ax0.set_title("ogbn-arxiv results")
ax0.set_xlabel("rank")
ax0.set_ylabel("Test accuracy")
plt.show()
