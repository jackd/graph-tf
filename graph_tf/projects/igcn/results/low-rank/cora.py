import matplotlib.pyplot as plt

k = [8, 16, 32, 64, 128, 256, 512]

# fmt:off
basic_results = [
    0.11099997907876968, 0.15999998152256012, 0.5859999656677246, 0.7109999060630798, 0.7629998922348022, 0.809999942779541, 0.8219999074935913,
]
quad_results = [
    0.6909999847412109, 0.7310000061988831, 0.8069999814033508, 0.8180000185966492, 0.8320000171661377, 0.8349999785423279, 0.8309999704360962,
]

assert len(k) == len(basic_results)
assert len(k) == len(quad_results)


def plot_and_scatter(x, y, label, ax: plt.Axes):
    line, = ax.plot(x, y, linestyle="dashed", label=label)
    ax.scatter(x, y, color=line._color) # pylint: disable=protected-access


ax = plt.gca()
plot_and_scatter(k, basic_results, "low-rank", ax)
plot_and_scatter(k, quad_results, "quad low-rank", ax)
ax.set_xscale("log")
ax.set_xlabel("rank")
ax.set_ylabel("accuracy")
ax.set_title("Cora")
plt.legend()
plt.show()
