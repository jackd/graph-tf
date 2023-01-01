import matplotlib.pyplot as plt
import numpy as np


def scatter(x, y, label: str, marker=None, va=None):
    x = np.asarray(x) * 1000
    y = np.asarray(y) * 100
    assert x.shape == y.shape, (x.shape, y.shape)
    cycle = plt.gca()._get_lines.prop_cycler  # pylint: disable=protected-access
    color = next(cycle)["color"]
    if x.size > 1:
        plt.plot(x, y, linestyle="dashed", color=color)
    else:
        kwargs = {}
        oy = 0
        if va:
            kwargs["va"] = va
            if va == "top":
                oy = -0.2
            elif va == "bottom":
                oy = 0.15
        plt.annotate(label, (x, y + oy), ha="center", **kwargs)
        label = None
        x = x.reshape((1,))
        y = y.reshape((1,))
    plt.scatter(x, y, marker=marker, color=color, label=label)


# relaxed tolerance
# fmt: off
tols = [1, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
times = [0.023210828304290772, 0.025132465362548827, 0.028145329952239992, 0.02915379047393799, 0.031570730209350584, 0.03273360252380371, 0.035237772464752196, 0.03685513734817505, 0.038984262943267824, 0.041272776126861574]
accs = [0.7646998941898346, 0.789099907875061, 0.8038998901844024, 0.8075998902320862, 0.8049998760223389, 0.8030998885631562, 0.8024998903274536, 0.8026999056339263, 0.8012998998165131, 0.8037999093532562]
# tols = [5e-1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
# times = [0.024346468448638917, 0.027722008228302, 0.03287738084793091, 0.03626770496368408, 0.040560505390167235, 0.04504312515258789]
# accs = [0.784599894285202, 0.8046999037265777, 0.803299880027771, 0.8030999064445495, 0.8038999021053315, 0.8037998914718628]
# fmt: on
scatter(times, accs, "MLP-PPR CG")

# low rank
# fmt: off
ranks = [32, 64, 128, 256, 512]
times = [0.021380081176757812, 0.022319774627685546, 0.02263925313949585, 0.025996685028076172, 0.031128764152526855]
accs = [0.7284999489784241, 0.7689998865127563, 0.7591998875141144, 0.787799882888794, 0.7965999186038971]
# fmt: on
scatter(times, accs, "MLP-PPR Low Rank")


scatter(0.01993891954421997, 0.8039999008178711, "MLP-PPR Precomp.", va="top")

scatter(0.004932131767272949, 0.7949998855590821, "PPR-MLP", va="top")

# scatter(0.032966253757476804, 0.8009999036788941, "DAGNN")

# scatter(0.02318448781967163, 0.8054999113082886 , "SS-DAGNN")

# scatter( 0.42008963108062747, 0.7968999087810517 ,"GCN2")

# scatter(0.27263712406158447, 0.7994998753070831, "SS-GCN2")

scatter(0.032966253757476804, 0.8062, "DAGNN", marker="x", va="bottom")

scatter(0.02318448781967163, 0.8059, "SS-DAGNN", marker="x", va="bottom")

# scatter(0.42008963108062747, 0.8032, "GCN2", marker="x")

# scatter(0.27263712406158447, 0.8003, "SS-GCN2", marker="x")


# plt.xscale("log")
left, right = plt.xlim()
plt.xlim((left - 2, right))
plt.xlabel("Train step time (ms)")
plt.legend(loc="lower right")
plt.ylabel("Converged test accuracy (%)")
plt.show()
