import matplotlib
import matplotlib.pyplot as plt
import numpy as np

batch_sizes = [
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
    2708,
]

acc_mean_std = [
    [0.5718015313148499, 0.025513825334746307],
    [0.6295014441013336, 0.04497811611321428],
    [0.673399168252945, 0.032252886558524166],
    [0.6881997942924499, 0.019135311322808393],
    [0.7156005442142487, 0.02141589284994981],
    [0.7347999632358551, 0.012014970487038973],
    [0.7482999622821808, 0.011532978930742545],
    [0.7638998925685883, 0.008982777796801279],
    [0.7786999046802521, 0.008234685161112273],
    [0.7984000146389008, 0.009789797410719963],
    [0.8049000263214111, 0.0052239801351685],
    [0.8134999930858612, 0.004609781409113673],
    [0.8185000061988831, 0.007736277577184404],
]
mean, std = zip(*acc_mean_std)
base_mean, base_std = 0.8205000162124634, 0.0026551909546995017

batch_sizes = np.array(batch_sizes)
mean = np.array(mean)
std = np.array(std)

plt.errorbar(batch_sizes, mean, std, label="stale")
(line,) = plt.plot(
    [batch_sizes[0], batch_sizes[-1]],
    [base_mean] * 2,
    # [base_std] * 2,
    label="fresh",
    linestyle="dashed",
)
plt.fill_between(
    [batch_sizes[0], batch_sizes[-1]],
    [base_mean - base_std] * 2,
    [base_mean + base_std] * 2,
    color=line._color,  # pylint: disable=protected-access
    alpha=0.1
    # [base_std] * 2,
)

ax = plt.gca()
ax.set_xscale("log")
# ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.xaxis.set_major_formatter(matplotlib.ticker.NullFormatter())
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
# ax.set_xticks([])
ax.set_xticks(batch_sizes[:-1], [str(bs) for bs in batch_sizes[:-1]])
plt.legend()
plt.xlabel("batch size")
plt.ylabel("accuracy")
plt.title("GCN Cora results")
plt.show()
