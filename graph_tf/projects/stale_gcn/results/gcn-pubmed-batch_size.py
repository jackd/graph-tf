import matplotlib.pyplot as plt

batch_sizes = [
    # 1,
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
    # 2048,
    # 4096,
    # 8192,
    # 16384,
    # 19717,
]

# python -m graph_tf stale_gcn/config/fit_and_test.gin stale_gcn/config/gcn/pubmed.gin\
# --bindings='train_batch_size=128'
stale_results = [
    0.7379999756813049,
    0.7409999966621399,
    0.7689999341964722,
    0.7589998841285706,
    0.7929998636245728,
    0.7939999103546143,
    0.7819998860359192,  # 128
    0.7879999279975891,
    0.7819998860359192,
    0.7889999151229858,
]

fresh_mean, fresh_std = 0.7905998885631561, 0.00280000141694776

(line,) = plt.plot(
    [batch_sizes[0], batch_sizes[-1]],
    [fresh_mean] * 2,
    # [base_std] * 2,
    label="fresh",
    linestyle="dashed",
)
plt.fill_between(
    [batch_sizes[0], batch_sizes[-1]],
    [fresh_mean - fresh_std] * 2,
    [fresh_mean + fresh_std] * 2,
    color=line._color,  # pylint: disable=protected-access
    alpha=0.1
    # [base_std] * 2,
)

plt.plot(batch_sizes, stale_results, label="stale")

plt.legend()
plt.xlabel("batch size")
plt.ylabel("test accuracy")
plt.title("GCN Pub Med")
ax = plt.gca()
ax.set_xscale("log")
ax.set_xticks(batch_sizes, [str(h) for h in batch_sizes])

plt.show()
