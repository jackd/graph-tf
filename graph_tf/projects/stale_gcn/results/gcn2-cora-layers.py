import matplotlib.pyplot as plt

hidden_layers = [
    1,
    2,
    4,
    8,
    16,
    32,
    64,
]

# python -m graph_tf stale_gcn/config/baseline/build_and_fit_many.gin stale_gcn/config/gcn2/cora.gin --bindings='num_hidden_layers=NUM_HIDDEN_LAYERS'
baseline_results = [
    [0.7888000190258027, 0.0054918091072367935],
    [0.8123000204563141, 0.009089033307176137],
    [0.8153000354766846, 0.008764138920495108],
    [0.8146999955177308, 0.006372576992942926],
    [0.8368000149726867, 0.004166552647199557],
    [0.8415000081062317, 0.0033837897390308603],
    [0.8419000029563903, 0.0030149513499308335],  # 64
]

# python -m graph_tf stale_gcn/config/fit_and_test_many.gin stale_gcn/config/gcn2/cora.gin --bindings='num_hidden_layers=1'
stale_results = [
    [0.6576000094413758, 0.014361044893891128],
    [0.4673000156879425, 0.04992003386769254],
    [0.46690000891685485, 0.06620491688345215],
    [0.5479999929666519, 0.07074038165395269],
    [0.5724999845027924, 0.05731883357047854],
    # [],
    # [],  # 64
]

# python -m graph_tf stale_gcn/config/fit_and_test_many.gin stale_gcn/config/gcn2/cora.gin --bindings='
# lr=1e-3
# epochs=3000
# num_hidden_layers=1
# '
slow_stale_results = [
    [0.7235999882221222, 0.012467542912518972],
    [0.6799000024795532, 0.023338585289869766],
    [0.6452000081539154, 0.021079859558644654],
    [0.6838000059127808, 0.014112398884842229],
    [0.7246000230312347, 0.008284918400009347],
    [0.7411999940872193, 0.008599999477734524],
    [0.753600001335144, 0.009068640753438044],  # 64
]

for results, label in (
    (baseline_results, "baseline"),
    (stale_results, "stale, lr=1e-2"),
    (slow_stale_results, "stale, lr=1e-3"),
):
    print(results)
    mean, std = zip(*results)
    x = hidden_layers[: len(mean)]
    plt.errorbar(x, mean, std, label=label)
plt.legend()

ax = plt.gca()
ax.set_xscale("log")
ax.set_xticks(hidden_layers, [str(h) for h in hidden_layers])
plt.xlabel("hidden layers")
plt.ylabel("test accuracy")
plt.title("GCN2 Cora")
plt.show()
