import typing as tp

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as la
import tensorflow as tf

from graph_tf.data.transforms import normalized_laplacian
from graph_tf.utils.graph_utils import get_component_labels
from graph_tf.utils.scipy_utils import to_scipy, to_tf

n0 = 50
n_random = n0 * 3
seed = 0
rng = np.random.default_rng(seed)


def get_random_adj(n: int, n_random: int, rng: np.random.Generator):

    adj = np.zeros((n, n), dtype=np.float32)
    # ring edges
    adj[np.arange(n - 1), np.arange(1, n)] = 1
    adj[np.arange(1, n), np.arange(n - 1)] = 1
    # random edges
    src = rng.choice(n, size=n_random)
    offset = rng.choice(n - 1, size=n_random)
    dst = (src + offset) % n
    adj[src, dst] = 1
    adj[dst, src] = 1

    adj = sp.coo_matrix(adj)
    return adj


def block_diagonalize(*coos: tp.Sequence[sp.coo_matrix]):
    row_offset = 0
    col_offset = 0
    rows = []
    cols = []
    datas = []
    for coo in coos:
        rows.append(coo.row + row_offset)
        cols.append(coo.col + col_offset)
        datas.append(coo.data)
        row_offset += coo.shape[0]
        col_offset += coo.shape[1]
    data = np.concatenate(datas)
    row = np.concatenate(rows)
    col = np.concatenate(cols)
    return sp.coo_matrix((data, (row, col)), shape=(row_offset, col_offset))


adj = block_diagonalize(
    get_random_adj(n0, n_random, rng),
    # get_random_adj(n0, n_random, rng),
)
n = adj.shape[0]
num_components, labels = get_component_labels(to_tf(adj))

d = np.squeeze(np.array(adj.sum(1)), axis=1)
L = normalized_laplacian(to_tf(adj), symmetric=True)
u0 = tf.scatter_nd(
    tf.stack((tf.range(n, dtype=labels.dtype), labels), axis=1),
    tf.sqrt(d),
    shape=(n, num_components),
)
u0 = u0 / tf.linalg.norm(u0, axis=0, keepdims=True)
# print(u0.shape)
# print(tf.linalg.norm(u0, axis=0))
# print(tf.sparse.sparse_dense_matmul(L, u0))

L = to_scipy(L)
u0 = u0.numpy()
b = rng.normal(size=n)

sol = np.linalg.solve(L.todense() @ (np.eye(L.shape[0]) - u0 @ u0.T), b)

# sol, istop, *_ = la.lsqr(L, b - u0 @ u0.T @ b, iter_lim=1000)
# (sol, istop, itn, r1norm, r2norm, anorm, acond, arnorm, xnorm, var) = la.lsqr(
#     L, b, iter_lim=1000
# )
# np.testing.assert_allclose(L @ sol, b - u0 @ u0.T)
np.testing.assert_allclose(L.todense() @ (np.eye(L.shape[0]) - u0 @ u0.T) @ sol, b)
exit()
# print(itn, r2norm)
# assert istop == 1, f"No approximate solution found, istop={istop}"
# # sol, istop, *_ = la.lsmr(L, b, maxiter=1000, show=True)
# # assert istop in (1, 2), f"No approximate solution found, istop={istop}"
# # print(sol)
# # print(b)
# print(L @ sol - b)
# # print(L @ sol - (b - u0 @ u0.T @ b))

# pinv = np.linalg.pinv(L.todense())
# sol2 = pinv @ b
# print(sol - sol2)

L_dense = L.todense()
np.testing.assert_allclose(L_dense, L_dense.T)
identity = np.eye(n, dtype=L.dtype)

pinv = np.linalg.pinv(L_dense, hermitian=True)
pinv -= pinv @ u0 @ u0.T

pinv_solve = np.linalg.solve(L_dense, identity)
pinv_solve -= pinv_solve @ u0 @ u0.T

pinv_solve_deflated = np.linalg.solve(L_dense - u0 @ u0.T, identity)

pinv_spsolve = la.spsolve(L, identity)
pinv_spsolve -= pinv_spsolve @ u0 @ u0.T

pinv_cg = np.stack([la.cg(L, col - u0 @ u0.T @ col)[0] for col in identity], axis=0)

w, v = np.linalg.eigh(L_dense)
tol = 1e-5
w_inv = np.where(w < tol, np.zeros_like(w), 1 / w)
pinv_eig = v @ np.diag(w_inv) @ v.T

pinv_lstsq = np.linalg.lstsq(L_dense, identity)[0]
pinv_lstsq -= pinv_lstsq @ u0 @ u0.T

pinv_splstsq = np.stack([la.lsqr(L, i, iter_lim=1000)[0] for i in identity], axis=0)
pinv_splstsq -= pinv_splstsq @ u0 @ u0.T

# pinv -= u0 @ u0.T @ pinv

# print(pinv)
# print("---")
# print(pinv_eig)
# print("---")
# print(pinv_eig - pinv)


def compare(actual, expected, actual_title, expected_title):
    _, ax = plt.subplots(1, 3)
    vmin = min(expected.min(), actual.min())
    vmax = max(expected.max(), actual.max())
    kwargs = dict(vmin=vmin, vmax=vmax)
    ax[0].imshow(expected, **kwargs)
    ax[0].set_title(expected_title)
    ax[1].imshow(actual, **kwargs)
    ax[1].set_title(actual_title)
    ax[2].imshow(np.abs(expected - actual) / (vmax - vmin), vmin=0, vmax=1)
    ax[2].set_title("Difference")
    print(
        f"{actual_title} vs {expected_title} diff: {np.max(np.abs(actual - expected))}"
    )
    print(np.max(np.abs(actual)))
    print(np.max(np.abs(expected)))


compare(pinv, pinv_eig, "pinv", "pinv_eig")
compare(pinv, pinv_lstsq, "pinv", "pinv_lstsqr")
compare(pinv, pinv_splstsq, "pinv", "pinv_splstsq")
compare(pinv, pinv_solve, "pinv", "pinv_solve")
compare(pinv, pinv_solve_deflated, "pinv", "pinv_solve_deflated")
compare(pinv, pinv_spsolve, "pinv", "pinv_spsolve")
compare(pinv, pinv_cg, "pinv", "pinv_cg")
plt.show()
# plt.gca().imshow
