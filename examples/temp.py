import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as la

from graph_tf.data.single import get_data
from graph_tf.utils.scipy_utils import to_scipy

A = to_scipy(get_data("cora").adjacency)
print("Got data")
num_nodes = A.shape[0]
d = np.array(A.sum(1)).squeeze(1)
num_components, labels = sp.csgraph.connected_components(
    A, return_labels=True, directed=False
)
u0 = np.zeros((num_nodes, num_components))
d_sqrt = np.sqrt(d)
for i in range(num_components):
    mask = labels == i
    u0[mask, i] = d_sqrt[mask]
u0 = u0 / np.linalg.norm(u0, axis=0, keepdims=True)

# num_components, labels = get_component_labels(A, directed=False)
# d = tf.sparse.reduce_sum(A, axis=0)

d = sp.diags(d**-0.5)
I = sp.eye(A.shape[0], dtype=A.dtype)

L = I - d @ A @ d
np.testing.assert_allclose(L @ u0, np.zeros_like(u0), atol=1e-6)
print("Naive")
# w, v = la.eigsh(A, num_components, which="SM")
w0, v0 = la.eigsh(L, k=num_components, which="SM")
print(w0)
print("Shifted")
w1, v1 = la.eigsh(L - I * 2, k=num_components, which="LM")
w1 += 2
print(w1)

print("Dense")
w2, v2 = np.linalg.eigh(L.todense())
print(w2[:num_components])

w3, v3 = la.eigsh(L - 2 * I)

print(np.stack((w0, w1, w2[:num_components]), axis=1))

import matplotlib.pyplot as plt

fig, (ax0, ax1, ax2) = plt.subplots(1, 3)
ax0.imshow(np.abs(v0.T @ v1))
ax1.imshow(np.abs(v0.T @ v2[:, :num_components]))
ax2.imshow(np.abs(v1.T @ v2[:, :num_components]))
plt.show()
