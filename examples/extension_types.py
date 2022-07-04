import tensorflow as tf

from graph_tf.projects.igcn.extension_types import ConjugateGradientSolver
from graph_tf.utils.linalg import SparseLinearOperator
from graph_tf.utils.test_utils import random_laplacian

tf.linalg.LinearOperator
n = 10
seed = 0
max_iter = 100
tol = 1e-5
rng = tf.random.Generator.from_seed(seed)
L, _ = random_laplacian(n, n * 2, rng, shift=-0.1)
L = SparseLinearOperator(
    L, is_self_adjoint=True, is_square=True, is_positive_definite=True
)
x = rng.normal((n,), dtype=tf.float32)
x = tf.linalg.LinearOperatorFullMatrix(tf.expand_dims(x, 1))
dataset = tf.data.Dataset.from_tensors(x)
for el in dataset:
    print(el.shape)
exit()
solver = ConjugateGradientSolver(L, max_iter, tol)
solver.matvec(x)

print(tf.type_spec_from_value(solver))

dataset = tf.data.Dataset.from_tensors(solver)
for s in dataset:
    s @ x
print("Passed from_tensors")


# def gen():
#     yield solver


# dataset = tf.data.Dataset.from_generator(
#     gen, output_signature=ConjugateGradientSolver.Spec()
# )
# for s in dataset:
#     s @ x
# print("Passed from_generator")
