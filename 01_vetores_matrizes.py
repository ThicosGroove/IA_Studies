"""
01 - Vectors and Matrices
-------------------------
In this file, you will learn the fundamental concepts of vectors and matrices, which are the building blocks of linear algebra and essential for Machine Learning.
"""
import numpy as np

# Vector:
# A vector is an ordered list of numbers (components), which can be interpreted as a point or a direction in space.
# Notation: v ∈ R^n, where n is the dimension.

vector_a = np.array([1, 2, 3])  # v = [1, 2, 3]^T ∈ R^3
vector_b = np.array([4, 5, 6])
print('Vector a:', vector_a)
print('Vector b:', vector_b)

# Vector addition:
# For u, v ∈ R^n: u + v = [u1 + v1, u2 + v2, ..., un + vn]
sum_vectors = vector_a + vector_b
print('Sum of vectors:', sum_vectors)

# Scalar multiplication:
# For α ∈ R, v ∈ R^n: α*v = [α*v1, α*v2, ..., α*vn]
alpha = 2
scaled_vector = alpha * vector_a
print('Scalar multiplication (2 * a):', scaled_vector)

# Dot product (inner product):
# For u, v ∈ R^n: u · v = u1*v1 + u2*v2 + ... + un*vn
# Geometric interpretation: u · v = ||u|| * ||v|| * cos(θ)
dot_product = np.dot(vector_a, vector_b)
print('Dot product (a · b):', dot_product)

# Norm (magnitude) of a vector:
# ||v|| = sqrt(v1^2 + v2^2 + ... + vn^2)
norm_a = np.linalg.norm(vector_a)
print('Norm of vector a:', norm_a)

# Matrix:
# A matrix is a rectangular array of numbers arranged in rows and columns.
# Notation: A ∈ R^{m x n} (m rows, n columns)

matrix_a = np.array([[1, 2], [3, 4]])  # 2x2 matrix
matrix_b = np.array([[5, 6], [7, 8]])
print('Matrix A:\n', matrix_a)
print('Matrix B:\n', matrix_b)

# Matrix addition:
# For A, B ∈ R^{m x n}: (A + B)_{ij} = A_{ij} + B_{ij}
sum_matrices = matrix_a + matrix_b
print('Sum of matrices:\n', sum_matrices)

# Matrix multiplication:
# For A ∈ R^{m x n}, B ∈ R^{n x p}: (A·B)_{ij} = sum_k A_{ik} * B_{kj}
product_matrices = np.dot(matrix_a, matrix_b)
print('Matrix multiplication (A·B):\n', product_matrices)

# Transpose:
# For A ∈ R^{m x n}: A^T ∈ R^{n x m}, (A^T)_{ij} = A_{ji}
transpose_a = matrix_a.T
print('Transpose of matrix A:\n', transpose_a)

# Special matrices:
# Identity matrix: I ∈ R^{n x n}, I_{ii} = 1, I_{ij} = 0 for i ≠ j
identity = np.eye(3)
print('Identity matrix 3x3:\n', identity)

# Diagonal matrix: only nonzero entries are on the main diagonal
diagonal = np.diag([1, 2, 3])
print('Diagonal matrix:\n', diagonal) 