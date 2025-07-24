"""
02 - Linear Systems
-------------------
In this file, you will learn how to represent and solve systems of linear equations using Python, with academic-level explanations.
"""
import numpy as np

# Linear system:
# A system of m linear equations in n variables can be written as:
#   a11*x1 + a12*x2 + ... + a1n*xn = b1
#   a21*x1 + a22*x2 + ... + a2n*xn = b2
#   ...
#   am1*x1 + am2*x2 + ... + amn*xn = bm
# In matrix form: A·x = b, where A ∈ R^{m x n}, x ∈ R^n, b ∈ R^m

# Example:
# 2x + y = 5
# x + 3y = 7
A = np.array([[2, 1], [1, 3]])  # Coefficient matrix
b = np.array([5, 7])            # Right-hand side vector

# Solution:
# If A is square (n x n) and invertible (rank(A) = n), the system has a unique solution:
# x = A^{-1}·b
# In practice, use numpy.linalg.solve for numerical stability.
x = np.linalg.solve(A, b)
print('Solution x:', x)

# Verification:
# Substitute x back into the original equations: A·x should equal b
print('Verification (A·x):', np.dot(A, x))

# Singular or underdetermined systems:
# If rank(A) < n, the system may have infinitely many or no solutions.
# Example: dependent equations
A_dep = np.array([[1, 2], [2, 4]])
b_dep = np.array([3, 6])
rank_A_dep = np.linalg.matrix_rank(A_dep)
print('\nRank of A_dep:', rank_A_dep)
try:
    x_dep = np.linalg.solve(A_dep, b_dep)
    print('Solution for dependent system:', x_dep)
except np.linalg.LinAlgError as e:
    print('No unique solution for dependent system:', e) 

