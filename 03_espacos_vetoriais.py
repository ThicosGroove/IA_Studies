"""
03 - Vector Spaces and Subspaces
--------------------------------
In this file, you will learn about vector spaces, subspaces, basis, dimension, and linear independence, with academic-level explanations and mathematical terminology.
"""
import numpy as np

# Vector Space:
# A vector space V over a field F (e.g., the real numbers R) is a set equipped with two operations:
# 1. Vector addition: u + v ∈ V for all u, v ∈ V
# 2. Scalar multiplication: a*v ∈ V for all a ∈ F, v ∈ V
# Satisfying properties such as associativity, commutativity, existence of zero vector, additive inverses, distributivity, etc.

# Example: R^2 is a vector space over R.

# Subspace:
# A subspace W of a vector space V is a subset of V that is itself a vector space under the same operations.
# To check if W is a subspace:
# 1. The zero vector is in W
# 2. Closed under addition: if x, y ∈ W, then x + y ∈ W
# 3. Closed under scalar multiplication: if x ∈ W and a ∈ W, then a*x ∈ W

# Example: All vectors of the form [x, 0] in R^2 form a subspace of R^2.

# Basis and Dimension:
# A basis of a vector space V is a set of linearly independent vectors that span V.
# The number of vectors in a basis is called the dimension of V.

# Linear Independence:
# A set of vectors {v1, v2, ..., vn} is linearly independent if the only solution to c1*v1 + c2*v2 + ... + cn*vn = 0 is c1 = c2 = ... = cn = 0.
# Otherwise, they are linearly dependent.

# Example: Check if a set of vectors is linearly independent using the rank of the matrix.
vectors = np.array([[1, 2], [2, 4]])
rank = np.linalg.matrix_rank(vectors)
print('Rank of the set:', rank)
print('Number of vectors:', vectors.shape[0])
if rank == vectors.shape[0]:
    print('The vectors are linearly independent')
else:
    print('The vectors are linearly dependent')

# Example with linearly independent vectors:
vectors_indep = np.array([[1, 0], [0, 1]])
rank_indep = np.linalg.matrix_rank(vectors_indep)
print('\nRank of independent set:', rank_indep)
if rank_indep == vectors_indep.shape[0]:
    print('The vectors are linearly independent')
else:
    print('The vectors are linearly dependent')

# Canonical basis of R^3:
canonical_basis = np.eye(3)
print('\nCanonical basis of R^3:\n', canonical_basis)
print('Dimension of R^3:', canonical_basis.shape[0])

# Linear Transformation:
# A function T: V → W between vector spaces is a linear transformation if:
# 1. T(x + y) = T(x) + T(y) for all x, y ∈ V
# 2. T(a*x) = a*T(x) for all a ∈ F, x ∈ V

# Example: T: R^2 → R^2 defined by T([x, y]) = [2x, 3y]
def T2(v):
    return np.array([2*v[0], 3*v[1]])

x = np.array([1, 2])
y = np.array([3, 4])
a = 5

print('\nT(x):', T2(x))
print('T(y):', T2(y))
print('T(x + y):', T2(x + y))
print('T(x) + T(y):', T2(x) + T2(y))
print('T(a * x):', T2(a * x))
print('a * T(x):', a * T2(x))

# The outputs above illustrate the linearity properties:
# T(x + y) == T(x) + T(y)
# T(a * x) == a * T(x) 

import numpy as np

print('\n--- Exercício: Transformação Linear T: R^3 → R^2 ---')

# 1- Encontrar a matriz associada a T
# T(x, y, z) = (2x - y + z, x + 3y - 2z)
# A matriz associada é obtida aplicando T aos vetores canônicos de R^3

# Vetores canônicos de R^3
e1 = np.array([1, 0, 0])
e2 = np.array([0, 1, 0])
e3 = np.array([0, 0, 1])

def T(v):
    x, y, z = v
    return np.array([2*x - y + z, x + 3*y - 2*z])

# Colunas da matriz associada
T_e1 = T(e1)  # T(1,0,0)
T_e2 = T(e2)  # T(0,1,0)
T_e3 = T(e3)  # T(0,0,1)

A = np.column_stack([T_e1, T_e2, T_e3])
print('\n1) Matriz associada a T:')
print(A)

# 2- Determinar o Nullspace (Núcleo) e sua dimensão (nullity)
# Nullspace: soluções de A @ v = 0
from scipy.linalg import null_space

nullspace = null_space(A)
print('\n2) Núcleo (Nullspace) de T:')
print(nullspace)
print('Dimensão do núcleo (nullity):', nullspace.shape[1])

# 3- Encontrar o Range (Imagem) e sua dimensão (Rank)
rank = np.linalg.matrix_rank(A)
print('\n3) Imagem (Range) de T: os vetores gerados pelas colunas de A')
print('Rank (dimensão da imagem):', rank)

# Podemos mostrar uma base para a imagem:
from sympy import Matrix
A_sym = Matrix(A)
imagem_base = A_sym.columnspace()
print('Base da imagem:')
for v in imagem_base:
    print(np.array(v).astype(np.float64).flatten())

# 4- Verificar o teorema do núcleo e imagem (Rank-Nullity Theorem)
# dim(Núcleo) + dim(Im) = dim(domínio)
nullity = nullspace.shape[1]
dim_imagem = rank
dim_dominio = 3
print('\n4) Teorema do Núcleo e Imagem (Rank-Nullity):')
print(f'nullity + rank = {nullity} + {dim_imagem} = {nullity + dim_imagem}')
print('Dimensão do domínio:', dim_dominio)
if nullity + dim_imagem == dim_dominio:
    print('O teorema do núcleo e imagem é satisfeito.')
else:
    print('O teorema do núcleo e imagem NÃO é satisfeito.')

# --- Exemplo: Determinando a base de um espaço vetorial em R^2 ---
# Dados os vetores v1 = <1,0>, v2 = <1,1>, v3 = <3,2> em R^2
# Vamos determinar uma base para o subespaço gerado por eles.

v1 = np.array([1, 0])
v2 = np.array([1, 1])
v3 = np.array([3, 2])

# Empilhando os vetores como linhas de uma matriz
V = np.vstack([v1, v2, v3])
print('\n--- Exemplo: Base do subespaço gerado por v1, v2, v3 em R^2 ---')
print('Matriz dos vetores (cada linha é um vetor):')
print(V)

# Forma 1: Usando o posto (rank) para identificar independência linear
dim = np.linalg.matrix_rank(V)
print(f'\nDimensão do subespaço gerado: {dim}')

# Podemos usar o método de eliminação de Gauss para encontrar uma base
from sympy import Matrix
V_sym = Matrix(V)
V_rref, pivots = V_sym.T.rref()  # Transpomos para trabalhar com colunas
print('\nBase encontrada por eliminação de Gauss (colunas linearmente independentes):')
for i in pivots:
    print(np.array(V[:, i]))

# Alternativamente, o método numpy.linalg.qr pode ser usado para obter vetores ortogonais (mas não necessariamente normalizados)
Q, R = np.linalg.qr(V.T)
print('\nBase ortogonal (via QR, pode conter zeros se vetores forem dependentes):')
for i in range(dim):
    print(Q[:, i])

# Resumo:
# - A base do subespaço gerado por v1, v2, v3 pode ser obtida selecionando os vetores linearmente independentes.
# - Tanto o método do posto (rank) quanto a eliminação de Gauss (rref) ajudam a identificar quais vetores formam a base.
