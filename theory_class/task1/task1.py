from math import sqrt
import numpy as np

# Vandermonde matrix is completely positive. See prove here: https://scask.ru/a_book_matrix.php?id=104
def generate_generalized_vandermonde_matrix(n: int) -> np.array:
    A = np.zeros((n, n), float)
    for i in range(n):
        a_i = i + 1
        for j in range(n):
            b_j = (j + 1) / 2
            A[i, j] = a_i ** b_j
    return A


def get_first_regularization_method_A_u(A: np.array, u: np.array, alfa: float):
    A = np.matrix(A)
    n = A.shape[0]
    new_A = A.H @ A + np.eye(n) * alfa
    new_u = A.H @ u
    return np.array(new_A), np.array(new_u).flatten()

def get_second_regularization_method_B_u(B: np.array, u: np.array, alfa: float):
    B = np.matrix(B)
    n = B.shape[0]
    inverse_B = np.linalg.inv(B)
    new_B = B.H @ B + np.eye(n) * alfa
    first = np.array(inverse_B @ u).flatten()
    new_u = B.H @ first
    return np.array(new_B), np.array(new_u).flatten()


def find_sqrt_root_of_matrix(A):
    n = A.shape[0]
    eigen_values, eigen_matrix = np.linalg.eig(A)
    assert sorted(eigen_values) == sorted(list(set(eigen_values))), f"Eigen values repeats! \n {eigen_values}"
    assert len([value for value in eigen_values if value <= 0]) == 0, f"Eigen value is negative! {n} \n {eigen_values}"

    lambda_matrix = np.zeros((n, n))
    for i in range(n):
        lambda_matrix[i, i] = eigen_values[i]
    sqrt_lambda_matrix = np.zeros((n, n))
    for i in range(n):
        sqrt_lambda_matrix[i, i] = sqrt(eigen_values[i])

    B = eigen_matrix.dot(sqrt_lambda_matrix.dot(np.linalg.inv(eigen_matrix)))
    return B

def pretty_print(A, precision):
    np.set_printoptions(precision=precision)
    print(A)