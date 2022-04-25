# Реализовать метод решения СЛАУ, на выбор: метод вращений или метод отражений(поощ).
# Вычислить числа обусловленности.
# Протестировать на тех же матрицах, что использовались в задании 2; сравнить

# метод вращений
import math

import numpy as np

from practice_class.task1.main import generate_hilbert_matrix, find_spectral_criterion_condition_number
from practice_class.task2.main import back_sub


def create_rotational_matrix(n, sin, cos, i, j) -> np.array:
    matrix = np.eye(n)
    matrix[i, i] = cos
    matrix[i, j] = -sin
    matrix[j, i] = sin
    matrix[j, j] = cos
    return matrix


def get_sin_cos(a, b):
    if a == 0 and b == 0:
        return 0, 1
    r = math.sqrt(a ** 2 + b ** 2)
    return b / r, -(a / r)


def qr_decompose(A: np.array):
    n = A.shape[0]

    Q = np.eye(n)
    R = A.copy()
    for j in range(n):
        for i in range(n - 1, j, -1):
            a, b = R[i - 1, j], R[i, j]
            sin, cos = get_sin_cos(a, b)
            rotational_matrix = create_rotational_matrix(n, sin, cos, i, j)
            R = rotational_matrix.T @ R
            Q = Q @ rotational_matrix
    return Q, R

# to solve QRx = b
# Q_tQRx = Q_tb
# Rx = Q_tb
def qr_solve(A, b):
    Q, R = qr_decompose(A)

    print(f"Condition A: {find_spectral_criterion_condition_number(A)}")
    print(f"Condition Q: {find_spectral_criterion_condition_number(Q)}")
    print(f"Condition R: {find_spectral_criterion_condition_number(R)}")

    y = Q.T.dot(b)
    x = back_sub(R, y)
    return x


if __name__ == "__main__":
    # A = np.array([[6, 5, 0],
    #               [5, 1, 4],
    #               [0, 4, 3]])
    # b = np.array([1, 1, 1])
    #
    # expected = np.linalg.solve(A, b)
    # actual = qr_solve(A, b)
    # print(np.linalg.norm(expected - actual))
    n = 15
    A = generate_hilbert_matrix(n)
    b = np.array(np.random.rand(n) * 100, float)

    actual = qr_solve(A, b)
    expected = np.linalg.solve(A, b)
    print(np.linalg.norm(actual - expected))
