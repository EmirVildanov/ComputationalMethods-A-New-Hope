# Реализовать метод решения СЛАУ, на выбор: метод вращений или метод отражений(поощ).
# Вычислить числа обусловленности.
# Протестировать на тех же матрицах, что использовались в задании 2; сравнить

# метод вращений
import math

import numpy as np

from tasks.task2.main import back_sub


def create_rotational_matrix(n, sin, cos, i, j) -> np.array:
    matrix = np.eye(n)
    # sin = math.sin(math.radians(fi))
    # cos = math.cos(math.radians(fi))

    matrix[i, i] = cos
    matrix[i, j] = -sin
    matrix[j, i] = sin
    matrix[j, j] = cos
    return matrix


def get_sin_cos(a, b):
    if a == 0 and b == 0:
        return 0, 1
    r = math.sqrt(a ** 2 + b ** 2)
    return -(b / r), a / r


    sign = lambda x: -1 if x < 0 else (0 if x == 0 else 1)

    if b == 0:
        cos = sign(a)
        if cos == 0:
            cos = 1
        sin = 0
    elif a == 0:
        cos = 0
        sin = sign(b)
    elif abs(a) > abs(b):
        t = b / a
        u = sign(a) * math.sqrt(1 + t * t)
        cos = 1 / u
        sin = cos * t
    else:
        t = a / b
        u = sign(b) * math.sqrt(1 + t * t)
        sin = 1 / u
        cos = sin * t
    return sin, cos


def qr_decompose(A: np.array):
    n = A.shape[0]

    Q = np.eye(n)
    R = A.copy()
    for j in range(n):
        for i in range(n - 1, j, -1):
            a, b = R[i - 1, j], R[i, j]
            sin, cos = get_sin_cos(a, b)
            rotational_matrix = create_rotational_matrix(n, sin, cos, i, j)
            R = rotational_matrix.T.dot(R)
            Q = Q.dot(rotational_matrix)
    return Q, R

# to solve QRx = b
# Q_tQRx = Q_tb
# Rx = Q_tb
def qr_solve(A, b):
    Q, R = qr_decompose(A)
    y = Q.T.dot(b)
    x = back_sub(R, y)
    return x


if __name__ == "__main__":
    A = np.array([[6, 5, 0],
                  [5, 1, 4],
                  [0, 4, 3]])
    b = np.array([1, 1, 1])

    expected = np.linalg.solve(A, b)
    actual = qr_solve(A, b)
    print(np.linalg.norm(expected - actual))
