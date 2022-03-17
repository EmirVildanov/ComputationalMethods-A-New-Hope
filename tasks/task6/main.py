import math

import numpy as np

from tasks.task3.main import create_rotational_matrix


def check_gershgorin(A: np.array, eigen_values: list) -> bool:
    n = len(eigen_values)
    r = []
    for i in range(n):
        sum = 0
        for j in range(n):
            if j == i:
                continue
            sum += abs(A[i, j])
        r.append(sum)

    for eigen_value, r_i in zip(eigen_values, r):
        for i in range(n):
            if abs(eigen_value - A[i, i]) < r_i:
                continue
            return False
    return True


def strategy_of_nulling_element1():
    print(1)


def strategy_of_nulling_element2():
    print(1)


# Yakobi method only for Ermit's matrices
def find_eig_values(A: np.array):
    n = A.shape[0]

    # from -2 to -5
    epsilon = 10 ** (-5)
    # until epsilon discrepancy
    # see iteration steps number from epsilon (the number of nulling element steps)

    # test on Hilbert matrix > 5
    # print iteration steps number
    # check eigen values using Gershgorin's theorem

    Q = np.eye(n)
    R = A.copy()
    for j in range(n):
        for i in range(n - 1, j, -1):
            a, b = R[i - 1, j], R[i, j]
            # sin, cos = get_sin_cos(a, b)

            if A[i, i]:
                fi = math.pi / 4

            rotational_matrix = create_rotational_matrix(n, sin, cos, i, j)
            R = rotational_matrix.T.dot(R)
            Q = Q.dot(rotational_matrix)

    print(1)


if __name__ == "__main__":
    A = np.array([[12, 1, 3],
                  [1, 1, 0],
                  [-1, -1, 1]], float)

    eig = np.linalg.eig(A)
    print(eig[0])
