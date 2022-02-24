from typing import List

from utils.utils import print_task_info
import numpy as np


# 4-5 tests
# task description
# task solution description
# showing tests (expect, actual). Where is discrepancy from

# bad system = slightly change b -> x change enormously
# bad system = big matrix

# the number of conditionality shows if matrix is bad (> 10^4)
# but we may get big number on diagonal matrix (but they are solving very good)

def generate_hilbert_matrix(n: int, digits: int = -1) -> List[List[float]]:
    matrix_rows = []
    for i in range(n):
        current_row = []
        for j in range(n):
            current_value = 1 / (i + j + 1)
            if digits != -1:
                if digits <= 0:
                    raise AttributeError("digits must be positive value")
                current_value = round(current_value, digits)
            current_row.append(current_value)
    return matrix_rows


def find_spectral_criterion_condition_number(matrix: np.ndarray) -> float:
    return np.linalg.norm(matrix) * np.linalg.norm(np.linalg.inv(matrix))


def find_volume_criterion_condition_number(matrix: np.ndarray):
    return np.linalg.cond(matrix)


def find_angular_criterion_condition_number(matrix: np.ndarray):
    print(1)


if __name__ == '__main__':
    print_task_info("The effect of an error on the solution of a system of linear equations")
    print_task_info("Solving equation Ax = b")

    A_matrix_rows = [[4, 3], [-5, 9]]
    A = np.array(A_matrix_rows)
    b = np.array([20, 26])

    inverse_A = np.linalg.inv(A)
    x = inverse_A.dot(b)
    print(x)
