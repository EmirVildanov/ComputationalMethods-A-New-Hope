import numpy as np


# matrix should be:
# 1) With diagonal predominance
# 2) A_T = A + all eigen values are positive
def solve_with_simple_iteration(A: np.array, b) -> np.array:
    epsilon = 10 ** (-3)  # should vary it

    n = A.shape[0]

    alfa_matrix = np.zeros((n, n))
    beta_vector = b.copy()
    for i in range(n):
        for j in range(n):
            if i != j:
                alfa_matrix[i, j] = -(A[i, j] / A[i, i])
            else:
                alfa_matrix[i, j] = 0
        beta_vector[i] = beta_vector[i] / A[i, i]

    print(f"Norm A: {np.linalg.norm(A)}")
    print(f"Eig: {np.linalg.eig(A)[0]}")
    print((np.eye(n) - alfa_matrix).dot(np.linalg.inv(A)))
    x = beta_vector.copy()
    for i in range(10):
        np.set_printoptions(3)
        print(x)
        x = beta_vector + alfa_matrix.dot(x)

    return x


def zeidel_method(A, b) -> np.array:
    epsilon = 10 ** (-3)  # should vary it


if __name__ == "__main__":
    A = np.array([[6, 5, 0],
                  [5, 1, 4],
                  [0, 4, 3]], float)

    A = np.array([[12, 1, 3],
                  [1, 1, 0],
                  [-1, -1, 1]], float)

    b = np.array([1, 1, 1], float)

    expected = np.linalg.solve(A, b)
    print(expected)
    print()
    actual = solve_with_simple_iteration(A, b)

    print(np.linalg.norm(actual - expected))
