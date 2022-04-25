import random
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt


def generate_hilbert_matrix(n: int, digits: int = -1) -> np.array:
    matrix = np.empty(shape=(n, n))
    for i in range(n):
        for j in range(n):
            current_value = 1 / (i + j + 1)
            if digits != -1:
                if digits <= 0:
                    raise AttributeError("digits must be positive value")
                current_value = round(current_value, digits)
            matrix[i, j] = current_value

    return matrix


def generate_tridiagonal_matrix(n: int, digits: int = -1) -> np.array:
    matrix = np.empty(shape=(n, n))
    for i in range(n):
        for j in range(n):
            if j == i - 1 or j == i + 1:
                current_value = -1
            elif j == i:
                current_value = 2
            else:
                current_value = 0
            if digits != -1:
                if digits <= 0:
                    raise AttributeError("digits must be positive value")
                current_value = round(current_value, digits)
            matrix[i, j] = current_value
    return matrix


def generate_strong_tridiagonal_matrix(n: int, digits: int = -1) -> np.array:
    matrix = np.empty(shape=(n, n))
    for i in range(n):
        for j in range(n):
            if j == i - 1 or j == i + 1:
                current_value = -1
            elif j == i:
                current_value = 10 ** 4
            else:
                current_value = random.uniform(7, 115)
            if digits != -1:
                if digits <= 0:
                    raise AttributeError("digits must be positive value")
                current_value = round(current_value, digits)
            matrix[i, j] = current_value
    return matrix


def generate_diagonal_matrix(n: int, digits: int = -1) -> np.array:
    matrix = np.empty(shape=(n, n))
    for i in range(n):
        for j in range(n):
            if j == i:
                current_value = 1
            else:
                current_value = 0
            if digits != -1:
                if digits <= 0:
                    raise AttributeError("digits must be positive value")
                current_value = round(current_value, digits)
            matrix[i, j] = current_value
    return matrix


def generate_random_diagonal_matrix(n: int, digits: int = -1) -> np.array:
    matrix = np.empty(shape=(n, n))
    for i in range(n):
        for j in range(n):
            if j == i:
                current_value = random.uniform(7.0, 113.0)
            else:
                current_value = 0
            if digits != -1:
                if digits <= 0:
                    raise AttributeError("digits must be positive value")
                current_value = round(current_value, digits)
            matrix[i, j] = current_value
    return matrix


def find_spectral_criterion_condition_number(matrix: np.ndarray) -> float:
    return np.linalg.norm(matrix) * np.linalg.norm(np.linalg.inv(matrix))


def find_volume_criterion_condition_number(matrix: np.ndarray) -> float:
    n = matrix.shape[0]
    multiplication = 1

    for i in range(0, n):
        sum = 0
        for j in range(0, n):
            sum += matrix[i, j] ** 2
        multiplication *= sqrt(sum)
    return multiplication / np.linalg.det(matrix)


def find_angular_criterion_condition_number(matrix: np.ndarray) -> float:
    n = matrix.shape[0]
    inverse_matrix = np.linalg.inv(matrix)
    products = []

    for i in range(n):
        a_n = matrix[i]
        c_n = inverse_matrix[:, i]
        products.append(np.linalg.norm(a_n) * np.linalg.norm(c_n))
    return max(products)


def solve_linear_equation(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.solve(A, b)
    # inverse_A = np.linalg.inv(A)
    # return inverse_A.dot(b)


def find_matrix_condition_numbers(matrix: np.array) -> list:
    condition_numbers = []
    condition_numbers.append(("Spectral", find_spectral_criterion_condition_number(matrix)))
    condition_numbers.append(("Angular", find_angular_criterion_condition_number(matrix)))
    condition_numbers.append(("Volume", find_volume_criterion_condition_number(matrix)))
    return condition_numbers


def matrix_and_n(matrix, cond_value=10 ** 4):
    n = matrix.shape[0]
    standard_vector = np.ones(n)
    b = matrix.dot(standard_vector)
    solution = solve_linear_equation(matrix, b)

    discrepancies = []
    condition_numbers = find_matrix_condition_numbers(matrix)
    digits = []
    for i in range(20, 1, -1):
        varied_matrix = np.round(matrix, i)
        varied_b = varied_matrix.dot(standard_vector)
        # varied_b = np.around(b)
        varied_solution = solve_linear_equation(varied_matrix, varied_b)
        discrepancies.append(np.linalg.norm(solution - varied_solution))
        digits.append(i)
        if i == 20:
            # print(varied_matrix)
            print(solution)
            print(varied_solution)
            print()
            print(f"20 -> {np.linalg.norm(solution - varied_solution)}")
        elif i == 2:
            print(solution)
            print(varied_solution)
            print()
            print(f"2 -> {np.linalg.norm(solution - varied_solution)}")

    # plt.title("Discrepancy to digits dependency")
    # plt.plot(digits, discrepancies)
    # plt.xlabel("Digits")
    # plt.ylabel("Discrepancy")
    # plt.show()

    print(f"n: {n}")
    print(f"Condidion numbers")
    for name, value in condition_numbers:
        print(f"{name} -> {value}, {value > cond_value} that it is > {cond_value}")


if __name__ == "__main__":
    n = 9
    matrix = generate_hilbert_matrix(n)
    matrix_and_n(matrix)