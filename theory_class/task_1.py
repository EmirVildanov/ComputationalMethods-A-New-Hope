from math import sqrt
from tabulate import tabulate
import plotly.express as px
import pandas as pd
import numpy as np

# Vandermonde matrix is completely positive. See prove here: https://scask.ru/a_book_matrix.php?id=104
from tasks.task1.main import find_matrix_condition_numbers, find_spectral_criterion_condition_number


def generate_generalized_vandermonde_matrix(n: int) -> np.array:
    A = np.zeros((n, n), float)
    for i in range(n):
        a_i = i + 1
        for j in range(n):
            A[i, j] = a_i ** ((j + 1) / 2)
    return A


def get_first_regularization_method_A_u(A: np.array, u: np.array, alfa: float):
    n = A.shape[0]
    new_A = A.T.dot(A) + np.eye(n) * alfa
    new_u = A.T.dot(A).dot(u)
    return new_A, new_u


def get_second_regularization_method_B_u(B: np.array, u: np.array, alfa: float):
    n = B.shape[0]
    new_A = B.T.dot(B) + np.eye(n) * alfa
    new_u = B.T.dot(np.linalg.inv(B).dot(u))
    return new_A, new_u


def print_statistics_for_sle(A: np.array, u: np.array, header: str):
    print(header)
    data = find_matrix_condition_numbers(A)
    header = ["Condition value name", "value"]
    print(f"{tabulate(data, header)}\n")


if __name__ == "__main__":
    n = 2
    A = generate_generalized_vandermonde_matrix(n)
    standard_vector = np.ones(n)
    u = A.dot(standard_vector)

    eigen_values, eigen_matrix = np.linalg.eig(A)
    assert sorted(eigen_values) == sorted(list(set(eigen_values))), "Eigen values repeats!"
    assert len([value for value in eigen_values if value <= 0]) == 0, "Eigen value is negative!"

    lambda_matrix = np.zeros((n, n))
    for i in range(n):
        lambda_matrix[i, i] = eigen_values[i]
    sqrt_lambda_matrix = np.zeros((n, n))
    for i in range(n):
        sqrt_lambda_matrix[i, i] = sqrt(eigen_values[i])

    B = eigen_matrix.dot(sqrt_lambda_matrix.dot(np.linalg.inv(eigen_matrix)))
    print(n)
    print(f"n = {n}")
    print_statistics_for_sle(A, u, f"Statistics for matrix A")
    print_statistics_for_sle(B, u, f"Statistics for matrix B")
    print(f"||A - B^2|| = {np.linalg.norm(A - np.linalg.matrix_power(B, 2))}\n")

    list_data = []
    for i in range(-2, 2):
        alfa = 10 ** i
        new_A, u = get_first_regularization_method_A_u(A, u, alfa)
        if np.linalg.det(new_A) != 0:
            list_data.append([alfa, find_spectral_criterion_condition_number(new_A)])

    np_data = np.array(list_data)
    x_label = "alfa = 10 in power"
    y_label = "cond_value"
    dp_data = pd.DataFrame(np_data, columns=[x_label, y_label])
    fig = px.scatter(dp_data, x=x_label, y=y_label)
    fig.show()

    list_data = []
    for i in range(-3, 3):
        alfa = 10 ** i
        new_B, u = get_second_regularization_method_B_u(B, u, alfa)
        if np.linalg.det(new_B) != 0:
            list_data.append([alfa, find_spectral_criterion_condition_number(new_B)])

    np_data = np.array(list_data)
    dp_data = pd.DataFrame(np_data, columns=[x_label, y_label])
    fig = px.scatter(dp_data, x=x_label, y=y_label)
    fig.show()