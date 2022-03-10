# Реализовать метод решения СЛАУ, на выбор: LU-разложение или метод квадратного корня.
# Для матриц A, L, U вычислить числа обусловленности (см. задание 1).
# Протестировать на разных матрицах: хорошо обусловленных, [очень] плохообусловленных.

# LU-разложение

# Для нескольких плохо обусловленных матриц (например, для матриц Гильберта разного,больше 15, порядка) реализовать метод регуляризации:
# * параметр α варьировать впределах от 10−12 до 10−1
# * длякаждогоконкретногозначенияαнайтичислаобусловленности (матрицA+αE)инормупогрешностиполучившегосярешения
# * понять,какоезначениеα=αвкаждомконкретномслучае (=длякаждойконкретнойматрицы)кажетсянаилучшим

# Наилучшееαможно
# * находитьизпредположений,чтоточнымрешениемявляетсявектор x0=(1, 1,..., 1)T
# * находитьизпредположений,чтоточнымрешениемявляетсяслучайныйвектор x0
# Проверитьрезультатна [другом]случайномвекторе x0
import numpy as np


# see https://courses.engr.illinois.edu/cs357/sp2020/notes/ref-9-linsys.html

# to solve Lx = b
def forward_sub(L, b):
    n = L.shape[0]
    x = np.zeros(n)
    for i in range(n):
        tmp = b[i]
        for j in range(i):
            tmp -= L[i, j] * x[j]
        x[i] = tmp / L[i, i]
    return x


# to solve Ux = b
def back_sub(U, b):
    n = U.shape[0]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        tmp = b[i]
        for j in range(i + 1, n):
            tmp -= U[i, j] * x[j]
        x[i] = tmp / U[i, i]
    return x


def pivot_matrix(matrix: np.array):
    n = matrix.shape[0]

    # Create an identity matrix, with floating point values
    id_mat = np.eye(n)

    # Rearrange the identity matrix such that the largest element of
    # each column of M is placed on the diagonal of M
    for j in range(n):
        row = max(range(j, n), key=lambda i: abs(matrix[i, j]))
        if j != row:
            # Swap the rows
            id_mat[[j, row]] = id_mat[[row, j]]

    return id_mat


def eliminate_matrix_by_gauss(A):
    n = A.shape[0]
    for k in range(0, n - 1):
        if A[k, k] == 0:
            return
        for i in range(k + 1, n):
            A[i, k] = A[i, k] / A[k, k]
        for j in range(k + 1, n):
            for i in range(k + 1, n):
                A[i, j] -= A[i, k] * A[k, j]
    return A


def get_LU_from_eliminated_matrix(A: np.array):
    n = A.shape[0]
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i <= j:
                U[i, j] = A[i, j]
            if i == j:
                L[i, j] = 1
            if i > j:
                L[i, j] = A[i, j]

    return L, U


# to solve LUx = b
def lu_solve(A, b):
    P = pivot_matrix(A)
    A = P.dot(A)
    b = P.dot(b)

    eliminated_matrix = eliminate_matrix_by_gauss(A)
    L, U = get_LU_from_eliminated_matrix(eliminated_matrix)

    y = forward_sub(L, b)
    x = back_sub(U, y)
    return x
