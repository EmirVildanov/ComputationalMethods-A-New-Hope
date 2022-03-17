import numpy as np


if __name__ == "__main__":
    A = np.array([[12, 1, 3],
                  [1, 1, 0],
                  [-1, -1, 1]], float)

    eig = np.linalg.eig(A)
    print(eig[0])
