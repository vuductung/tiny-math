import numpy as np

def check_for_square_matrix(matrix):
    if len(matrix.shape) != 2:
        raise IndexError("Input must be a 2-dimensional array.")
    return matrix.shape[0] == matrix.shape[1]

