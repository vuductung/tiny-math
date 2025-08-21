import numpy as np

def check_for_square_matrix(matrix):
    """
    Checks if the input is a 2-dimensional square matrix.

    Parameters:
        matrix (np.ndarray): Input matrix.

    Returns:
        bool: True if matrix is square, False otherwise.

    Raises:
        IndexError: If input is not 2-dimensional.
        AttributeError: If input does not have a 'shape' attribute.
    """
    if not hasattr(matrix, "shape"):
        raise AttributeError("Input must have a 'shape' attribute (e.g., a numpy array).")
    if len(matrix.shape) != 2:
        raise IndexError("Input must be a 2-dimensional array.")
    return matrix.shape[0] == matrix.shape[1]

def check_for_2x2_matrix(matrix):
    """
    Checks if the input is a 2x2 matrix.

    Parameters:
        matrix (np.ndarray): Input matrix.

    Returns:
        bool: True if matrix is 2x2, False otherwise.

    Raises:
        IndexError: If input is not 2-dimensional.
        AttributeError: If input does not have a 'shape' attribute.
    """
    if not hasattr(matrix, "shape"):
        raise AttributeError("Input must have a 'shape' attribute (e.g., a numpy array).")
    if len(matrix.shape) != 2:
        raise IndexError("Input must be a 2-dimensional array.")
    return matrix.shape == (2, 2)

def det(matrix):
    """
    Computes the determinant of a 2x2 matrix.

    Parameters:
        matrix (np.ndarray): Input 2x2 matrix.

    Returns:
        float or int: Determinant value if input is 2x2 matrix.
        None: If input is not a 2x2 matrix.

    Raises:
        IndexError: If input is not 2-dimensional.
        AttributeError: If input does not have a 'shape' attribute.
    """
    if not hasattr(matrix, "shape"):
        raise AttributeError("Input must have a 'shape' attribute (e.g., a numpy array).")
    if len(matrix.shape) != 2:
        raise IndexError("Input must be a 2-dimensional array.")
    if not check_for_2x2_matrix(matrix):
        return None
    return matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]

def check_for_invertibility(matrix):
    """
    Checks if a 2x2 matrix is invertible (i.e., determinant is not zero).

    Parameters:
        matrix (np.ndarray): Input 2x2 matrix.

    Returns:
        bool: True if matrix is invertible, False if not invertible.
        None: If input is not a 2x2 matrix.

    Raises:
        IndexError: If input is not 2-dimensional.
        AttributeError: If input does not have a 'shape' attribute.
    """
    if not hasattr(matrix, "shape"):
        raise AttributeError("Input must have a 'shape' attribute (e.g., a numpy array).")
    if len(matrix.shape) != 2:
        raise IndexError("Input must be a 2-dimensional array.")
    if not check_for_2x2_matrix(matrix):
        return None
    return det(matrix) != 0

def inv(matrix):
    """
    Computes the inverse of a 2x2 matrix.

    Parameters:
        matrix (np.ndarray): Input 2x2 matrix.

    Returns:
        np.ndarray: Inverse of the input matrix if invertible.
        None: If input is not a 2x2 matrix or not invertible.

    Raises:
        IndexError: If input is not 2-dimensional.
        AttributeError: If input does not have a 'shape' attribute.
        ValueError: If the matrix is singular (not invertible).
    """
    if not check_for_2x2_matrix(matrix):
        return None

    determinant = det(matrix)
    if determinant == 0:
        raise ValueError("Matrix is singular and not invertible.")

    # For a 2x2 matrix [[a, b], [c, d]], the inverse is (1/det) * [[d, -b], [-c, a]]
    a, b = matrix[0, 0], matrix[0, 1]
    c, d = matrix[1, 0], matrix[1, 1]
    inverse = np.array([[d, -b], [-c, a]], dtype=matrix.dtype)
    return (1.0 / determinant) * inverse

def transpose(matrix):
    """
    Returns the transpose of the input matrix.

    Parameters:
        matrix (np.ndarray): Input 2D matrix.

    Returns:
        np.ndarray: Transposed matrix.

    Raises:
        AttributeError: If input does not have 'shape' attribute.
        IndexError: If input is not 2-dimensional.
    """
    if not hasattr(matrix, "shape"):
        raise AttributeError("Input must have a 'shape' attribute (e.g., a numpy array).")
    if len(matrix.shape) != 2:
        raise IndexError("Input must be a 2-dimensional array.")

    rows, cols = matrix.shape
    matrix_transposed = np.empty((cols, rows), dtype=matrix.dtype)
    for i in range(rows):
        for j in range(cols):
            matrix_transposed[j, i] = matrix[i, j]
    return matrix_transposed