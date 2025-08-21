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
