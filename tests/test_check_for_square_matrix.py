import unittest
import numpy as np
from tiny_math.linear_algebra import check_for_square_matrix


class TestCheckForSquareMatrix(unittest.TestCase):
    def test_returns_true_for_square_matrix(self):
        matrix = np.array([[1, 2], [3, 4]])
        self.assertTrue(check_for_square_matrix(matrix))

    def test_returns_false_for_rectangular_matrix(self):
        matrix = np.array([[1, 2, 3], [4, 5, 6]])
        self.assertFalse(check_for_square_matrix(matrix))

    def test_returns_true_for_empty_square_matrix(self):
        matrix = np.empty((0, 0))
        self.assertTrue(check_for_square_matrix(matrix))

    def test_raises_indexerror_for_one_dimensional_input(self):
        vector = np.array([1, 2, 3])
        with self.assertRaises(IndexError):
            check_for_square_matrix(vector)

    def test_raises_attributeerror_for_non_array_like_without_shape(self):
        not_array_like = [[1, 2], [3, 4]]
        with self.assertRaises(AttributeError):
            check_for_square_matrix(not_array_like)

if __name__ == "__main__":
    unittest.main()


