import unittest
import numpy as np
from tiny_math.linear_algebra import (
    check_for_square_matrix,
    check_for_2x2_matrix,
    det,
    check_for_invertibility,
    inv,
    transpose
)


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


class TestCheckFor2x2Matrix(unittest.TestCase):
    def test_returns_true_for_2x2_matrix(self):
        matrix = np.array([[1, 2], [3, 4]])
        self.assertTrue(check_for_2x2_matrix(matrix))

    def test_returns_false_for_3x3_matrix(self):
        matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.assertFalse(check_for_2x2_matrix(matrix))

    def test_returns_false_for_1x1_matrix(self):
        matrix = np.array([[5]])
        self.assertFalse(check_for_2x2_matrix(matrix))

    def test_returns_false_for_rectangular_matrix(self):
        matrix = np.array([[1, 2, 3], [4, 5, 6]])
        self.assertFalse(check_for_2x2_matrix(matrix))

    def test_returns_false_for_empty_matrix(self):
        matrix = np.empty((0, 0))
        self.assertFalse(check_for_2x2_matrix(matrix))

    def test_raises_indexerror_for_one_dimensional_input(self):
        vector = np.array([1, 2, 3])
        with self.assertRaises(IndexError):
            check_for_2x2_matrix(vector)


class TestDet(unittest.TestCase):
    def test_returns_correct_determinant_for_2x2_matrix(self):
        matrix = np.array([[1, 2], [3, 4]])
        expected_det = 1 * 4 - 2 * 3  # 4 - 6 = -2
        self.assertEqual(det(matrix), expected_det)

    def test_returns_correct_determinant_for_identity_matrix(self):
        matrix = np.array([[1, 0], [0, 1]])
        expected_det = 1 * 1 - 0 * 0  # 1 - 0 = 1
        self.assertEqual(det(matrix), expected_det)

    def test_returns_correct_determinant_for_zero_matrix(self):
        matrix = np.array([[0, 0], [0, 0]])
        expected_det = 0 * 0 - 0 * 0  # 0 - 0 = 0
        self.assertEqual(det(matrix), expected_det)

    def test_returns_none_for_non_2x2_matrix(self):
        matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.assertIsNone(det(matrix))

    def test_returns_none_for_rectangular_matrix(self):
        matrix = np.array([[1, 2, 3], [4, 5, 6]])
        self.assertIsNone(det(matrix))

    def test_raises_indexerror_for_one_dimensional_input(self):
        vector = np.array([1, 2, 3])
        with self.assertRaises(IndexError):
            det(vector)


class TestCheckForInvertibility(unittest.TestCase):
    def test_returns_true_for_invertible_2x2_matrix(self):
        matrix = np.array([[1, 2], [3, 4]])
        self.assertTrue(check_for_invertibility(matrix))

    def test_returns_false_for_non_invertible_2x2_matrix(self):
        matrix = np.array([[1, 2], [2, 4]])  # det = 1*4 - 2*2 = 4-4 = 0
        self.assertFalse(check_for_invertibility(matrix))

    def test_returns_false_for_zero_matrix(self):
        matrix = np.array([[0, 0], [0, 0]])  # det = 0
        self.assertFalse(check_for_invertibility(matrix))

    def test_returns_none_for_non_2x2_matrix(self):
        matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.assertIsNone(check_for_invertibility(matrix))

    def test_returns_none_for_rectangular_matrix(self):
        matrix = np.array([[1, 2, 3], [4, 5, 6]])
        self.assertIsNone(check_for_invertibility(matrix))

    def test_raises_indexerror_for_one_dimensional_input(self):
        vector = np.array([1, 2, 3])
        with self.assertRaises(IndexError):
            check_for_invertibility(vector)


class TestInv(unittest.TestCase):
    def test_returns_correct_inverse_for_2x2_matrix(self):
        matrix = np.array([[1, 2], [3, 4]])
        expected_inverse = np.array([[-2.0, 1.0], [1.5, -0.5]])
        result = inv(matrix)
        np.testing.assert_array_almost_equal(result, expected_inverse)

    def test_returns_correct_inverse_for_identity_matrix(self):
        matrix = np.array([[1, 0], [0, 1]])
        expected_inverse = np.array([[1, 0], [0, 1]])
        result = inv(matrix)
        np.testing.assert_array_almost_equal(result, expected_inverse)

    def test_returns_correct_inverse_for_simple_matrix(self):
        matrix = np.array([[2, 0], [0, 3]])
        expected_inverse = np.array([[0.5, 0], [0, 1/3]])
        result = inv(matrix)
        np.testing.assert_array_almost_equal(result, expected_inverse)

    def test_raises_valueerror_for_singular_matrix(self):
        matrix = np.array([[1, 2], [2, 4]])  # det = 1*4 - 2*2 = 0
        with self.assertRaises(ValueError):
            inv(matrix)

    def test_raises_valueerror_for_zero_matrix(self):
        matrix = np.array([[0, 0], [0, 0]])  # det = 0
        with self.assertRaises(ValueError):
            inv(matrix)

    def test_returns_none_for_non_2x2_matrix(self):
        matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.assertIsNone(inv(matrix))

    def test_returns_none_for_rectangular_matrix(self):
        matrix = np.array([[1, 2, 3], [4, 5, 6]])
        self.assertIsNone(inv(matrix))

    def test_returns_none_for_1x1_matrix(self):
        matrix = np.array([[5]])
        self.assertIsNone(inv(matrix))

    def test_returns_none_for_empty_matrix(self):
        matrix = np.empty((0, 0))
        self.assertIsNone(inv(matrix))

    def test_preserves_data_type(self):
        matrix = np.array([[1, 2], [3, 4]], dtype=np.float32)
        result = inv(matrix)
        self.assertEqual(result.dtype, np.float32)

    def test_raises_indexerror_for_one_dimensional_input(self):
        vector = np.array([1, 2, 3])
        with self.assertRaises(IndexError):
            inv(vector)

    def test_raises_attributeerror_for_non_array_like_without_shape(self):
        not_array_like = [[1, 2], [3, 4]]
        with self.assertRaises(AttributeError):
            inv(not_array_like)

    def test_inverse_multiplication_identity(self):
        matrix = np.array([[1, 2], [3, 4]])
        inverse = inv(matrix)
        identity = np.eye(2)
        result = np.dot(matrix, inverse)
        np.testing.assert_array_almost_equal(result, identity)

    def test_inverse_multiplication_commutative(self):
        matrix = np.array([[1, 2], [3, 4]])
        inverse = inv(matrix)
        identity = np.eye(2)
        result = np.dot(inverse, matrix)
        np.testing.assert_array_almost_equal(result, identity)


class TestTranspose(unittest.TestCase):
    def test_returns_correct_transpose_for_2x2_matrix(self):
        matrix = np.array([[1, 2], [3, 4]])
        expected_transpose = np.array([[1, 3], [2, 4]])
        result = transpose(matrix)
        np.testing.assert_array_equal(result, expected_transpose)

    def test_returns_correct_transpose_for_2x3_matrix(self):
        matrix = np.array([[1, 2, 3], [4, 5, 6]])
        expected_transpose = np.array([[1, 4], [2, 5], [3, 6]])
        result = transpose(matrix)
        np.testing.assert_array_equal(result, expected_transpose)

    def test_returns_correct_transpose_for_3x2_matrix(self):
        matrix = np.array([[1, 2], [3, 4], [5, 6]])
        expected_transpose = np.array([[1, 3, 5], [2, 4, 6]])
        result = transpose(matrix)
        np.testing.assert_array_equal(result, expected_transpose)

    def test_returns_correct_transpose_for_1x3_matrix(self):
        matrix = np.array([[1, 2, 3]])
        expected_transpose = np.array([[1], [2], [3]])
        result = transpose(matrix)
        np.testing.assert_array_equal(result, expected_transpose)

    def test_returns_correct_transpose_for_3x1_matrix(self):
        matrix = np.array([[1], [2], [3]])
        expected_transpose = np.array([[1, 2, 3]])
        result = transpose(matrix)
        np.testing.assert_array_equal(result, expected_transpose)

    def test_returns_correct_transpose_for_identity_matrix(self):
        matrix = np.array([[1, 0], [0, 1]])
        expected_transpose = np.array([[1, 0], [0, 1]])
        result = transpose(matrix)
        np.testing.assert_array_equal(result, expected_transpose)

    def test_returns_correct_transpose_for_zero_matrix(self):
        matrix = np.array([[0, 0], [0, 0]])
        expected_transpose = np.array([[0, 0], [0, 0]])
        result = transpose(matrix)
        np.testing.assert_array_equal(result, expected_transpose)

    def test_preserves_data_type(self):
        matrix = np.array([[1, 2], [3, 4]], dtype=np.float32)
        result = transpose(matrix)
        self.assertEqual(result.dtype, np.float32)

    def test_preserves_data_type_for_int_matrix(self):
        matrix = np.array([[1, 2], [3, 4]], dtype=np.int32)
        result = transpose(matrix)
        self.assertEqual(result.dtype, np.int32)

    def test_raises_indexerror_for_one_dimensional_input(self):
        vector = np.array([1, 2, 3])
        with self.assertRaises(IndexError):
            transpose(vector)

    def test_raises_indexerror_for_three_dimensional_input(self):
        tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        with self.assertRaises(IndexError):
            transpose(tensor)

    def test_raises_attributeerror_for_non_array_like_without_shape(self):
        not_array_like = [[1, 2], [3, 4]]
        with self.assertRaises(AttributeError):
            transpose(not_array_like)

    def test_raises_attributeerror_for_none_input(self):
        with self.assertRaises(AttributeError):
            transpose(None)

    def test_raises_attributeerror_for_string_input(self):
        with self.assertRaises(AttributeError):
            transpose("matrix")

    def test_transpose_twice_returns_original(self):
        matrix = np.array([[1, 2, 3], [4, 5, 6]])
        transposed_once = transpose(matrix)
        transposed_twice = transpose(transposed_once)
        np.testing.assert_array_equal(transposed_twice, matrix)

    def test_transpose_of_transpose_equals_original(self):
        matrix = np.array([[1, 2], [3, 4], [5, 6]])
        transposed = transpose(matrix)
        original = transpose(transposed)
        np.testing.assert_array_equal(original, matrix)


if __name__ == "__main__":
    unittest.main()
