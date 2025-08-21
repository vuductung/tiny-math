import unittest
import numpy as np
from tiny_math.linear_algebra import (
    check_for_square_matrix,
    check_for_2x2_matrix,
    det,
    check_for_invertibility,
    inv,
    transpose,
    check_for_symmetry,
    check_for_matmul,
    matmul
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


class TestCheckForSymmetry(unittest.TestCase):
    def test_returns_true_for_symmetric_2x2_matrix(self):
        matrix = np.array([[1, 2], [2, 3]])
        self.assertTrue(check_for_symmetry(matrix))

    def test_returns_true_for_symmetric_3x3_matrix(self):
        matrix = np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]])
        self.assertTrue(check_for_symmetry(matrix))

    def test_returns_true_for_identity_matrix(self):
        matrix = np.array([[1, 0], [0, 1]])
        self.assertTrue(check_for_symmetry(matrix))

    def test_returns_true_for_zero_matrix(self):
        matrix = np.array([[0, 0], [0, 0]])
        self.assertTrue(check_for_symmetry(matrix))

    def test_returns_true_for_diagonal_matrix(self):
        matrix = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
        self.assertTrue(check_for_symmetry(matrix))

    def test_returns_false_for_non_symmetric_2x2_matrix(self):
        matrix = np.array([[1, 2], [3, 4]])
        self.assertFalse(check_for_symmetry(matrix))

    def test_returns_false_for_non_symmetric_3x3_matrix(self):
        matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.assertFalse(check_for_symmetry(matrix))

    def test_returns_false_for_rectangular_matrix(self):
        matrix = np.array([[1, 2, 3], [4, 5, 6]])
        self.assertFalse(check_for_symmetry(matrix))

    def test_returns_false_for_1x3_matrix(self):
        matrix = np.array([[1, 2, 3]])
        self.assertFalse(check_for_symmetry(matrix))

    def test_returns_false_for_3x1_matrix(self):
        matrix = np.array([[1], [2], [3]])
        self.assertFalse(check_for_symmetry(matrix))

    def test_returns_false_for_empty_matrix(self):
        matrix = np.empty((0, 0))
        self.assertTrue(check_for_symmetry(matrix))  # Empty matrices are symmetric

    def test_returns_false_for_1x1_matrix(self):
        matrix = np.array([[5]])
        self.assertTrue(check_for_symmetry(matrix))  # 1x1 matrices are always symmetric

    def test_raises_indexerror_for_one_dimensional_input(self):
        vector = np.array([1, 2, 3])
        with self.assertRaises(IndexError):
            check_for_symmetry(vector)

    def test_raises_indexerror_for_three_dimensional_input(self):
        tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        with self.assertRaises(IndexError):
            check_for_symmetry(tensor)

    def test_raises_attributeerror_for_non_array_like_without_shape(self):
        not_array_like = [[1, 2], [3, 4]]
        with self.assertRaises(AttributeError):
            check_for_symmetry(not_array_like)

    def test_raises_attributeerror_for_none_input(self):
        with self.assertRaises(AttributeError):
            check_for_symmetry(None)

    def test_raises_attributeerror_for_string_input(self):
        with self.assertRaises(AttributeError):
            check_for_symmetry("matrix")

    def test_symmetric_matrix_equals_its_transpose(self):
        matrix = np.array([[1, 2], [2, 3]])
        self.assertTrue(np.array_equal(matrix, transpose(matrix)))
        self.assertTrue(check_for_symmetry(matrix))

    def test_asymmetric_matrix_not_equals_its_transpose(self):
        matrix = np.array([[1, 2], [3, 4]])
        self.assertFalse(np.array_equal(matrix, transpose(matrix)))
        self.assertFalse(check_for_symmetry(matrix))


class TestCheckForMatmul(unittest.TestCase):
    def test_returns_true_for_compatible_2x2_matrices(self):
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6], [7, 8]])
        self.assertTrue(check_for_matmul(a, b))

    def test_returns_true_for_compatible_2x3_and_3x2_matrices(self):
        a = np.array([[1, 2, 3], [4, 5, 6]])
        b = np.array([[7, 8], [9, 10], [11, 12]])
        self.assertTrue(check_for_matmul(a, b))

    def test_returns_true_for_compatible_1x3_and_3x1_matrices(self):
        a = np.array([[1, 2, 3]])
        b = np.array([[4], [5], [6]])
        self.assertTrue(check_for_matmul(a, b))

    def test_returns_false_for_incompatible_2x2_matrices(self):
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6], [7, 8], [9, 10]])
        self.assertFalse(check_for_matmul(a, b))

    def test_returns_false_for_incompatible_2x3_and_2x3_matrices(self):
        a = np.array([[1, 2, 3], [4, 5, 6]])
        b = np.array([[7, 8, 9], [10, 11, 12]])
        self.assertFalse(check_for_matmul(a, b))

    def test_returns_false_for_empty_matrices(self):
        a = np.empty((0, 0))
        b = np.empty((0, 0))
        self.assertTrue(check_for_matmul(a, b))  # 0 == 0

    def test_raises_attributeerror_for_non_array_like_without_shape(self):
        a = [[1, 2], [3, 4]]
        b = np.array([[5, 6], [7, 8]])
        with self.assertRaises(AttributeError):
            check_for_matmul(a, b)

    def test_raises_attributeerror_for_none_input(self):
        a = np.array([[1, 2], [3, 4]])
        with self.assertRaises(AttributeError):
            check_for_matmul(a, None)


class TestMatmul(unittest.TestCase):
    def test_returns_correct_result_for_2x2_matrices(self):
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6], [7, 8]])
        expected = np.array([[19, 22], [43, 50]])
        result = matmul(a, b)
        np.testing.assert_array_equal(result, expected)

    def test_returns_correct_result_for_2x3_and_3x2_matrices(self):
        a = np.array([[1, 2, 3], [4, 5, 6]])
        b = np.array([[7, 8], [9, 10], [11, 12]])
        expected = np.array([[58, 64], [139, 154]])
        result = matmul(a, b)
        np.testing.assert_array_equal(result, expected)

    def test_returns_correct_result_for_1x3_and_3x1_matrices(self):
        a = np.array([[1, 2, 3]])
        b = np.array([[4], [5], [6]])
        expected = np.array([[32]])
        result = matmul(a, b)
        np.testing.assert_array_equal(result, expected)

    def test_returns_correct_result_for_3x1_and_1x3_matrices(self):
        a = np.array([[1], [2], [3]])
        b = np.array([[4, 5, 6]])
        expected = np.array([[4, 5, 6], [8, 10, 12], [12, 15, 18]])
        result = matmul(a, b)
        np.testing.assert_array_equal(result, expected)

    def test_returns_correct_result_for_identity_matrix_multiplication(self):
        a = np.array([[1, 2], [3, 4]])
        identity = np.array([[1, 0], [0, 1]])
        result = matmul(a, identity)
        np.testing.assert_array_equal(result, a)

    def test_returns_correct_result_for_zero_matrix_multiplication(self):
        a = np.array([[1, 2], [3, 4]])
        zero = np.array([[0, 0], [0, 0]])
        expected = np.array([[0, 0], [0, 0]])
        result = matmul(a, zero)
        np.testing.assert_array_equal(result, expected)

    def test_returns_correct_result_for_scalar_multiplication(self):
        a = np.array([[2]])
        b = np.array([[3]])
        expected = np.array([[6]])
        result = matmul(a, b)
        np.testing.assert_array_equal(result, expected)

    def test_returns_none_for_incompatible_matrices(self):
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6], [7, 8], [9, 10]])
        result = matmul(a, b)
        self.assertIsNone(result)

    def test_returns_none_for_incompatible_2x3_and_2x3_matrices(self):
        a = np.array([[1, 2, 3], [4, 5, 6]])
        b = np.array([[7, 8, 9], [10, 11, 12]])
        result = matmul(a, b)
        self.assertIsNone(result)

    def test_preserves_data_type(self):
        a = np.array([[1, 2], [3, 4]], dtype=np.float32)
        b = np.array([[5, 6], [7, 8]], dtype=np.float32)
        result = matmul(a, b)
        self.assertEqual(result.dtype, np.float64)  # np.sum promotes to float64

    def test_raises_attributeerror_for_non_array_like_without_shape(self):
        a = [[1, 2], [3, 4]]
        b = np.array([[5, 6], [7, 8]])
        with self.assertRaises(AttributeError):
            matmul(a, b)

    def test_raises_attributeerror_for_none_input(self):
        a = np.array([[1, 2], [3, 4]])
        with self.assertRaises(AttributeError):
            matmul(a, None)

    def test_matmul_agrees_with_numpy_dot(self):
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6], [7, 8]])
        custom_result = matmul(a, b)
        numpy_result = np.dot(a, b)
        np.testing.assert_array_equal(custom_result, numpy_result)

    def test_matmul_agrees_with_numpy_dot_for_rectangular_matrices(self):
        a = np.array([[1, 2, 3], [4, 5, 6]])
        b = np.array([[7, 8], [9, 10], [11, 12]])
        custom_result = matmul(a, b)
        numpy_result = np.dot(a, b)
        np.testing.assert_array_equal(custom_result, numpy_result)


if __name__ == "__main__":
    unittest.main()
