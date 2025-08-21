import numpy as np


def check_for_square_matrix(matrix):
	if len(matrix.shape) != 2:
		raise IndexError("Input must be a 2-dimensional array.")
	return matrix.shape[0] == matrix.shape[1]

def check_for_2x2_matrix(matrix):
	if check_for_square_matrix(matrix):
		return (matrix.shape[0] == 2) & (matrix.shape[1]==2) 

def det(matrix):
	if check_for_2x2_matrix(matrix):
		return matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]
	
def check_for_invertibility(matrix):
	if check_for_2x2_matrix(matrix):
		return det(matrix) != 0


