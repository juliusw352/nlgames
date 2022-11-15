import numpy as np
import cvxpy

class xorgame:
	def __init__(self, probMatrix: np.ndarray, predMatrix: np.ndarray):
		self.probMatrix = probMatrix
		self.predMatrix = predMatrix

		#Catching errors
		if (self.probMatrix.shape != self.predMatrix.shape):
			raise TypeError("Probability and predicate matrix must have the same dimensions.")
		if (np.sum(self.probMatrix) != 1):
			raise ValueError("The probabilities must sum up to one.")


	# TODO: Figure out expansion beyond simple binary games
	def cvalue(self):
		maxval = 0

		q_0, q_1 = self.probMatrix.shape

		# Iterate through all strategies
		for a_ans in range(2**q_0):
			for b_ans in range(2**q_1):
				val = 0

				# Generate full strategy vectors. Index represents the question, value represents the answer.
				# This is done slightly differently from the toqito version, because I find this one more legible.
				a_strategy = np.array([1 if a_ans & (1 << (q_0 - 1 - n)) else 0 for n in range (q_0)])
				b_strategy = np.array([1 if b_ans & (1 << (q_1 - 1 - n)) else 0 for n in range (q_1)])

				# XOR games don't really need both strategies separately, but only the XOR of both,
				# so we can create a matrix that represents all the info we need.
				a_matrix = np.multiply(a_strategy.T.reshape(-1,1), np.ones((1,q_1)))
				b_matrix = np.multiply(b_strategy.T.reshape(-1,1), np.ones((1, q_0)))
				b_matrix = b_matrix.T
				combined_matrix = np.mod(a_matrix + b_matrix, 2)

				if val == 1: return val # check for perfect strategy
				maxval = val if val > maxval else maxval # check if current strategy is better than others encountered
		return maxval


q_0, q_1 = 2, 2
for a_ans in range(2**q_0):
	for b_ans in range(2**q_1):
		val = 0
		# Generate full strategy vectors. Index represents the question, value represents the answer.
		# This is done slightly differently from the toqito version, because I find this one more legible.
		a_strategy = np.array([1 if a_ans & (1 << (q_0 - 1 - n)) else 0 for n in range (q_0)])
		b_strategy = np.array([1 if b_ans & (1 << (q_1 - 1 - n)) else 0 for n in range (q_1)])
		# XOR games don't really need both strategies separately, but only the XOR of both,
		# so we can create a matrix that represents all the info we need.
		a_matrix = np.multiply(a_strategy.T.reshape(-1,1), np.ones((1,q_1)))
		b_matrix = np.multiply(b_strategy.T.reshape(-1,1), np.ones((1, q_0)))
		b_matrix = b_matrix.T

		print(np.mod(a_matrix + b_matrix, 2))