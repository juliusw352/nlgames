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
		val = 0

		return val

q_0 = 2
q_1 = 2

for a_ans in range(2**q_0):
	for b_ans in range(2**q_1):
		print(a_ans)
		print(b_ans)
		#! in toqito: Index is question, value is answer!
		a_vec = a_ans >> np.arange(q_0) & 1
		a_vec_tmp = a_ans >> np.arange(q_0)
		b_vec = b_ans >> np.arange(q_1) & 1

		print(a_vec)
		print(a_vec_tmp)