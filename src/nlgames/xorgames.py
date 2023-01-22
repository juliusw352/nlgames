import numpy as np
import cvxpy

class xorgame:
	def __init__(self, predMatrix: np.ndarray, probMatrix: np.ndarray, reps: int):
		self.probMatrix = probMatrix
		self.predMatrix = predMatrix
		self.reps = reps

		#Catching errors
		if (self.probMatrix.shape != self.predMatrix.shape):
			raise TypeError("Probability and predicate matrix must have the same dimensions.")
		if (np.sum(self.probMatrix) != 1):
			raise ValueError("The probabilities must sum up to one.")


	def cvalue(self):
		maxval = 0
		reps = self.reps
		augmented_prob = self.probMatrix

		for i in range(reps - 1):
			augmented_prob = np.kron(augmented_prob, self.probMatrix)

		q_0_original, q_1_original = self.probMatrix.shape
		q_0, q_1 = augmented_prob.shape
		augmented_pred = np.ndarray((reps, q_0, q_1))

		for i in range(reps):
			mask = np.ndarray((q_0, q_1))
			mask2 = np.ndarray((q_0, q_1))
			for j in range(q_0):
				for k in range(q_1):
					augmented_pred[i, j, k] = self.predMatrix[(j >> i) % 2, (k >> i) % 2]


		# Iterate through all strategies
		for a_ans in range(2**q_0):
			for b_ans in range(2**q_1):
				val = 0

				# Generate full strategy vectors. Index represents the question, value represents the answer.
				# This is done slightly differently from the toqito version, because I find this one more legible.
				a_strategy = np.array([1 if a_ans & (1 << ((q_0_original * reps) - 1 - n)) else 0 for n in range (q_0_original * reps)]).reshape(reps, q_0_original)
				b_strategy = np.array([1 if b_ans & (1 << ((q_1_original * reps) - 1 - n)) else 0 for n in range (q_1_original * reps)]).reshape(reps, q_1_original)

				a_full_strategy = np.ndarray((reps, q_0))
				b_full_strategy = np.ndarray((reps, q_1))

				for i in range(reps):
					l = 2**i
					pattern = np.append(np.repeat(0, l), np.repeat(1, l))
					mask = np.tile(pattern, int(q_0 / (2*l)))
					for j in range(q_0):
						a_full_strategy[i, j] = a_strategy[i, mask[j]]
						b_full_strategy[i, j] = b_strategy[i, mask[j]]
				# XOR games don't really need both strategies separately, but only the XOR of both,
				# so we can create a matrix that represents all the info we need.
				a_matrix = np.ndarray((reps, q_0, q_0))
				b_matrix = np.ndarray((reps, q_1, q_1))
				combined_matrix = np.ndarray((reps, q_0, q_0))
				for i in range(reps):
					
					a_matrix[i] = np.multiply(a_full_strategy[i].T.reshape(-1,1), np.ones((1,q_0)))
					b_matrix[i] = np.multiply(b_full_strategy[i].T.reshape(-1,1), np.ones((1, q_1))).T
					
					combined_matrix[i] = np.mod(a_matrix[i] + b_matrix[i], 2)
				success_matrix = combined_matrix == augmented_pred
				reduced_matrix = np.multiply.reduce(success_matrix, 0)

				# Factor in the probabilities of each combination occuring, and sum up all the results
				final_matrix = np.multiply(reduced_matrix, augmented_prob)
				val = np.sum(final_matrix)


				if val == 1: return val # check for perfect strategy
				if val > maxval:
					maxval = val
		return maxval
	
	def qvalue(self):
		maxvalue = 0
		q_0, q_1 = self.probMatrix.shape

		return maxvalue


prob = np.array([[0.25, 0.25],[0.25, 0.25]])
pred = np.array([[0, 0],[0, 1]])
chsh = xorgame(pred, prob, 3)

print(chsh.cvalue())
