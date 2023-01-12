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

		q_0, q_1 = self.probMatrix.shape
		reps = self.reps
		q_0_repeats, q_1_repeats = q_0 * reps, q_1 * reps

		# Iterate through all strategies
		for a_ans in range(2**q_0_repeats):
			for b_ans in range(2**q_1_repeats):
				val = 0

				# Generate full strategy vectors. Index represents the question, value represents the answer.
				# This is done slightly differently from the toqito version, because I find this one more legible.
				# Strategy vectors are converted into 2d matrices, with each row representing a separate repetition
				a_strategy = np.array([1 if a_ans & (1 << (q_0_repeats - 1 - n)) else 0 for n in range (q_0_repeats)]).reshape((reps, q_0))
				b_strategy = np.array([1 if b_ans & (1 << (q_1_repeats - 1 - n)) else 0 for n in range (q_1_repeats)]).reshape((reps, q_1))
				# Initialising a bunch of matrices that will be populated in the for loop
				a_matrix = np.ndarray((reps, q_0, q_0))
				b_matrix = np.ndarray((reps, q_1, q_1))
				combined_matrix = np.ndarray((reps, q_0, q_0))
				result_matrix = np.ndarray((reps, q_0, q_0))
				
				for repetition in range(reps):
					a_matrix[repetition] = np.multiply(a_strategy[repetition].T.reshape(-1,1), np.ones((1,q_0)))
					b_matrix[repetition] = np.multiply(b_strategy[repetition].T.reshape(-1,1), np.ones((1, q_1)))
					#b_matrix[repetition] = b_matrix[repetition].T # Transpose so that one "direction" is s, and the other is t
					# XOR games don't really need both strategies separately, but only the XOR of both,
					# so we can create a matrix that represents all the info we need. 
					combined_matrix[repetition] = np.mod(a_matrix[repetition] + b_matrix[repetition].T, 2)
					result_matrix[repetition] = combined_matrix[repetition] == self.predMatrix

				condensed_matrix = np.ndarray((q_0, q_0))
				for i in range(condensed_matrix.shape[0]):
					for j in range(condensed_matrix.shape[1]):
						condensed_matrix[i,j] = np.logical_and.reduce(result_matrix[:,i,j])
				print(condensed_matrix)

				# Factor in the probabilities of each combination occuring, and sum up all the results
				val = np.sum(np.multiply(condensed_matrix, self.probMatrix))
				
				

				if val == 1: return val # check for perfect strategy
				maxval = val if val > maxval else maxval # check if current strategy is better than others encountered
		return maxval
	
	def qvalue(self):
		maxvalue = 0
		q_0, q_1 = self.probMatrix.shape

		return maxvalue


prob = np.array([[0.25, 0.25],[0.25, 0.25]])
pred = np.array([[0, 0],[0, 1]])
chsh = xorgame(pred, prob, 2)

print(chsh.cvalue())

"""q_0 = 2
reps = 2
length = q_0 * reps
for a_ans in range(length**2):

	a_strategy = np.array([1 if a_ans & (1 << (length - 1 - n)) else 0 for n in range (length)]).reshape((q_0, reps))
	a_matrix = np.zeros((reps, q_0, q_0))
	for repetition in range(reps):
		a_matrix[repetition] = np.multiply(a_strategy[repetition].T.reshape(-1,1), np.ones((1,q_0)))

	print("Matrix: ")
	print(a_matrix)
	print("condensed:")
	for i in range(q_0):
		for j in range(q_0):
			print(np.multiply.reduce(a_matrix[:,i,j]))"""