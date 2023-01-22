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
	
	def qbias(self):
		value = 0
		q_0, q_1 = self.probMatrix.shape

		# Variables below correspond to the equivalently-named ones in the Watrous lecture notes.
		# https://cs.uwaterloo.ca/~watrous/QIT-notes/QIT-notes.06.pdf
		# https://cs.uwaterloo.ca/~watrous/QIT-notes/QIT-notes.07.pdf
		f = self.predMatrix 
		pi = self.probMatrix 
		d = np.ndarray((q_0,q_1))
		for i in range(q_0):
			for j in range(q_1):
				d[i,j] = pi[i,j] * ((-1)**f[i,j])


		u = cvxpy.Variable(q_0, complex=False) # Complex = false is equivalent to the 2nd & 3rd restrictions (?)
		v = cvxpy.Variable(q_1, complex=False)

		# To be minimised:
		# (toqito omits the 1/2 part, not sure why)
		objective = cvxpy.Minimize((1/2 * cvxpy.sum(u)) + (1/2 * cvxpy.sum(v)))

		constraints = [
			cvxpy.bmat(
				[[cvxpy.diag(u), -d], 
				[-(d.conj().T), cvxpy.diag(v)]]
			)
			>> 0 # This line states that the matrix is positive semidefinite.
		]

		sdp = cvxpy.Problem(objective, constraints)
		sdp.solve()

		bias = np.real(sdp.value)
		return bias

	def cbias(self):
		return 2 * self.cvalue() - 1

	def qvalue(self):
		# toqito divides the bias by 4 for some reason. This is how it's done in the Watrous lecture notes.
		bias = self.qbias()
		value = 1/2 + bias/2

		return value


# Demonstration type beat
prob = np.array([[0.25, 0.25],[0.25, 0.25]])
pred = np.array([[0, 0],[0, 1]])
chsh = xorgame(pred, prob, 3)


print("\n\n############### NONLOCAL GAME ANALYSIS DEMO ###############\n")
print("Example: CHSH game\n")
print("Classical value: " + str(chsh.cvalue()))
print("Classical bias: " + str(chsh.cbias()))
print("Quantum value: " + str(chsh.qvalue()))
print("Quantum bias: " + str(chsh.qbias()))
