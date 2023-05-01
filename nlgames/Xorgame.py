import numpy as np
import cvxpy
from collections import defaultdict
from ortools.init import pywrapinit
from ortools.linear_solver import pywraplp

class Xorgame:
	def __init__(self, predMatrix: np.ndarray, probMatrix: np.ndarray):
		self.probMatrix = probMatrix
		self.predMatrix = predMatrix


		#Catching errors
		if (self.probMatrix.shape != self.predMatrix.shape):
			raise TypeError("Probability and predicate matrix must have the same dimensions.")
		if (np.sum(self.probMatrix) != 1):
			raise ValueError("The probabilities must sum up to one.")
		if (np.any(probMatrix<0)):
			raise ValueError("The probability matrix cannot contain negative values.")


	def cvalue(self, reps: int) -> float:
		maxval = 0.0
		augmented_prob = self.probMatrix

		# Create larger probability matrix in case of repetition
		for i in range(reps - 1):
			augmented_prob = np.kron(augmented_prob, self.probMatrix)

		q_0_original, q_1_original = self.probMatrix.shape
		q_0, q_1 = augmented_prob.shape
		augmented_pred = np.ndarray((reps, q_0, q_1))

		# Create multi-layer predicate matrix in case of parallel repetition
		for i in range(reps):
			for j in range(q_0):
				for k in range(q_1):
					augmented_pred[i, j, k] = self.predMatrix[(j >> i) % 2, (k >> i) % 2]

		# print(augmented_pred)


		# Iterate through all strategies
		for a_ans in range(2**(q_0*reps)):
			for b_ans in range(2**(q_1*reps)):
				val = 0

				# Generate full strategy vectors. Index represents the question, value represents the answer.
				# This is done slightly differently from the toqito version, because I find this one more legible and easier to understand.
				# 
				# Each row represents one answer to a set of questions, with the column index representing the question in binary,
				# e.g. a_stategy[0,0] is the answer to the first question if all questions are 0.
				a_strategy = np.array([1 if a_ans & (1 << ((q_0 * reps) - 1 - n)) else 0 for n in range (q_0 * reps)]).reshape(reps, q_0)
				b_strategy = np.array([1 if b_ans & (1 << ((q_1 * reps) - 1 - n)) else 0 for n in range (q_1 * reps)]).reshape(reps, q_1)


				# XOR games don't really need both strategies separately, but only the XOR of both,
				# so we can create a matrix that represents all the info we need.
				a_matrix = np.ndarray((reps, q_0, q_0))
				b_matrix = np.ndarray((reps, q_1, q_1))
				combined_matrix = np.ndarray((reps, q_0, q_1))
				for i in range(reps):
					
					a_matrix[i] = np.multiply(a_strategy[i].T.reshape(-1,1), np.ones((1,q_0)))
					b_matrix[i] = np.multiply(b_strategy[i].T.reshape(-1,1), np.ones((1, q_1))).T
					
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

	def __singlebias(self) -> float:
		value = 0.0
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

	def cbias(self, reps: int) -> float:
		return 2 * self.cvalue(reps) - 1

	def qvalue(self, reps: int) -> float:
		# toqito divides the bias by 4 for some reason. This is how it's done in the Watrous lecture notes.
		bias = self.__singlebias()
		value = 1/2 + bias/2
		return value**reps
	
	def qbias(self, reps: int) -> float:
		value = self.qvalue(reps)
		return value - (1-value)
	
	def nsvalue_single(self) -> float:

		# Some of this function is hardcoded for 2x2 matrices.
		if self.probMatrix.shape[0] > 2:
			raise ValueError("This function only supports matrices of the size 2x2.")

		# Initialise solver
		solver = pywraplp.Solver.CreateSolver('GLOP')

		# Initialise probabilities p(a,b|x,y)
		prob_space = defaultdict(pywraplp.Variable)
		for a in range(2):
			for b in range(2):
				for s in range(2):
					for t in range(2):
						prob_space[a, b, s, t] = solver.NumVar(0.0, 1.0, 'prob_space[%i,%i,%i,%i]' % (a,b,s,t))



		# Conditions over \sum_b
		# Because this only works for 2x2 matrices, it is possible to define all these constraints manually.
		solver.Add(prob_space[0,0,0,0] + prob_space[0,1,0,0] == prob_space[0,0,0,1] + prob_space[0,1,0,1])
		solver.Add(prob_space[0,0,1,0] + prob_space[0,1,1,0] == prob_space[0,0,1,1] + prob_space[0,1,1,1])
		solver.Add(prob_space[1,0,0,0] + prob_space[1,1,0,0] == prob_space[1,0,0,1] + prob_space[1,1,0,1])
		solver.Add(prob_space[1,0,1,0] + prob_space[1,1,1,0] == prob_space[1,0,1,1] + prob_space[1,1,1,1])

		# Conditions over \sum_a
		solver.Add(prob_space[0,0,0,0] + prob_space[1,0,0,0] == prob_space[0,0,1,0] + prob_space[1,0,1,0])
		solver.Add(prob_space[0,0,0,1] + prob_space[1,0,0,1] == prob_space[0,0,1,1] + prob_space[1,0,1,1])
		solver.Add(prob_space[0,1,0,0] + prob_space[1,1,0,0] == prob_space[0,1,1,0] + prob_space[1,1,1,0])
		solver.Add(prob_space[0,1,0,1] + prob_space[1,1,0,1] == prob_space[0,1,1,1] + prob_space[1,1,1,1])

		# No-signaling value, which is to be maximised
		val = solver.NumVar(0.0, 1.0, 'val')
		solver.Maximize(val)


		# Iterate through all possibilities and determine how much is added to the total win value
		for a in range(2):
			for b in range(2):
				for s in range(2):
					for t in range(2):
						# Decide whether the combination of a,b,s,t wins the game
						win = 1 if self.predMatrix[s,t] == a ^ b else 0

						# If the combination wins, the probability that the questions occur gets multiplied with the value from the probability space.
						val += self.probMatrix[s,t] * (win * prob_space[a,b,s,t])

		# Solve the linear program
		status = solver.Solve()
		if status == pywraplp.Solver.OPTIMAL: 
			return solver.Objective().Value()
		else:
			raise ValueError("There may not be an optimal solution for this problem.")
	def nsbias_single(self) -> float:
		value = self.nsvalue_single(reps)
		return value - (1-value)
		
	def nsval_rep_upper_bound(self, reps) -> float:
		return (1-(((1 - self.nsvalue_single())**2) / (6400)))**reps
	
	def to_nonlocal_game(self) -> np.ndarray:
		
		q_0, q_1 = self.probMatrix.shape
		pred_mat = self.predMatrix
		result = np.ndarray((2,2,q_0,q_1))

		for a in range(2):
			for b in range(2):
				for s in range(q_0):
					for t in range(q_1):
						result[a,b,s,t] = pred_mat[s,t] == a ^ b

		return result
