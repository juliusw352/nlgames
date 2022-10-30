import numpy as np
import cvxpy

class xorgame:
	def __init__(self, probMatrix: np.ndarray, predMatrix: np.ndarray):
		self.probMatrix = probMatrix
		self.predMatrix = predMatrix

		#Catching errors
		if (probMatrix.shape != predMatrix.shape):
			raise TypeError("Probability and predicate matrix must have the same dimensions.")
		if (np.sum(probMatrix) != 1):
			raise ValueError("The probabilities must sum up to one.")