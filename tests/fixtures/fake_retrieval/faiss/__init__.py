"""A stand-in for faiss: an index class with its own add, train and search, as faiss's numpy
wrappers are set on each class at import."""

import numpy as np


class Index:
	pass


class IndexFlatIP(Index):
	def __init__(self, d):
		self.d, self.xb = d, np.zeros((0, d), dtype="float32")

	def train(self, x):
		pass

	def add(self, x):
		self.xb = np.concatenate([self.xb, x])

	def search(self, x, k):
		s = x @ self.xb.T
		i = np.argsort(-s, axis=1)[:, :k]
		return np.take_along_axis(s, i, 1), i
