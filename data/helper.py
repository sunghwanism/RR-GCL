import numpy as np


class CKACalculator:
    def __init__(self, eps=1e-8):
        self.eps = eps

    def normalize_rows(self, X):
        mean = np.mean(X, axis=1, keepdims=True)
        std = np.std(X, axis=1, keepdims=True)
        return (X - mean) / (std + self.eps)

    def _centering(self, K):
        n = K.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        return H @ K @ H

    def _hsic(self, K, L):
        K_c = self._centering(K)
        L_c = self._centering(L)
        return np.sum(K_c * L_c)

    def score(self, X, Y, perform_normalization=True):
        if perform_normalization:
            X = self.normalize_rows(X)
            Y = self.normalize_rows(Y)

        K = X @ X.T
        L = Y @ Y.T

        hsic_kl = self._hsic(K, L)
        hsic_kk = self._hsic(K, K)
        hsic_ll = self._hsic(L, L)

        return hsic_kl / np.sqrt(hsic_kk * hsic_ll + self.eps)

def RVcoefficient(X, Y):

    X = X - np.mean(X, axis=0)
    Y = Y - np.mean(Y, axis=0)
    
    S_X = np.dot(X, X.T)
    S_Y = np.dot(Y, Y.T)
    
    numerator = np.trace(np.dot(S_X, S_Y))
    
    denominator = np.sqrt(np.trace(np.dot(S_X, S_X)) * np.trace(np.dot(S_Y, S_Y)))
    
    return numerator / denominator

