import numpy as np
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest

class VariantA:
    def __init__(self, method='mahalanobis', contamination=0.01, random_state=42):
        """
        method: 'mahalanobis' or 'isolation_forest'
        """
        self.method = method
        if self.method == 'mahalanobis':
            # robust covariance estimation
            self.model = EllipticEnvelope(contamination=contamination, random_state=random_state)
        elif self.method == 'isolation_forest':
            self.model = IsolationForest(contamination=contamination, random_state=random_state)
        else:
            raise ValueError(f"Unknown method {method}")
            
    def _flatten(self, X):
        """X is (N, W, D). Flatten to (N, W*D)"""
        N, W, D = X.shape
        return X.reshape(N, W * D)

    def fit(self, X_benign):
        """Fit on benign flattened windows."""
        X_flat = self._flatten(X_benign)
        self.model.fit(X_flat)

    def score(self, X):
        """
        Returns anomaly scores. Higher score = more anomalous.
        For EllipticEnvelope, we can use mahalanobis distance directly.
        For IsolationForest, score_samples returns negative anomaly score (lower is more anomalous), 
        so we negate it.
        """
        X_flat = self._flatten(X)
        if self.method == 'mahalanobis':
            # Returns Mahalanobis distances
            return self.model.mahalanobis(X_flat)
        elif self.method == 'isolation_forest':
            # score_samples returns opposite of anomaly score (closer to 0 is normal, more negative is anomalous)
            # We want higher score for anomalous
            return -self.model.score_samples(X_flat)
