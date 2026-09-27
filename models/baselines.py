"""
Baseline models for ShadowLoc comparison.

B1 — Supervised GNSS-only Random Forest classifier
B2 — GNSS-only LSTM autoencoder (same architecture as Variant B
     but trained only on GNSS features)
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from models.variant_b import VariantB


class Baseline1:
    """B1: Supervised GNSS-only Random Forest classifier.

    Trained on labeled windows (benign=0, spoofed=1) using only
    GNSS signal features. This is a genuine supervised classifier
    baseline, clearly distinct from the unsupervised anomaly-detection
    approach of Variant A (EllipticEnvelope).
    """

    def __init__(self, random_state=42):
        self.model = RandomForestClassifier(
            n_estimators=100, random_state=random_state,
            class_weight='balanced', n_jobs=-1
        )

    def _flatten(self, X):
        N, W, D = X.shape
        return X.reshape(N, W * D)

    def fit(self, X_train, y_train):
        """Train on labeled data. X_train: (N, W, D), y_train: (N,) bool."""
        self.model.fit(self._flatten(X_train), y_train)

    def predict(self, X):
        return self.model.predict(self._flatten(X))

    def predict_proba(self, X):
        """Return spoofing probability (class 1)."""
        return self.model.predict_proba(self._flatten(X))[:, 1]

    def score(self, X):
        """Anomaly score = probability of class 1 (spoofed)."""
        return self.predict_proba(X)


class Baseline2:
    """B2: GNSS-only LSTM autoencoder.

    Same architecture as Variant B but trained only on GNSS features
    (d_pos, d_speed, d_bearing, a_gnss, mean_cn0, std_cn0, n_sat,
    agc, agc_slope).
    """

    def __init__(self, input_dim, **kwargs):
        self.model = VariantB(input_dim, **kwargs)

    def fit(self, X_benign):
        self.model.fit(X_benign)

    def score(self, X):
        return self.model.score(X)
