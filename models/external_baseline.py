from sklearn.svm import OneClassSVM
import numpy as np

class BhattiBaseline:
    """
    Re-implementation of a simplified Bhatti et al. (2020) One-Class SVM based spoofing detector.
    This external baseline uses traditional GNSS signal features:
    - mean C/N0
    - std C/N0
    - AGC
    - GNSS Speed (Doppler proxy)
    Trained strictly on benign data to identify outliers (spoofing).
    """
    def __init__(self, nu=0.01, kernel='rbf', gamma='scale'):
        self.model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
        # Assuming features are provided as [N_windows, W, D]
        # We will extract specific features for this SVM.
        
    def _extract_bhatti_features(self, X):
        """
        Extract relevant GNSS features for Bhatti SVM.
        Assuming X is standardized [N, W, D].
        We take the mean over the window (W) for:
        - d_speed (idx 1)
        - mean_cn0 (idx 4)
        - std_cn0 (idx 5)
        - agc (idx 7)
        """
        # Take mean over W dimension for each window
        X_mean = np.mean(X, axis=1)
        
        # Select features: d_speed, mean_cn0, std_cn0, agc
        # Note: Indexing must match the FeatureExtractor output!
        # 1: d_speed, 4: mean_cn0, 5: std_cn0, 7: agc
        return X_mean[:, [1, 4, 5, 7]]
        
    def fit(self, X_benign):
        X_train = self._extract_bhatti_features(X_benign)
        self.model.fit(X_train)
        
    def score(self, X):
        X_test = self._extract_bhatti_features(X)
        # score_samples returns positive for inliers, negative for outliers.
        # We negate so higher score = more anomalous (spoofing).
        return -self.model.score_samples(X_test)
