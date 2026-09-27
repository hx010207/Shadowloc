import numpy as np

class ConformalCalibrator:
    def __init__(self):
        self.calib_scores = None
        self.n = 0
        
    def fit(self, calib_scores):
        """
        calib_scores: 1D array of anomaly scores from a held-out benign calibration set.
        Higher score = more anomalous.
        """
        self.calib_scores = np.sort(calib_scores)
        self.n = len(calib_scores)
        
    def compute_p_values(self, test_scores):
        """
        Compute conformal p-value for each test score.
        p(x) = (1 + |{i : a_i >= a(x)}|) / (n + 1)
        where a_i are the calibration scores.
        """
        if self.calib_scores is None:
            raise ValueError("Calibrator not fitted.")
            
        p_values = np.zeros(len(test_scores))
        for i, score in enumerate(test_scores):
            # count how many calib scores are >= score
            # since calib_scores is sorted, we can use searchsorted
            # searchsorted with side='left' gives index of first element >= score
            idx = np.searchsorted(self.calib_scores, score, side='left')
            count_greater_equal = self.n - idx
            p_values[i] = (1.0 + count_greater_equal) / (self.n + 1.0)
            
        return p_values
        
    def predict(self, test_scores, alpha=0.05):
        """
        Returns boolean array where True means anomaly (spoofed), i.e., p(x) <= alpha.
        """
        p_values = self.compute_p_values(test_scores)
        return p_values <= alpha
