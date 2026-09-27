"""
Logistic Fusion layer and physics-based sub-detectors for ShadowLoc.

SubDetectors computes three heuristic scores:
  P_rf  — RF anomaly (C/N0 rise + AGC drop during spoofing)
  P_t   — Timing anomaly (clock offset exceeds benign jitter floor)
  r_mc  — Motion-consistency residual (GNSS displacement vs IMU)

FusionLogistic combines ML anomaly score (P_ml) with the three
sub-detector outputs via logistic regression.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression


class SubDetectors:
    """Physics-based sub-detectors with adaptive threshold calibration.

    Bug 1 Fix: Instead of static w1/w2/w3 that saturate the sigmoid on
    benign noise, fit() computes adaptive scales from the benign training
    distribution so that 95th percentile of benign scores maps to
    sigmoid output = 0.5.
    """

    def __init__(self):
        # These will be set by fit()
        self.baseline_cn0 = None
        self.baseline_agc = None
        self.scale_rf = 1.0
        self.bias_rf = 0.0
        self.scale_t = 1.0
        self.bias_t = 0.0
        self._fitted = False

    @staticmethod
    def _sigmoid(x):
        x_clip = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x_clip))

    def fit(self, df_benign, percentile=95):
        """Calibrate scales and biases against benign training distribution.

        After fit():
        - Benign P_rf and P_t values cluster near 0.0-0.5
        - The 95th percentile of benign scores gives sigmoid output = 0.5
        - B0 FAR ≈ 5% by construction
        """
        from data_gen.attacks import haversine_dist

        self.baseline_cn0 = np.median(df_benign['mean_cn0'])
        self.baseline_agc = np.median(df_benign['agc'])

        # Raw RF scores (before scaling)
        delta_cn0 = df_benign['mean_cn0'].values - self.baseline_cn0
        delta_agc = df_benign['agc'].values - self.baseline_agc
        raw_rf = delta_cn0 - delta_agc  # unweighted

        # Raw timing scores
        raw_t = np.abs(df_benign['delta_tau'].values)

        # Motion-consistency residual on benign data
        lat = df_benign['gnss_lat'].values
        lon = df_benign['gnss_lon'].values
        d_pos = np.zeros(len(df_benign))
        d_pos[1:] = haversine_dist(lat[:-1], lon[:-1], lat[1:], lon[1:])
        raw_rmc = np.abs(d_pos) - df_benign['d_imu'].values

        # Adaptive scaling: normalize so 1 std ≈ 1.0 in sigmoid input
        self.scale_rf = max(np.std(raw_rf), 1e-6)
        self.scale_t = max(np.std(raw_t), 1e-6)

        # Set bias so that 95th percentile of benign → sigmoid(0) = 0.5
        # i.e., bias = percentile_95(scaled_score)
        scaled_rf = raw_rf / self.scale_rf
        scaled_t = raw_t / self.scale_t

        self.bias_rf = float(np.percentile(scaled_rf, percentile))
        self.bias_t = float(np.percentile(scaled_t, percentile))

        self._fitted = True

    def compute_pointwise(self, df):
        """Compute per-epoch sub-detector scores using calibrated parameters."""
        from data_gen.attacks import haversine_dist

        if not self._fitted:
            raise RuntimeError("SubDetectors.fit() must be called before compute_pointwise()")

        delta_cn0 = df['mean_cn0'].values - self.baseline_cn0
        delta_agc = df['agc'].values - self.baseline_agc
        raw_rf = delta_cn0 - delta_agc

        # Scaled and biased → sigmoid
        P_rf = self._sigmoid((raw_rf / self.scale_rf) - self.bias_rf)

        raw_t = np.abs(df['delta_tau'].values)
        P_t = self._sigmoid((raw_t / self.scale_t) - self.bias_t)

        # Motion-consistency residual (not sigmoided — raw meters)
        lat = df['gnss_lat'].values
        lon = df['gnss_lon'].values
        d_pos = np.zeros(len(df))
        d_pos[1:] = haversine_dist(lat[:-1], lon[:-1], lat[1:], lon[1:])
        r_mc = np.abs(d_pos) - df['d_imu'].values

        return P_rf, P_t, r_mc

    def debug_print(self, df_benign_sample, label=""):
        """Print P_rf and P_t statistics on a benign sample for verification."""
        P_rf, P_t, r_mc = self.compute_pointwise(df_benign_sample)
        print(f"  [SubDetectors DEBUG {label}]")
        print(f"    P_rf — mean: {np.mean(P_rf):.4f}, std: {np.std(P_rf):.4f}, "
              f"5th: {np.percentile(P_rf, 5):.4f}, 95th: {np.percentile(P_rf, 95):.4f}")
        print(f"    P_t  — mean: {np.mean(P_t):.4f}, std: {np.std(P_t):.4f}, "
              f"5th: {np.percentile(P_t, 5):.4f}, 95th: {np.percentile(P_t, 95):.4f}")
        print(f"    r_mc — mean: {np.mean(r_mc):.4f}, std: {np.std(r_mc):.4f}")


class FusionLogistic:
    """Logistic regression fusion of ML anomaly score with sub-detectors.

    Combines four features: [P_ml, P_rf, P_t, r_mc] into a single
    spoofing probability via logistic regression.
    """

    def __init__(self, random_state=42):
        self.model = LogisticRegression(
            class_weight='balanced', random_state=random_state, max_iter=1000
        )
        self._fitted = False

    def fit(self, X_fusion, y):
        """Fit logistic regression on fusion feature matrix.

        X_fusion: (N_windows, 4) — columns [P_ml, P_rf, P_t, r_mc]
        y: boolean labels (True = spoofed)
        """
        self.model.fit(X_fusion, y)
        self._fitted = True

    def predict_proba(self, X_fusion):
        """Return spoofing probability for each window."""
        if not self._fitted:
            raise RuntimeError("FusionLogistic not fitted")
        return self.model.predict_proba(X_fusion)[:, 1]

    def predict(self, X_fusion, threshold=0.5):
        """Binary spoofing prediction."""
        probs = self.predict_proba(X_fusion)
        return probs >= threshold

    def debug_print(self):
        """Print learned coefficients for verification."""
        if not self._fitted:
            print("  [FusionLogistic] Not fitted yet")
            return
        coefs = self.model.coef_[0]
        names = ['P_ml', 'P_rf', 'P_t', 'r_mc']
        print(f"  [FusionLogistic] Learned coefficients:")
        for name, c in zip(names, coefs):
            print(f"    {name}: {c:.4f}")
        print(f"  [FusionLogistic] Intercept: {self.model.intercept_[0]:.4f}")
