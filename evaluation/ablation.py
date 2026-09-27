"""
Ablation study for ShadowLoc Variant B.

Tests 6 feature configurations to quantify the contribution
of each modality and the conformal calibration layer.
"""

import numpy as np
from sklearn.metrics import f1_score, recall_score
from tqdm import tqdm

from evaluation.harness import (
    prepare_data, GNSS_ONLY_IDX, FEATURE_NAMES,
    compute_detection_delay
)
from features.extractor import FeatureExtractor
from models.variant_b import VariantB
from conformal.calibration import ConformalCalibrator
from data_gen.attacks import (a1_jump, a2_drift, a3_intermittent,
                              a4_time_bias, a5_adaptive)


# Feature index groups
IMU_IDX = [FEATURE_NAMES.index(f) for f in ['d_imu', 'omega_imu']]
RADIO_IDX = [FEATURE_NAMES.index(f) for f in ['j_wifi', 'j_cell', 'rho_rssi']]
TIMING_IDX = [FEATURE_NAMES.index(f) for f in ['delta_tau', 'delta_tau_rate']]
AGC_IDX = [FEATURE_NAMES.index(f) for f in ['agc', 'agc_slope']]

CONFIGS = {
    'Full': None,  # All features
    'Minus IMU': IMU_IDX,
    'Minus Radio': RADIO_IDX,
    'Minus Timing': TIMING_IDX,
    'Minus AGC': AGC_IDX,
    'Minus Conformal': 'no_conformal',
}


def _exclude_features(X, exclude_idx):
    """Remove feature columns from windowed array X: (N, W, D)."""
    keep = [i for i in range(X.shape[2]) if i not in exclude_idx]
    return X[:, :, keep]


def run_ablation_single(seed, fs=1.0, W=10, alpha=0.05):
    """Run ablation study for a single seed."""
    df_train, df_calib, df_val, df_test, _ = prepare_data(seed, fs)

    extractor = FeatureExtractor(W=W, stride=1, fs=fs)
    extractor.fit(df_train)
    X_train, _, _ = extractor.transform(df_train)
    X_calib, _, _ = extractor.transform(df_calib)

    attack_fns = {
        'A1': lambda df: a1_jump(df, fs=fs),
        'A2': lambda df: a2_drift(df, fs=fs, drift_rate=1.0),
        'A3': lambda df: a3_intermittent(df, fs=fs),
        'A4': lambda df: a4_time_bias(df, fs=fs),
        'A5': lambda df: a5_adaptive(df, fs=fs),
    }

    test_sets = {}
    for att_name, att_fn in attack_fns.items():
        df_att = att_fn(df_test)
        X_att, y_att, _ = extractor.transform(df_att)
        test_sets[att_name] = (X_att, y_att)

    results = {}
    var_b_full = None
    calib_scores_full = None

    for config_name, exclude in CONFIGS.items():
        # Prepare feature subset
        if exclude is None:
            X_tr = X_train
            X_cal = X_calib
        elif exclude == 'no_conformal':
            X_tr = X_train
            X_cal = X_calib
        else:
            X_tr = _exclude_features(X_train, exclude)
            X_cal = _exclude_features(X_calib, exclude)

        # Train Variant B on this subset (reuse Full model for no_conformal to strictly isolate thresholding)
        if exclude == 'no_conformal' and var_b_full is not None:
            var_b = var_b_full
            calib_scores = calib_scores_full
        else:
            var_b = VariantB(input_dim=X_tr.shape[2], epochs=5)
            var_b.fit(X_tr)
            calib_scores = var_b.score(X_cal)
            if config_name == 'Full':
                var_b_full = var_b
                calib_scores_full = calib_scores

        if exclude == 'no_conformal':
            # Uncalibrated fixed threshold targeting the nominal alpha=0.05 level
            # (Note: mean + 3*sigma corresponds to alpha ~ 0.001-0.01 on right-skewed MSE distributions,
            # which severely starves recall. The nominal fixed baseline is the empirical (1-alpha) percentile.)
            threshold = np.percentile(calib_scores, (1.0 - alpha) * 100)
        else:
            calibrator = ConformalCalibrator()
            calibrator.fit(calib_scores)

        # Evaluate on all attacks
        f1s = []
        for att_name, (X_att, y_att) in test_sets.items():
            if exclude is None or exclude == 'no_conformal':
                X_att_sub = X_att
            else:
                X_att_sub = _exclude_features(X_att, exclude)

            scores = var_b.score(X_att_sub)

            if exclude == 'no_conformal':
                y_pred = scores >= threshold
            else:
                y_pred = calibrator.predict(scores, alpha)

            f1 = f1_score(y_att, y_pred, zero_division=0)
            f1s.append(f1)

        pooled_f1 = np.mean(f1s)
        results[config_name] = pooled_f1

    return results


def run_ablation(num_seeds=30):
    """Run ablation across all seeds."""
    all_results = {config: [] for config in CONFIGS}

    print("Running ablation study...")
    for seed in tqdm(range(num_seeds), desc="Ablation"):
        try:
            res = run_ablation_single(seed)
            for config, f1 in res.items():
                all_results[config].append(f1)
        except Exception as e:
            print(f"  Error on seed {seed}: {e}")

    return all_results


def aggregate_ablation(all_results):
    """Aggregate ablation results into mean ± 95% CI."""
    rows = []
    for config in CONFIGS:
        vals = np.array(all_results[config])
        valid = vals[~np.isnan(vals)]
        mean = np.mean(valid) if len(valid) > 0 else 0.0
        ci = (1.96 * np.std(valid) / np.sqrt(len(valid))
              if len(valid) > 1 else 0.0)
        rows.append({
            'config': config,
            'pooled_f1_mean': mean,
            'pooled_f1_ci': ci,
        })
    return rows
