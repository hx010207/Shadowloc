"""
ShadowLoc evaluation harness.

Fixes applied:
  Bug 1: SubDetectors.fit() calibrates against benign distribution
  Bug 2: Fusion trained on balanced mix of all 5 attacks + benign
  Bug 3: Shuffle-then-split eliminates temporal distribution mismatch
  Missing 1: Detection delay for both Variant A and B on A2
  Missing 5: Cross-device detection on all 5 attacks
  Missing 6: All models retrained per-seed with seed-dependent splits
"""

import random
import torch
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix)
from tqdm import tqdm

from data_gen.gsdc_loader import load_gsdc_trace
from data_gen.hybrid_synthesizer import synthesize_hybrid_modalities
from data_gen.attacks import (a1_jump, a2_drift, a3_intermittent,
                              a4_time_bias, a5_adaptive, a2_stealth)
from features.extractor import FeatureExtractor
from models.variant_a import VariantA
from models.variant_b import VariantB
from models.baselines import Baseline1, Baseline2
from models.external_baseline import BhattiBaseline
from models.fusion import SubDetectors, FusionLogistic
from conformal.calibration import ConformalCalibrator


def set_seed(seed):
    """Set random seeds across random, numpy, and torch for 100% determinism."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Feature extractor produces 16 features in this order:
FEATURE_NAMES = [
    'd_pos', 'd_speed', 'd_bearing', 'a_gnss',       # 0-3: GNSS dynamics
    'mean_cn0', 'std_cn0', 'n_sat',                   # 4-6: Signal stats
    'agc', 'agc_slope',                                # 7-8: Interference
    'delta_tau', 'delta_tau_rate',                      # 9-10: Timing
    'd_imu', 'omega_imu',                              # 11-12: Inertial
    'j_wifi', 'j_cell', 'rho_rssi'                     # 13-15: Radio context
]
GNSS_ONLY_IDX = list(range(9))  # First 9 features are GNSS-only

METHODS = ['B0', 'B1', 'B2', 'ExtBaseline', 'Variant A', 'Variant B', 'Fusion']
ATTACKS = ['A1', 'A2', 'A3', 'A4', 'A5']


def compute_metrics(y_true, y_pred, y_score=None):
    """Compute precision, recall, F1, AUC, FAR from predictions."""
    if len(np.unique(y_true)) > 1 and y_score is not None:
        try:
            roc_auc = roc_auc_score(y_true, y_score)
        except ValueError:
            roc_auc = np.nan
    else:
        roc_auc = np.nan

    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {'precision': prec, 'recall': rec, 'f1': f1,
            'roc_auc': roc_auc, 'far': far}


def compute_detection_delay(y_true, y_pred):
    """Compute detection delay in epochs from first spoofed window to first alarm."""
    spoof_starts = np.where(y_true)[0]
    if len(spoof_starts) == 0:
        return np.nan
    start_idx = spoof_starts[0]
    preds_after = y_pred[start_idx:]
    alarms = np.where(preds_after)[0]
    return float(alarms[0]) if len(alarms) > 0 else np.nan


def extract_subdetector_features(X_test, df_raw, sub_detectors, W=10):
    """Compute windowed sub-detector features by taking max over each window."""
    P_rf, P_t, r_mc = sub_detectors.compute_pointwise(df_raw)
    num_windows = X_test.shape[0]
    P_rf_w = np.array([np.max(P_rf[i:i+W]) for i in range(num_windows)])
    P_t_w = np.array([np.max(P_t[i:i+W]) for i in range(num_windows)])
    r_mc_w = np.array([np.max(np.abs(r_mc[i:i+W])) for i in range(num_windows)])
    return P_rf_w, P_t_w, r_mc_w


def prepare_data(seed, fs=1.0):
    """Load and prepare shuffled IID data splits for one seed.

    Bug 3 Fix: Synthesize modalities first (needs temporal order),
    then shuffle rows to make all splits IID.
    Missing 6 Fix: seed controls shuffle → different splits per seed.
    """
    set_seed(seed)

    # Load and synthesize (temporal order preserved for derivative computation)
    df_a = load_gsdc_trace('Pixel4')
    df_a = synthesize_hybrid_modalities(df_a, fs=fs)

    df_b = load_gsdc_trace('Pixel4XL')
    df_b = synthesize_hybrid_modalities(df_b, fs=fs)

    # Shuffle Device A rows AFTER synthesis (Bug 3 fix)
    perm_a = np.random.permutation(len(df_a))
    df_a = df_a.iloc[perm_a].reset_index(drop=True)

    perm_b = np.random.permutation(len(df_b))
    df_b = df_b.iloc[perm_b].reset_index(drop=True)

    # Split Device A: 60% train, 20% calib, 10% val, 10% test
    n = len(df_a)
    n_train = int(n * 0.60)
    n_calib = int(n * 0.20)
    n_val = int(n * 0.10)

    df_train = df_a.iloc[:n_train].reset_index(drop=True)
    df_calib = df_a.iloc[n_train:n_train+n_calib].reset_index(drop=True)
    df_val = df_a.iloc[n_train+n_calib:n_train+n_calib+n_val].reset_index(drop=True)
    df_test = df_a.iloc[n_train+n_calib+n_val:].reset_index(drop=True)

    return df_train, df_calib, df_val, df_test, df_b


def run_experiment(seed, fs=1.0, W=10, alpha=0.05, verbose=False):
    """Run one complete evaluation experiment for a single seed."""
    set_seed(seed)
    df_train, df_calib, df_val, df_test, df_device_b = prepare_data(seed, fs)

    # --- Generate attacks on test split ---
    attack_fns = {
        'A1': lambda df: a1_jump(df, fs=fs),
        'A2': lambda df: a2_drift(df, fs=fs, drift_rate=1.0),
        'A3': lambda df: a3_intermittent(df, fs=fs),
        'A4': lambda df: a4_time_bias(df, fs=fs),
        'A5': lambda df: a5_adaptive(df, fs=fs),
    }
    df_attacks = {name: fn(df_test) for name, fn in attack_fns.items()}

    # Cross-device attacks
    df_xdev_attacks = {
        f'CrossDevice_{name}': fn(df_device_b)
        for name, fn in attack_fns.items()
    }

    # --- Feature extraction ---
    extractor = FeatureExtractor(W=W, stride=1, fs=fs)
    extractor.fit(df_train)

    X_train, _, _ = extractor.transform(df_train)
    X_calib, _, _ = extractor.transform(df_calib)

    if verbose:
        print(f"  Calibration set size n: {len(X_calib)}")

    # --- Train all models (per-seed → Missing 6 fix) ---
    # Variant A: Mahalanobis (unsupervised, all features)
    var_a = VariantA(method='mahalanobis', random_state=seed)
    var_a.fit(X_train)

    # Variant B: LSTM autoencoder (unsupervised, all features)
    var_b = VariantB(input_dim=X_train.shape[2], epochs=5, batch_size=64)
    var_b.fit(X_train)

    # B1: Supervised RandomForest (GNSS-only, needs labeled data)
    # Create labeled training data by mixing benign + attacked windows
    X_train_gnss = X_train[:, :, GNSS_ONLY_IDX]
    y_train_benign = np.zeros(len(X_train), dtype=int)

    # Generate attacks on TRAINING data for B1 supervised labels
    df_train_a2 = a2_drift(df_train, fs=fs, drift_rate=1.5)
    X_train_a2, y_train_a2, _ = extractor.transform(df_train_a2)
    X_train_a2_gnss = X_train_a2[:, :, GNSS_ONLY_IDX]

    X_b1_train = np.vstack([X_train_gnss, X_train_a2_gnss])
    y_b1_train = np.concatenate([y_train_benign, y_train_a2.astype(int)])

    b1 = Baseline1(random_state=seed)
    b1.fit(X_b1_train, y_b1_train)

    # B2: GNSS-only LSTM (unsupervised)
    b2 = Baseline2(input_dim=len(GNSS_ONLY_IDX), epochs=5)
    b2.fit(X_train_gnss)

    # ExtBaseline: Bhatti One-Class SVM (unsupervised)
    ext_base = BhattiBaseline()
    ext_base.fit(X_train)

    # --- Conformal calibration ---
    calib_a = ConformalCalibrator()
    calib_a.fit(var_a.score(X_calib))

    calib_b = ConformalCalibrator()
    calib_b.fit(var_b.score(X_calib))

    calib_b2 = ConformalCalibrator()
    calib_b2.fit(b2.score(X_calib[:, :, GNSS_ONLY_IDX]))

    calib_ext = ConformalCalibrator()
    calib_ext.fit(ext_base.score(X_calib))

    # Bug 3 verification: check calibration vs test score overlap
    if verbose:
        X_test_benign, _, _ = extractor.transform(df_test)
        scores_calib = var_a.score(X_calib)
        scores_test = var_a.score(X_test_benign)
        ks_stat, _ = ks_2samp(scores_calib, scores_test)
        print(f"  Calibration vs test score KS statistic: {ks_stat:.3f}")

    # --- Sub-detectors (Bug 1 fix: calibrated) ---
    sub_det = SubDetectors()
    sub_det.fit(df_train)

    if verbose:
        sub_det.debug_print(df_test.head(200), label="benign test sample")
        # Verify B0 FAR on benign
        X_test_b, y_test_b, _ = extractor.transform(df_test)
        P_rf_b, P_t_b, _ = extract_subdetector_features(
            X_test_b, df_test, sub_det, W)
        b0_preds = (P_rf_b > 0.5) | (P_t_b > 0.5)
        b0_far = np.mean(b0_preds)
        print(f"  B0 benign FAR after fix: {b0_far:.3f}")

    # --- Logistic Fusion training (Bug 2 fix: balanced multi-attack) ---
    # Build balanced fusion training set: benign + all 5 attack types
    X_val_benign, _, _ = extractor.transform(df_val)
    n_benign = len(X_val_benign)

    fusion_X_parts = [X_val_benign]
    fusion_y_parts = [np.zeros(n_benign, dtype=int)]

    n_per_attack = max(1, n_benign // 5)
    for att_name, att_fn in attack_fns.items():
        df_val_att = att_fn(df_val)
        X_val_att, y_val_att, _ = extractor.transform(df_val_att)
        # Take attacked windows only
        att_mask = y_val_att.astype(bool)
        if np.sum(att_mask) > 0:
            X_att_only = X_val_att[att_mask][:n_per_attack]
            fusion_X_parts.append(X_att_only)
            fusion_y_parts.append(np.ones(len(X_att_only), dtype=int))

    X_fusion_all = np.vstack(fusion_X_parts)
    y_fusion_all = np.concatenate(fusion_y_parts)

    # Compute fusion features
    # We need raw DataFrames for sub-detector computation — use val + attacked val
    # For simplicity, compute P_ml from Variant B scores on fusion windows
    score_fusion = var_b.score(X_fusion_all)
    P_ml_fusion = 1.0 - calib_b.compute_p_values(score_fusion)

    # Sub-detector features: compute on full val DF, then pick windows
    # For the benign portion, use df_val; for attacked, compute separately
    P_rf_parts, P_t_parts, rmc_parts = [], [], []

    # Benign part
    P_rf_b, P_t_b, rmc_b = extract_subdetector_features(
        X_val_benign, df_val, sub_det, W)
    P_rf_parts.append(P_rf_b)
    P_t_parts.append(P_t_b)
    rmc_parts.append(rmc_b)

    # Attacked parts
    for att_name, att_fn in attack_fns.items():
        df_val_att = att_fn(df_val)
        X_val_att, y_val_att, _ = extractor.transform(df_val_att)
        att_mask = y_val_att.astype(bool)
        if np.sum(att_mask) > 0:
            P_rf_a, P_t_a, rmc_a = extract_subdetector_features(
                X_val_att, df_val_att, sub_det, W)
            P_rf_parts.append(P_rf_a[att_mask][:n_per_attack])
            P_t_parts.append(P_t_a[att_mask][:n_per_attack])
            rmc_parts.append(rmc_a[att_mask][:n_per_attack])

    P_rf_fusion = np.concatenate(P_rf_parts)
    P_t_fusion = np.concatenate(P_t_parts)
    rmc_fusion = np.concatenate(rmc_parts)

    X_fus_features = np.column_stack([
        P_ml_fusion[:len(P_rf_fusion)],
        P_rf_fusion, P_t_fusion, rmc_fusion
    ])
    y_fus_labels = y_fusion_all[:len(P_rf_fusion)]

    fusion = FusionLogistic(random_state=seed)
    if len(np.unique(y_fus_labels)) > 1:
        fusion.fit(X_fus_features, y_fus_labels)

        if verbose:
            fusion.debug_print()
            variances = np.var(X_fus_features, axis=0)
            print(f"  Fusion feature variances: {variances}")

    # --- Evaluate on all test scenarios ---
    all_test_dfs = {'Benign': df_test}
    all_test_dfs.update(df_attacks)
    all_test_dfs['CrossDevice_Benign'] = df_device_b
    all_test_dfs.update(df_xdev_attacks)

    results = {}
    for name, df_raw in all_test_dfs.items():
        X_test, y_test, delta_pos = extractor.transform(df_raw)
        X_test_gnss = X_test[:, :, GNSS_ONLY_IDX]

        # Sub-detector features
        P_rf_w, P_t_w, r_mc_w = extract_subdetector_features(
            X_test, df_raw, sub_det, W)

        # Variant A
        score_a = var_a.score(X_test)
        y_pred_a = calib_a.predict(score_a, alpha)

        # Variant B
        score_b = var_b.score(X_test)
        p_val_b = calib_b.compute_p_values(score_b)
        y_pred_b = p_val_b <= alpha

        # B1 (supervised RF, GNSS-only)
        score_b1 = b1.score(X_test_gnss)
        y_pred_b1 = score_b1 >= 0.5  # RF probability threshold

        # B2 (GNSS-only LSTM)
        score_b2 = b2.score(X_test_gnss)
        y_pred_b2 = calib_b2.predict(score_b2, alpha)

        # ExtBaseline
        score_ext = ext_base.score(X_test)
        y_pred_ext = calib_ext.predict(score_ext, alpha)

        # B0 heuristics
        y_pred_b0 = (P_rf_w > 0.5) | (P_t_w > 0.5)

        # Logistic Fusion
        P_ml = 1.0 - p_val_b
        if fusion._fitted:
            X_fus_test = np.column_stack([P_ml, P_rf_w, P_t_w, r_mc_w])
            y_pred_fusion = fusion.predict(X_fus_test)
            score_fusion = fusion.predict_proba(X_fus_test)
        else:
            y_pred_fusion = y_pred_b
            score_fusion = P_ml

        res = {
            'Variant A': compute_metrics(y_test, y_pred_a, score_a),
            'Variant B': compute_metrics(y_test, y_pred_b, score_b),
            'Fusion': compute_metrics(y_test, y_pred_fusion, score_fusion),
            'B0': compute_metrics(y_test, y_pred_b0, P_rf_w + P_t_w),
            'B1': compute_metrics(y_test, y_pred_b1, score_b1),
            'B2': compute_metrics(y_test, y_pred_b2, score_b2),
            'ExtBaseline': compute_metrics(y_test, y_pred_ext, score_ext),
        }

        # Detection delay on A2 (Missing 1)
        if name == 'A2':
            res['Variant A']['delay'] = compute_detection_delay(y_test, y_pred_a)
            res['Variant B']['delay'] = compute_detection_delay(y_test, y_pred_b)
            res['B0']['delay'] = compute_detection_delay(y_test, y_pred_b0)
            res['B1']['delay'] = compute_detection_delay(y_test, y_pred_b1)
            res['B2']['delay'] = compute_detection_delay(y_test, y_pred_b2)
            res['ExtBaseline']['delay'] = compute_detection_delay(y_test, y_pred_ext)
            res['Fusion']['delay'] = compute_detection_delay(y_test, y_pred_fusion)

        # Verification gates (verbose mode)
        if verbose and name == 'A4':
            print(f"  Variant A Recall on A4 after fix: {res['Variant A']['recall']:.3f}")
        if verbose and name == 'A1':
            agree = np.mean(y_pred_fusion == y_pred_b0)
            print(f"  Fusion vs B0 agreement rate on A1: {agree:.3f}")

        results[name] = res

    # Store raw scores for debug plots
    if verbose:
        results['_debug'] = {
            'calib_scores_a': var_a.score(X_calib),
            'test_benign_scores_a': var_a.score(
                extractor.transform(df_test)[0]),
        }
        for att_name in ['A1', 'A2']:
            X_att, _, _ = extractor.transform(df_attacks[att_name])
            results['_debug'][f'{att_name}_scores_a'] = var_a.score(X_att)

    return results


def run_experiment_drift(seed, drift_rate, fs=1.0, W=10, alpha=0.05):
    """Run A2 at a specific drift rate for drift sensitivity analysis."""
    set_seed(seed)
    df_train, df_calib, df_val, df_test, _ = prepare_data(seed, fs)

    extractor = FeatureExtractor(W=W, stride=1, fs=fs)
    extractor.fit(df_train)
    X_train, _, _ = extractor.transform(df_train)
    X_calib, _, _ = extractor.transform(df_calib)

    # Train models
    var_a = VariantA(method='mahalanobis', random_state=seed)
    var_a.fit(X_train)

    var_b = VariantB(input_dim=X_train.shape[2], epochs=5)
    var_b.fit(X_train)

    X_train_gnss = X_train[:, :, GNSS_ONLY_IDX]
    df_train_a2 = a2_drift(df_train, fs=fs, drift_rate=1.5)
    X_train_a2, y_train_a2, _ = extractor.transform(df_train_a2)
    X_b1_train = np.vstack([X_train_gnss, X_train_a2[:, :, GNSS_ONLY_IDX]])
    y_b1_train = np.concatenate([np.zeros(len(X_train)), y_train_a2.astype(int)])
    b1 = Baseline1(random_state=seed)
    b1.fit(X_b1_train, y_b1_train)

    b2 = Baseline2(input_dim=len(GNSS_ONLY_IDX), epochs=5)
    b2.fit(X_train_gnss)

    ext_base = BhattiBaseline()
    ext_base.fit(X_train)

    sub_det = SubDetectors()
    sub_det.fit(df_train)

    # Calibrate
    calib_a = ConformalCalibrator()
    calib_a.fit(var_a.score(X_calib))
    calib_b = ConformalCalibrator()
    calib_b.fit(var_b.score(X_calib))
    calib_b2 = ConformalCalibrator()
    calib_b2.fit(b2.score(X_calib[:, :, GNSS_ONLY_IDX]))
    calib_ext = ConformalCalibrator()
    calib_ext.fit(ext_base.score(X_calib))

    # Inject A2 at specified drift rate
    df_test_a2 = a2_drift(df_test, fs=fs, drift_rate=drift_rate)
    X_test, y_test, _ = extractor.transform(df_test_a2)
    X_test_gnss = X_test[:, :, GNSS_ONLY_IDX]

    P_rf_w, P_t_w, r_mc_w = extract_subdetector_features(
        X_test, df_test_a2, sub_det, W)

    preds = {
        'Variant A': calib_a.predict(var_a.score(X_test), alpha),
        'Variant B': calib_b.compute_p_values(var_b.score(X_test)) <= alpha,
        'B0': (P_rf_w > 0.5) | (P_t_w > 0.5),
        'B1': b1.score(X_test_gnss) >= 0.5,
        'B2': calib_b2.predict(b2.score(X_test_gnss), alpha),
        'ExtBaseline': calib_ext.predict(ext_base.score(X_test), alpha),
    }

    res = {}
    for method, y_pred in preds.items():
        rec = recall_score(y_test, y_pred, zero_division=0)
        delay = compute_detection_delay(y_test, y_pred)
        res[method] = {'recall': rec, 'delay': delay}

    return res


def run_all_seeds(num_seeds=30, verbose_seed=0):
    """Run evaluation over all seeds, with verbose output for one seed."""
    all_results = []
    print(f"Running hybrid evaluation over {num_seeds} seeds...")
    for seed in tqdm(range(num_seeds)):
        try:
            res = run_experiment(
                seed, verbose=(seed == verbose_seed))
            all_results.append(res)
        except Exception as e:
            print(f"Error on seed {seed}: {e}")
            import traceback
            traceback.print_exc()

    return all_results
