"""
Script to complete pending evaluation steps for ShadowLoc:
- Runs ablation study (10 seeds) -> results/ablation.csv
- Runs conformal calibration curve (10 seeds) -> results/conformal_calibration.csv
- Runs inference cost profiling -> results/inference_cost.csv
- Collects ROC / debug data and renders all 8 figures in figures/
- Generates results/RESULTS_SUMMARY.md
"""

import os
import csv
import json
import numpy as np
from sklearn.metrics import roc_curve, auc

from main import write_results_summary, run_calibration_check
from evaluation.ablation import run_ablation, aggregate_ablation
from evaluation.profiler import profile_inference
from models.variant_a import VariantA
from models.variant_b import VariantB
from models.baselines import Baseline1, Baseline2
from models.external_baseline import BhattiBaseline
from models.fusion import SubDetectors, FusionLogistic
from conformal.calibration import ConformalCalibrator
from features.extractor import FeatureExtractor
from data_gen.attacks import a1_jump, a2_drift, a3_intermittent, a4_time_bias, a5_adaptive
from evaluation.harness import prepare_data, GNSS_ONLY_IDX, ATTACKS, METHODS, extract_subdetector_features
import plot.plots as plots


def load_agg(filepath='results/results_summary.json'):
    with open(filepath, 'r') as f:
        return json.load(f)


def load_drift_rows(filepath='results/a2_drift_sensitivity.csv'):
    rows = []
    with open(filepath, 'r') as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({
                'drift_rate': float(row['drift_rate']),
                'method': row['method'],
                'recall_mean': float(row['recall_mean']),
                'recall_ci': float(row['recall_ci']),
                'delay_mean': float(row['delay_mean']) if row['delay_mean'] != 'NaN' else np.nan,
                'delay_ci': float(row['delay_ci']) if row['delay_ci'] != 'NaN' else np.nan,
            })
    return rows


def generate_roc_and_debug_data(seed=0):
    print("Collecting ROC and debug score distributions for seed 0...")
    df_train, df_calib, df_val, df_test, _ = prepare_data(seed, fs=1.0)
    extractor = FeatureExtractor(W=10, stride=1, fs=1.0)
    extractor.fit(df_train)

    X_train, _, _ = extractor.transform(df_train)
    X_calib, _, _ = extractor.transform(df_calib)
    X_test_b, _, _ = extractor.transform(df_test)
    X_train_gnss = X_train[:, :, GNSS_ONLY_IDX]

    # Models
    var_a = VariantA(method='mahalanobis', random_state=seed)
    var_a.fit(X_train)

    var_b = VariantB(input_dim=X_train.shape[2], epochs=5)
    var_b.fit(X_train)

    b2 = Baseline2(input_dim=len(GNSS_ONLY_IDX), epochs=5)
    b2.fit(X_train_gnss)

    ext_base = BhattiBaseline()
    ext_base.fit(X_train)

    # Conformal calibration for p-values
    cal_b = ConformalCalibrator()
    cal_b.fit(var_b.score(X_calib))

    sub_det = SubDetectors()
    sub_det.fit(df_train)

    # Train Fusion on val
    X_val_b, _, _ = extractor.transform(df_val)
    n_b = len(X_val_b)
    attack_fns = {
        'A1': lambda df: a1_jump(df, fs=1.0),
        'A2': lambda df: a2_drift(df, fs=1.0, drift_rate=1.0),
        'A3': lambda df: a3_intermittent(df, fs=1.0),
        'A4': lambda df: a4_time_bias(df, fs=1.0),
        'A5': lambda df: a5_adaptive(df, fs=1.0),
    }

    fusion_X_parts, fusion_y_parts = [X_val_b], [np.zeros(n_b, dtype=int)]
    n_per = max(1, n_b // 5)
    for aname, afn in attack_fns.items():
        df_att = afn(df_val)
        X_att, y_att, _ = extractor.transform(df_att)
        mask = y_att.astype(bool)
        if np.sum(mask) > 0:
            fusion_X_parts.append(X_att[mask][:n_per])
            fusion_y_parts.append(np.ones(min(n_per, np.sum(mask)), dtype=int))

    X_fus_all = np.vstack(fusion_X_parts)
    y_fus_all = np.concatenate(fusion_y_parts)
    p_ml_val = 1.0 - cal_b.compute_p_values(var_b.score(X_fus_all))

    # Sub-detector features for val
    P_rf_parts, P_t_parts, rmc_parts = [], [], []
    P_rf_b, P_t_b, rmc_b = extract_subdetector_features(X_val_b, df_val, sub_det, W=10)
    P_rf_parts.append(P_rf_b)
    P_t_parts.append(P_t_b)
    rmc_parts.append(rmc_b)

    for aname, afn in attack_fns.items():
        df_att = afn(df_val)
        X_att, y_att, _ = extractor.transform(df_att)
        mask = y_att.astype(bool)
        if np.sum(mask) > 0:
            prf_a, pt_a, rmc_a = extract_subdetector_features(X_att, df_att, sub_det, W=10)
            P_rf_parts.append(prf_a[mask][:n_per])
            P_t_parts.append(pt_a[mask][:n_per])
            rmc_parts.append(rmc_a[mask][:n_per])

    X_fus_features = np.column_stack([
        p_ml_val[:len(np.concatenate(P_rf_parts))],
        np.concatenate(P_rf_parts),
        np.concatenate(P_t_parts),
        np.concatenate(rmc_parts)
    ])
    fusion = FusionLogistic(random_state=seed)
    fusion.fit(X_fus_features, y_fus_all[:len(X_fus_features)])

    # Collect ROC data per attack
    roc_data = {}
    debug_data = {
        'calib_scores_a': var_a.score(X_calib),
        'test_benign_scores_a': var_a.score(X_test_b),
    }

    for att in ATTACKS:
        df_attack = attack_fns[att](df_test)
        X_att, y_att, _ = extractor.transform(df_attack)
        X_att_gnss = X_att[:, :, GNSS_ONLY_IDX]

        prf_w, pt_w, rmc_w = extract_subdetector_features(X_att, df_attack, sub_det, W=10)

        s_a = var_a.score(X_att)
        s_b = var_b.score(X_att)
        s_b2 = b2.score(X_att_gnss)
        s_ext = ext_base.score(X_att)

        p_ml_test = 1.0 - cal_b.compute_p_values(s_b)
        X_fus_test = np.column_stack([p_ml_test, prf_w, pt_w, rmc_w])
        s_fus = fusion.predict_proba(X_fus_test)

        roc_data[att] = {
            'y_true': y_att,
            'Variant A': s_a,
            'Variant B': s_b,
            'B2': s_b2,
            'ExtBaseline': s_ext,
            'Fusion': s_fus,
        }

        if att in ['A1', 'A2']:
            debug_data[f'{att}_scores_a'] = s_a

    return roc_data, debug_data


def render_roc_curves_proper(roc_data, output_dir):
    plots._setup()
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5), sharey=True)

    methods = ['Variant A', 'Variant B', 'Fusion', 'B2', 'ExtBaseline']

    for i, att in enumerate(ATTACKS):
        ax = axes[i]
        d = roc_data[att]
        y_true = d['y_true']
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')

        for m in methods:
            y_score = d[m]
            if len(np.unique(y_true)) > 1:
                fpr, tpr, _ = roc_curve(y_true, y_score)
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, label=f"{m} ({roc_auc:.3f})",
                        color=plots.COLORS.get(m, '#333'), linewidth=1.8)

        ax.set_title(f'Attack {att}', fontweight='bold')
        ax.set_xlabel('False Positive Rate')
        if i == 0:
            ax.set_ylabel('True Positive Rate')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc='lower right')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    plt.close()
    print("Saved roc_curve.png")


def main():
    os.makedirs('results', exist_ok=True)
    os.makedirs('figures', exist_ok=True)

    print("Loading precomputed 30-seed results and A2 drift sensitivity...")
    agg = load_agg('results/results_summary.json')
    drift_rows = load_drift_rows('results/a2_drift_sensitivity.csv')

    # 1. Ablation Study
    print("Running ablation study (10 seeds)...")
    ablation_raw = run_ablation(num_seeds=10)
    ablation_rows = aggregate_ablation(ablation_raw)

    with open('results/ablation.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['config', 'pooled_f1_mean', 'pooled_f1_ci'])
        for row in ablation_rows:
            w.writerow([row['config'],
                        f"{row['pooled_f1_mean']:.4f}",
                        f"{row['pooled_f1_ci']:.4f}"])
    print("Saved results/ablation.csv")

    # 2. Conformal Calibration
    print("Running conformal calibration check (10 seeds)...")
    calib_data = run_calibration_check(num_seeds=10)

    with open('results/conformal_calibration.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['alpha', 'va_empirical_far', 'vb_empirical_far'])
        for i, alpha in enumerate(calib_data['alphas']):
            w.writerow([alpha, f"{calib_data['va_fars'][i]:.4f}",
                        f"{calib_data['vb_fars'][i]:.4f}"])
    print("Saved results/conformal_calibration.csv")

    # 3. Inference Cost Profiling
    print("Profiling Inference Cost (CPU)...")
    X_prof = np.random.randn(100, 10, 16)
    var_a = VariantA()
    var_a.fit(X_prof)
    lat_a, mem_a = profile_inference(var_a, X_prof, model_type='sklearn')

    var_b = VariantB(input_dim=16, epochs=1)
    var_b.fit(X_prof)
    lat_b, mem_b = profile_inference(var_b, X_prof, model_type='pytorch')

    cost_metrics = {
        'Variant A': {'latency_ms': lat_a, 'peak_mem_mb': mem_a},
        'Variant B': {'latency_ms': lat_b, 'peak_mem_mb': mem_b},
        'Note': 'Measured on host CPU using time.perf_counter and tracemalloc. '
                'Mobile ARM numbers will differ.'
    }

    with open('results/inference_cost.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['variant', 'latency_ms', 'peak_mem_mb'])
        for m in ['Variant A', 'Variant B']:
            w.writerow([m, f"{cost_metrics[m]['latency_ms']:.2f}",
                        f"{cost_metrics[m]['peak_mem_mb']:.2f}"])
    print("Saved results/inference_cost.csv")

    # 4. Generate all figures
    print("Generating all figures...")
    plots.plot_pooled_f1_bar(agg, 'figures')
    plots.plot_per_attack_recall(agg, 'figures')
    plots.plot_drift_sensitivity(drift_rows, 'figures')
    plots.plot_ablation(ablation_rows, 'figures')
    plots.plot_conformal_calibration(calib_data, 'figures')
    plots.plot_cross_device(agg, 'figures')

    # ROC curves and debug distributions
    roc_data, debug_data = generate_roc_and_debug_data(seed=0)
    render_roc_curves_proper(roc_data, 'figures')
    plots.plot_debug_score_distributions(debug_data, 'figures')
    print("All 8 figures successfully generated in figures/")

    # 5. Write RESULTS_SUMMARY.md
    print("Writing results/RESULTS_SUMMARY.md...")
    write_results_summary(agg, cost_metrics, calib_data, drift_rows, ablation_rows,
                          filepath='results/RESULTS_SUMMARY.md')
    print("results/RESULTS_SUMMARY.md written successfully.")


if __name__ == '__main__':
    main()
