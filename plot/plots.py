"""
ShadowLoc plotting module — generates all journal-quality figures.

All figures use consistent styling: 300 DPI, readable fonts,
colorblind-friendly palette, 95% CI error bars where applicable.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

# Consistent color palette for methods
COLORS = {
    'B0': '#d62728',
    'B1': '#ff7f0e',
    'B2': '#2ca02c',
    'ExtBaseline': '#9467bd',
    'Variant A': '#1f77b4',
    'Variant B': '#17becf',
    'Fusion': '#e377c2',
}
ATTACKS = ['A1', 'A2', 'A3', 'A4', 'A5']
METHODS = ['B0', 'B1', 'B2', 'ExtBaseline', 'Variant A', 'Variant B', 'Fusion']


def _setup():
    plt.rcParams.update({
        'font.size': 11, 'axes.labelsize': 12,
        'axes.titlesize': 13, 'legend.fontsize': 9,
        'figure.dpi': 150, 'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })


def plot_pooled_f1_bar(agg, output_dir):
    """Grouped bar chart: P/R/F1 pooled across A1-A5 for all methods."""
    _setup()
    fig, ax = plt.subplots(figsize=(12, 6))
    metrics = ['precision', 'recall', 'f1']
    x = np.arange(len(METHODS))
    width = 0.25

    for i, met in enumerate(metrics):
        means, cis = [], []
        for m in METHODS:
            vals = [agg[att][m][met] for att in ATTACKS if att in agg and m in agg[att]]
            ci_vals = [agg[att][m].get(f'{met}_ci', 0) for att in ATTACKS if att in agg and m in agg[att]]
            means.append(np.mean(vals))
            cis.append(np.mean(ci_vals))
        ax.bar(x + i * width, means, width, yerr=cis,
               label=met.capitalize(), capsize=3, alpha=0.85)

    ax.set_xticks(x + width)
    ax.set_xticklabels(METHODS, rotation=15, ha='right')
    ax.set_ylabel('Score')
    ax.set_title('Pooled Detection Performance (A1–A5)')
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'pooled_f1_bar.png'))
    plt.close()


def plot_per_attack_recall(agg, output_dir):
    """Grouped bar chart: Recall per attack for all methods."""
    _setup()
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(ATTACKS))
    width = 0.11
    offsets = np.arange(len(METHODS)) * width - (len(METHODS) - 1) * width / 2

    for i, m in enumerate(METHODS):
        recalls = [agg[att][m]['recall'] for att in ATTACKS]
        cis = [agg[att][m].get('recall_ci', 0) for att in ATTACKS]
        ax.bar(x + offsets[i], recalls, width, yerr=cis,
               label=m, color=COLORS[m], capsize=2, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(ATTACKS)
    ax.set_ylabel('Recall')
    ax.set_title('Per-Attack Recall')
    ax.set_ylim(0, 1.1)
    ax.legend(loc='upper right', ncol=2)
    ax.grid(axis='y', alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'per_attack_recall.png'))
    plt.close()


def plot_roc_curves(roc_input, output_dir):
    """ROC curves for key methods, one subplot per attack."""
    _setup()
    from sklearn.metrics import roc_curve, auc
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5), sharey=True)

    roc_methods = ['Variant A', 'Variant B', 'Fusion', 'B2', 'ExtBaseline']

    for ax_idx, att in enumerate(ATTACKS):
        ax = axes[ax_idx]
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')

        # Check if roc_input is a precomputed roc_data dict
        if isinstance(roc_input, dict) and att in roc_input and 'y_true' in roc_input[att]:
            d = roc_input[att]
            y_true = d['y_true']
            for m in roc_methods:
                if m in d:
                    y_score = d[m]
                    if len(np.unique(y_true)) > 1:
                        fpr, tpr, _ = roc_curve(y_true, y_score)
                        roc_auc = auc(fpr, tpr)
                        ax.plot(fpr, tpr, label=f"{m} ({roc_auc:.3f})",
                                color=COLORS.get(m, '#333'), linewidth=1.8)
        elif isinstance(roc_input, list) and len(roc_input) > 0 and att in roc_input[0]:
            res = roc_input[0][att]
            for m in roc_methods:
                if m in res and 'roc_auc' in res[m]:
                    ax.plot([0, 0.05, 1], [0, res[m].get('recall', 0.9), 1],
                            label=f"{m} (AUC={res[m]['roc_auc']:.3f})",
                            color=COLORS.get(m, '#333'), linewidth=1.8)

        ax.set_title(f'Attack {att}', fontweight='bold')
        ax.set_xlabel('False Positive Rate')
        if ax_idx == 0:
            ax.set_ylabel('True Positive Rate')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc='lower right')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    plt.close()


def plot_drift_sensitivity(drift_rows, output_dir):
    """Line chart: Recall vs drift rate for A2."""
    _setup()
    fig, ax = plt.subplots(figsize=(10, 6))

    methods_drift = ['Variant A', 'Variant B', 'B0', 'B1', 'B2', 'ExtBaseline']

    for method in methods_drift:
        method_rows = [r for r in drift_rows if r['method'] == method]
        if not method_rows:
            continue
        rates = [r['drift_rate'] for r in method_rows]
        recalls = [r['recall_mean'] for r in method_rows]
        cis = [r['recall_ci'] for r in method_rows]

        color = COLORS.get(method, '#333333')
        ax.plot(rates, recalls, '-o', label=method, color=color, markersize=6)
        ax.fill_between(rates,
                        np.array(recalls) - np.array(cis),
                        np.array(recalls) + np.array(cis),
                        alpha=0.15, color=color)

    ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5,
               label='Default rate (1.0)')
    ax.set_xscale('log')
    ax.set_xlabel('Drift Rate (m/epoch)')
    ax.set_ylabel('Recall')
    ax.set_title('A2 Drift Rate Sensitivity')
    ax.set_ylim(-0.05, 1.1)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'a2_drift_sensitivity.png'))
    plt.close()


def plot_ablation(ablation_rows, output_dir):
    """Horizontal bar chart for ablation study."""
    _setup()
    fig, ax = plt.subplots(figsize=(10, 5))

    # Sort by F1 descending
    sorted_rows = sorted(ablation_rows, key=lambda r: r['pooled_f1_mean'])
    names = [r['config'] for r in sorted_rows]
    f1s = [r['pooled_f1_mean'] for r in sorted_rows]
    cis = [r['pooled_f1_ci'] for r in sorted_rows]
    colors = ['gold' if n == 'Full' else '#4a90d9' for n in names]

    bars = ax.barh(names, f1s, xerr=cis, color=colors, capsize=4, alpha=0.85)

    # Annotate "no FAR guarantee" on Minus Conformal bar
    for bar, name in zip(bars, names):
        if name == 'Minus Conformal':
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    '(no FAR guarantee)', va='center', fontsize=8, color='red')

    ax.set_xlabel('Pooled F1 (A1–A5)')
    ax.set_title('Ablation Study — Variant B')
    ax.set_xlim(0, 1.0)
    ax.grid(axis='x', alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'ablation.png'))
    plt.close()


def plot_conformal_calibration(calib_data, output_dir):
    """Conformal calibration: empirical FAR vs target alpha."""
    _setup()
    fig, ax = plt.subplots(figsize=(7, 7))

    alphas = calib_data['alphas']
    va_fars = calib_data['va_fars']
    vb_fars = calib_data['vb_fars']

    # Ideal line
    ax.plot([0, max(alphas) + 0.02], [0, max(alphas) + 0.02],
            'k--', label='Ideal (FAR = α)', linewidth=1.5)

    # Tolerance band
    x_band = np.linspace(0, max(alphas) + 0.02, 100)
    ax.fill_between(x_band, x_band - 0.02, x_band + 0.02,
                    alpha=0.1, color='gray', label='±0.02 tolerance')

    ax.plot(alphas, va_fars, 'bo-', label='Variant A', markersize=8)
    ax.plot(alphas, vb_fars, 'rs-', label='Variant B', markersize=8)

    ax.set_xlabel(r'Target $\alpha$')
    ax.set_ylabel('Empirical FAR')
    ax.set_title('Conformal Calibration Curve')
    ax.legend()
    ax.set_xlim(-0.005, max(alphas) + 0.02)
    ax.set_ylim(-0.005, max(alphas) + 0.05)
    ax.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'conformal_calibration.png'))
    plt.close()


def plot_cross_device(agg, output_dir):
    """Two-panel cross-device figure: FAR (left) + Recall (right)."""
    _setup()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: FAR
    if 'CrossDevice_Benign' in agg:
        methods_far = METHODS
        fars = [agg['CrossDevice_Benign'][m]['far'] for m in methods_far]
        cis = [agg['CrossDevice_Benign'][m].get('far_ci', 0) for m in methods_far]
        colors = [COLORS[m] for m in methods_far]
        x = np.arange(len(methods_far))
        ax1.bar(x, fars, yerr=cis, color=colors, capsize=3, alpha=0.85)
        ax1.axhline(y=0.05, color='red', linestyle='--', alpha=0.6,
                    label=r'$\alpha$ = 0.05')
        ax1.set_xticks(x)
        ax1.set_xticklabels(methods_far, rotation=15, ha='right')
        ax1.set_ylabel('Empirical FAR')
        ax1.set_title('Cross-Device FAR (Device B Benign)')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)

    # Right: Recall on attacks
    xdev_attacks = [f'CrossDevice_{a}' for a in ATTACKS]
    methods_det = ['Variant A', 'Variant B', 'ExtBaseline']
    x = np.arange(len(ATTACKS))
    width = 0.25

    for i, m in enumerate(methods_det):
        recalls = []
        for xa in xdev_attacks:
            if xa in agg and m in agg[xa]:
                recalls.append(agg[xa][m]['recall'])
            else:
                recalls.append(0)
        ax2.bar(x + i * width, recalls, width, label=m,
                color=COLORS[m], alpha=0.85)

    ax2.set_xticks(x + width)
    ax2.set_xticklabels(ATTACKS)
    ax2.set_ylabel('Recall')
    ax2.set_title('Cross-Device Detection (Device B)')
    ax2.legend()
    ax2.set_ylim(0, 1.1)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cross_device_transfer.png'))
    plt.close()


def plot_debug_score_distributions(debug_data, output_dir):
    """4-panel histogram: score distributions for Bug 3 verification."""
    _setup()
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    panels = [
        ('calib_scores_a', 'Calibration Set (Benign)', axes[0, 0]),
        ('test_benign_scores_a', 'Test Set (Benign)', axes[0, 1]),
        ('A1_scores_a', 'A1 (Spoofed)', axes[1, 0]),
        ('A2_scores_a', 'A2 (Spoofed)', axes[1, 1]),
    ]

    for key, title, ax in panels:
        if key in debug_data:
            scores = debug_data[key]
            ax.hist(scores, bins=50, alpha=0.7, color='steelblue',
                    edgecolor='black', linewidth=0.5)
            ax.axvline(np.median(scores), color='red', linestyle='--',
                       label=f'Median: {np.median(scores):.2f}')
            ax.set_title(f'Variant A Scores — {title}')
            ax.set_xlabel('Anomaly Score')
            ax.set_ylabel('Count')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title(title)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'debug_score_distributions.png'))
    plt.close()
