"""
A2 drift rate sensitivity analysis for ShadowLoc.

Tests detection performance at multiple carry-off drift rates
to validate the cross-modal advantage hypothesis: at low drift
rates where GNSS signal statistics look normal, ShadowLoc should
outperform GNSS-only methods via IMU-GNSS divergence detection.
"""

import numpy as np
from tqdm import tqdm
from evaluation.harness import run_experiment_drift, set_seed


DRIFT_RATES = [0.1, 0.5, 1.0, 2.0, 5.0]
METHODS = ['Variant A', 'Variant B', 'B0', 'B1', 'B2', 'ExtBaseline']


def run_drift_sensitivity(num_seeds=30, drift_rates=None):
    """Run A2 at multiple drift rates across all seeds.

    Returns: dict keyed by drift_rate, containing per-method
    recall and delay lists across seeds.
    """
    if drift_rates is None:
        drift_rates = DRIFT_RATES

    results = {rate: {m: {'recall': [], 'delay': []}
                      for m in METHODS}
               for rate in drift_rates}

    print("Running A2 drift rate sensitivity analysis...")
    for rate in drift_rates:
        print(f"  Drift rate = {rate:.1f} m/epoch")
        for seed in tqdm(range(num_seeds), desc=f"  rate={rate}"):
            set_seed(seed)
            try:
                res = run_experiment_drift(seed, drift_rate=rate)
                for method in METHODS:
                    if method in res:
                        results[rate][method]['recall'].append(
                            res[method]['recall'])
                        results[rate][method]['delay'].append(
                            res[method]['delay'])
            except Exception as e:
                print(f"    Error seed={seed} rate={rate}: {e}")

    return results


def aggregate_drift_results(results):
    """Aggregate drift results into mean ± 95% CI."""
    rows = []
    for rate in sorted(results.keys()):
        for method in METHODS:
            recalls = np.array(results[rate][method]['recall'])
            delays = np.array(results[rate][method]['delay'])

            valid_recalls = recalls[~np.isnan(recalls)]
            valid_delays = delays[~np.isnan(delays)]

            r_mean = np.mean(valid_recalls) if len(valid_recalls) > 0 else 0.0
            r_ci = (1.96 * np.std(valid_recalls) / np.sqrt(len(valid_recalls))
                    if len(valid_recalls) > 1 else 0.0)

            d_mean = np.mean(valid_delays) if len(valid_delays) > 0 else np.nan
            d_ci = (1.96 * np.std(valid_delays) / np.sqrt(len(valid_delays))
                    if len(valid_delays) > 1 else np.nan)

            rows.append({
                'drift_rate': rate,
                'method': method,
                'recall_mean': r_mean,
                'recall_ci': r_ci,
                'delay_mean': d_mean,
                'delay_ci': d_ci if not np.isnan(d_ci) else 0.0,
            })
    return rows
