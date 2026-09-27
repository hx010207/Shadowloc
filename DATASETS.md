# DATASETS

This document tracks the provenance, licenses, and hybrid nature of all datasets
used in the ShadowLoc evaluation pipeline.

## 1. Google Smartphone Decimeter Challenge (GSDC)

- **Source**: [Kaggle — Google Smartphone Decimeter Challenge 2021](https://www.kaggle.com/c/google-smartphone-decimeter-challenge)
- **Version**: 2021 competition release
- **License**: Apache 2.0 (via Kaggle terms)
- **Modality**: Raw GNSS measurements (C/N0, pseudoranges, Doppler, position) and high-accuracy Ground Truth
- **Role**: Provides the foundation for all evaluation. Used for real GNSS trajectory and base RF signal statistics.

### Device Traces Used

| Device | Role | Trace Count | Epochs per Trace |
|---|---|---|---|
| Pixel 4 | Device A (train/calib/val/test) | 1 | 1800 |
| Pixel 4 XL | Device B (cross-device transfer) | 1 | 1800 |

**Note**: In the current implementation, GSDC data is simulated via `data_gen/gsdc_loader.py`
using a mock trajectory generator (random walk + sinusoidal CN0 drift). This is because
the full GSDC GnssLog.txt parsing pipeline requires Google's GnssAnalysis toolchain.
The mock data preserves the statistical properties needed for the evaluation but does not
contain real satellite observations.

## 2. Feature Provenance

| Feature | Real or Synthetic | Source |
|---|---|---|
| gnss_lat, gnss_lon | Simulated (mock GSDC) | `gsdc_loader.py` random walk |
| gnss_speed, gnss_bearing | Simulated (mock GSDC) | Derived from trajectory |
| mean_cn0, std_cn0 | Simulated (mock GSDC) | Uniform + sinusoidal drift |
| agc | Simulated (mock GSDC) | Uniform distribution |
| n_sat | Simulated (mock GSDC) | Uniform [6, 15] |
| delta_tau | **SYNTHETIC** | Gaussian N(0, 0.01) with drift |
| d_imu | **SYNTHETIC** | Derived from true trajectory + Gaussian noise (σ=0.05m) |
| omega_imu | **SYNTHETIC** | Derived from true bearing + Gaussian noise (σ=1.0°/s) |
| j_wifi | **SYNTHETIC** | Speed-dependent: stationary [0.9,1.0], moving [0.7,0.95] |
| j_cell | **SYNTHETIC** | Speed-dependent: stationary [0.95,1.0], moving [0.8,1.0] |
| rho_rssi | **SYNTHETIC** | Gaussian N(0, 1.0) |

**IMPORTANT**: All tables and plots containing Variant A, Variant B, or Fusion results
are evaluated on the **Hybrid** dataset (Simulated GNSS + Synthetic IMU/Radio/Clock).
Baselines B1, B2, and ExtBaseline are trained on all features but primarily sensitive
to GNSS-only channels.

## 3. Random Seeds

All 30 seeds used for the evaluation:

```
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
```

Each seed controls:
- IMU/Radio/Clock noise generation in `synthesize_hybrid_modalities()`
- DataFrame row shuffling for IID train/calib/val/test splits
- Model random state for reproducibility (RandomForest, EllipticEnvelope)

## 4. Reproducibility

```bash
python main.py
```

This runs the full 30-seed evaluation and generates all outputs in `results/` and `figures/`.
Expected runtime: 15–25 minutes on a modern desktop CPU.

## 5. Legacy Synthetic Dataset

The pure-synthetic dataset generator (`data_gen/generator.py`) simulates a simple
walking/stationary random walk with full ground-truth multi-modal signals.
- **Role**: Kept for secondary debugging only. Not used in headline metrics.
