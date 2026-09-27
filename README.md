# ShadowLoc: Multi-Modal GNSS Spoofing Detection

A complete, reproducible implementation of **ShadowLoc**, a multi-modal GNSS spoofing detection system for mobile devices based on the *Uncontrollable-Witness Principle*.

An attacker can transmit spoofed radio-frequency GNSS signals to falsify perceived satellite position, velocity, and timing. However, they cannot concurrently control the physical inertia measured by the device's IMU, the spatial network geometry of terrestrial radio beacons (Wi-Fi/Cellular), or local clock dynamics. ShadowLoc leverages these uncontrollable physical witnesses to detect stealthy spoofing attacks—including gradual carry-off drift where GNSS signals appear completely healthy.

---

## Key Architecture & Components

1. **Hybrid Data Provenance**:
   - **Real GNSS Backbone**: Raw GNSS pseudorange/carrier statistics and real-world trajectory dynamics derived from the [Google Smartphone Decimeter Challenge (GSDC)](https://www.kaggle.com/c/google-smartphone-decimeter-challenge) (Pixel 4 and Pixel 4 XL traces).
   - **Physics-Coupled Witnesses**: Synthetic companion IMU dead-reckoning ($d_{\text{imu}}, \omega_{\text{imu}}$), radio context ($j_{\text{wifi}}, j_{\text{cell}}, \rho_{\text{rssi}}$), and clock bias ($\Delta\tau$) coupled directly to ground truth motion. See [`DATASETS.md`](DATASETS.md) for full provenance.

2. **Attack Models (`data_gen/attacks.py`)**:
   - **A1 — Step Jump**: Sudden positional discontinuity (50 m offset).
   - **A2 — Carry-Off Drift**: Stealthy linear divergence ($v_{\text{drift}} \in [0.1, 5.0]\text{ m/epoch}$) mimicking true vehicle dynamics.
   - **A3 — Intermittent**: Periodic pulse jamming and spoofing bursts (30 epochs on, 30 off).
   - **A4 — Time-Bias**: Satellite clock manipulation inducing position/timing inconsistency.
   - **A5 — Adaptive Adversary**: Intelligent adversary that attempts to constrain trajectory drift while matching expected kinematic profiles.

3. **Detector Variants (`models/`)**:
   - **Variant A**: Unsupervised multivariate distance estimator (Mahalanobis / EllipticEnvelope) over all 16 multi-modal features.
   - **Variant B**: Unsupervised sequence-to-sequence LSTM Autoencoder reconstructing normal multi-sensor dynamics.
   - **B0 Baseline**: Heuristic physics sub-detectors ($P_{\text{rf}}$ interference, $P_t$ clock discrepancy, $r_{\text{mc}}$ motion consistency).
   - **B1 Baseline**: Supervised GNSS-only Random Forest classifier.
   - **B2 Baseline**: Unsupervised GNSS-only LSTM Autoencoder.
   - **ExtBaseline**: Re-implementation of Bhatti et al. (2020) One-Class SVM on Doppler, C/N0, and AGC.
   - **Fusion**: Learned cross-modal logistic regression combining conformal ML anomaly scores with physics sub-detectors without requiring naive conditional independence assumptions.

4. **Split Conformal Calibration (`conformal/calibration.py`)**:
   - Bounded false alarm guarantee: $P(\text{Alarm} \mid H_0) \le \alpha$ for chosen significance $\alpha \in \{0.01, 0.02, 0.05, 0.10\}$.

---

## Directory Structure

```text
shadowloc/
├── conformal/           # Inductive split conformal prediction layer
├── data/raw/gsdc/       # Cached GSDC multi-device trajectory traces
├── data_gen/            # Attack injection, GSDC loader, and hybrid synthesizer
├── evaluation/          # Experiment harness, drift sweep, ablation, and profiler
├── features/            # Sliding-window feature extractor (16 channels)
├── figures/             # Generated publication figures
├── models/              # Variant A, Variant B, B0-B2, ExtBaseline, Fusion
├── plot/                # Publication figure generation scripts
├── results/             # Raw per-seed CSVs, summaries, and metrics JSON
├── DATASETS.md          # Complete provenance and modality breakdown
├── THEORY.md            # Detectability and detection delay proofs
├── requirements.txt     # Python package requirements
├── main.py              # Master pipeline orchestrator
└── README.md
```

---

## Quickstart & Reproducibility

### 1. Installation

```bash
git clone https://github.com/hx010207/Shadowloc.git
cd Shadowloc
pip install -r requirements.txt
```

### 2. Run Complete Evaluation

Execute the entire multi-seed evaluation pipeline, drift rate sweep, ablation study, calibration analysis, and figure rendering:

```bash
python main.py
```

### 3. Generated Artifacts

- `results/summary_by_attack.csv`: Pooled and per-attack metrics (F1, AUC, Precision, Recall, FAR, Delay) with 95% confidence intervals across all seeds.
- `results/a2_drift_sensitivity.csv`: Sweep of A2 drift rates validating Proposition 2 ($D = \mathcal{O}(1/v_{\text{drift}})$).
- `results/ablation.csv`: 6-configuration ablation isolating modality contributions.
- `results/conformal_calibration.csv`: Empirical false alarm rates vs. target $\alpha$.
- `results/cross_device_*.csv`: Cross-device transferability results (Pixel 4 $\to$ Pixel 4 XL).
- `figures/*.png`: Publication-ready figures.
- `results/RESULTS_SUMMARY.md`: Formatted tables matching publication standards.
