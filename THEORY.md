# Theoretical Bounds of ShadowLoc

This document details the rigorous step-by-step proofs for the detectability bounds and the scaling of detection delay in the presence of drift-style (A2) attacks.

## Proposition 1: Detectability Bound via Cross-Modal Consistency

**Hypothesis**: Let $x_k$ be a windowed feature vector containing GNSS dynamics $\Delta p_{gnss}$ and IMU displacement $d_{imu}$. Under benign conditions, the motion-consistency residual $r_{mc} = |\|\Delta p_{gnss}\| - d_{imu}|$ follows a zero-mean sub-Gaussian distribution with variance proxy $\sigma^2$. If an attacker injects a spoofed GNSS trajectory deviating from the true path by a distance $\Delta_{attack}$, the probability of misdetection drops exponentially.

**Proof**:
1. **Benign State**: Under the null hypothesis $H_0$ (no spoofing), $r_{mc}$ is bounded by the sum of GNSS position noise and IMU integration noise. Assuming bounded sensor noise, $r_{mc} \sim \text{sub-Gaussian}(0, \sigma^2)$.
2. **Attacked State**: Under the alternative hypothesis $H_1$, the attacker controls $\Delta p_{gnss}$ independently of the uncontrollable witness $d_{imu}$. The induced residual becomes $r_{mc}' = |\|\Delta p_{gnss} + \Delta_{attack}\| - d_{imu}|$.
3. **Concentration Bound**: By Hoeffding's inequality for sub-Gaussian random variables, the probability that the benign residual exceeds a detection threshold $T$ is bounded by:
   $$ P(r_{mc} > T | H_0) \le \exp\left(-\frac{T^2}{2\sigma^2}\right) $$
   Setting $T$ ensures a bounded False Alarm Rate (FAR) $\le \alpha$.
4. **Detection Guarantee**: If the attack induces a deviation $\Delta_{attack}$ such that the expectation $E[r_{mc}'] > T$, the probability of missed detection is bounded by applying the reverse tail bound:
   $$ P(r_{mc}' < T | H_1) \le \exp\left(-\frac{(E[r_{mc}'] - T)^2}{2\sigma^2}\right) $$
   Thus, as the spoofing divergence $\Delta_{attack}$ grows beyond the noise floor $T$, the detection probability approaches 1 exponentially. $\blacksquare$

## Proposition 2: Detection Delay Scaling for Drift Attacks

**Hypothesis**: For an A2 drift attack where the spoofed trajectory diverges from the true trajectory at a constant rate $v_{drift}$ (m/s), the expected detection delay $D$ (in seconds) scales linearly with the threshold $T$ and inversely with $v_{drift}$.

**Proof**:
1. **Drift Accumulation**: Let the attack commence at $t=0$. At time $t$, the induced spoofing deviation is $\Delta_{attack}(t) = v_{drift} \cdot t$.
2. **Threshold Crossing**: The system alarms when the residual $r_{mc}'(t) > T$. Assuming a worst-case scenario where the attacker exactly mimics the true motion vectors but adds the drift vector, $r_{mc}'(t) \approx v_{drift} \cdot t$.
3. **Delay Calculation**: To achieve $P_{detect} \ge 1 - \beta$ (for a target missed detection rate $\beta$), we require $E[r_{mc}'(D)] - T \ge \sigma \sqrt{2 \ln(1/\beta)}$.
   Substituting the drift model:
   $$ v_{drift} \cdot D - T \ge \sigma \sqrt{2 \ln(1/\beta)} $$
   Solving for $D$:
   $$ D \ge \frac{T + \sigma \sqrt{2 \ln(1/\beta)}}{v_{drift}} $$
   This establishes that detection delay is $\mathcal{O}(1/v_{drift})$, consistent with empirical sensitivity sweeps showing exponentially faster detection at higher drift rates. $\blacksquare$

## Section 3: Multi-Witness Cross-Modal Fusion

**Motivation**: A naive Bayesian product $\text{logit}(P_{fused}) = \text{logit}(\pi) + \sum_{j} \gamma_j \text{logit}(P_j)$ assumes conditional independence among sub-detectors given $H_0$ or $H_1$. In real mobile multi-sensor environments, RF distortion ($P_{rf}$), clock jitter ($P_t$), motion discrepancy ($r_{mc}$), and deep reconstruction loss ($P_{ml}$) exhibit non-negligible cross-correlations.

**Formulation**:
ShadowLoc formulates the cross-modal consensus as a regularized logistic decision surface:
$$ P(\text{Spoof} \mid x) = \sigma\left( \beta_0 + \beta_{ml} P_{ml} + \beta_{rf} P_{rf} + \beta_{t} P_{t} + \beta_{mc} r_{mc} \right) $$
where:
- $P_{ml} = 1 - p(x)$ is the calibrated ML anomaly confidence from split conformal prediction.
- $P_{rf} = \sigma\left(\frac{\Delta C/N_0 - \Delta AGC}{\sigma_{rf}} - b_{rf}\right)$ is the adaptively calibrated RF-layer anomaly score.
- $P_{t} = \sigma\left(\frac{|\Delta\tau|}{\sigma_t} - b_t\right)$ is the adaptively calibrated clock discrepancy score.
- $r_{mc} = |\|\Delta p_{gnss}\| - d_{imu}|$ is the raw motion consistency residual.
- $\boldsymbol{\beta} = [\beta_0, \beta_{ml}, \beta_{rf}, \beta_t, \beta_{mc}]^T$ is estimated via maximum likelihood on a balanced multi-attack validation set.

This learned formulation eliminates the overconfidence artifacts of naive Bayesian independence, automatically dampens redundant channels, and weights reliable witnesses (such as $P_{rf}$ and $P_{ml}$) according to empirical discriminative power.

