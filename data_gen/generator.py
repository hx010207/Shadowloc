import numpy as np
import pandas as pd

def generate_benign_trace(num_epochs=3600, fs=1.0, seed=None):
    """
    Generate a realistic synthetic multi-modal benign trace.
    fs: sampling rate in Hz (default 1 Hz).
    Returns a pandas DataFrame.
    """
    if seed is not None:
        np.random.seed(seed)

    # 1. Regimes: Alternate between "stationary" and "walking"
    # Let's say regimes change roughly every 5-10 minutes.
    regime = np.zeros(num_epochs)
    current_regime = 0 # 0 = stationary, 1 = walking
    switch_probs = 1.0 / (600 * fs) # avg 10 mins per regime
    for i in range(1, num_epochs):
        if np.random.rand() < switch_probs:
            current_regime = 1 - current_regime
        regime[i] = current_regime
        
    # True trajectory (2D for simplicity, altitude is roughly constant)
    true_lat = np.zeros(num_epochs)
    true_lon = np.zeros(num_epochs)
    true_speed = np.zeros(num_epochs)
    true_bearing = np.zeros(num_epochs)
    
    # Start coordinates (approximate)
    lat0, lon0 = 37.7749, -122.4194
    earth_radius = 6378137.0 # meters
    
    # Generate True Trajectory
    for i in range(1, num_epochs):
        if regime[i] == 1:
            # Walking: speed around 1.0 - 1.5 m/s
            true_speed[i] = np.clip(true_speed[i-1] + np.random.normal(0, 0.1), 0.5, 2.0)
            # Bearing: slow random walk
            true_bearing[i] = (true_bearing[i-1] + np.random.normal(0, 5.0)) % 360
        else:
            # Stationary
            true_speed[i] = 0.0
            true_bearing[i] = true_bearing[i-1]
            
        # Update true position based on true speed and bearing
        dt = 1.0 / fs
        dist = true_speed[i] * dt
        dlat = dist * np.cos(np.radians(true_bearing[i])) / earth_radius
        dlon = dist * np.sin(np.radians(true_bearing[i])) / (earth_radius * np.cos(np.radians(lat0)))
        
        true_lat[i] = true_lat[i-1] + np.degrees(dlat)
        true_lon[i] = true_lon[i-1] + np.degrees(dlon)

    # Add GNSS Receiver Noise (Random walk + Gaussian)
    gnss_pos_noise = np.cumsum(np.random.normal(0, 0.05, (num_epochs, 2)), axis=0) + np.random.normal(0, 2.0, (num_epochs, 2))
    # Convert noise to degrees
    gnss_lat = true_lat + lat0 + gnss_pos_noise[:, 0] / earth_radius * 180 / np.pi
    gnss_lon = true_lon + lon0 + gnss_pos_noise[:, 1] / (earth_radius * np.cos(np.radians(lat0))) * 180 / np.pi
    
    gnss_speed = np.clip(true_speed + np.random.normal(0, 0.2, num_epochs), 0, None)
    gnss_bearing = (true_bearing + np.random.normal(0, 3.0, num_epochs)) % 360
    
    # GNSS Acceleration proxy (dv/dt)
    gnss_accel = np.append([0], np.diff(gnss_speed) * fs)

    # 2. Signal Stats (C/N0 and AGC)
    # Simulate 6-12 visible satellites
    n_sat = np.random.randint(6, 13, num_epochs)
    
    # C/N0 baseline: mean ~ 35-45 dB-Hz, slow drift
    base_cn0 = 40.0 + np.cumsum(np.random.normal(0, 0.02, num_epochs))
    mean_cn0 = base_cn0 + np.random.normal(0, 1.0, num_epochs)
    std_cn0 = np.random.uniform(2.0, 5.0, num_epochs) + np.random.normal(0, 0.2, num_epochs)
    
    # AGC level: arbitrary units, normally distributed with slow drift
    # Let's say baseline AGC is around 50
    base_agc = 50.0 + np.cumsum(np.random.normal(0, 0.01, num_epochs))
    agc = base_agc + np.random.normal(0, 0.5, num_epochs)
    
    # 3. IMU (Dead-reckoned displacement and integrated turn rate)
    # In benign, IMU matches true trajectory with small noise
    dt = 1.0 / fs
    d_imu = true_speed * dt + np.random.normal(0, 0.05, num_epochs)
    d_imu = np.clip(d_imu, 0, None)
    
    # turn rate: derivative of true bearing + noise
    omega_imu = np.append([0], np.diff(true_bearing)) 
    # Wrap to -180, 180
    omega_imu = (omega_imu + 180) % 360 - 180
    omega_imu = omega_imu * fs + np.random.normal(0, 1.0, num_epochs)

    # 4. Radio Context (Wi-Fi/Cell BSSID Jaccard, RSSI drift)
    # Jaccard should be high when stationary, slightly lower when walking
    j_wifi = np.where(regime == 0, 
                      np.random.uniform(0.9, 1.0, num_epochs), 
                      np.random.uniform(0.7, 0.95, num_epochs))
    j_cell = np.where(regime == 0, 
                      np.random.uniform(0.95, 1.0, num_epochs), 
                      np.random.uniform(0.8, 1.0, num_epochs))
    # RSSI drift: near zero
    rho_rssi = np.random.normal(0, 1.0, num_epochs)

    # 5. Clock (GNSS vs network time offset)
    # Near zero, small noise
    delta_tau = np.random.normal(0, 0.01, num_epochs)
    # Add some slow clock drift
    delta_tau += np.cumsum(np.random.normal(0, 0.0001, num_epochs))

    df = pd.DataFrame({
        'time_sec': np.arange(num_epochs) / fs,
        'regime': regime,
        'true_lat': true_lat + lat0,
        'true_lon': true_lon + lon0,
        'gnss_lat': gnss_lat,
        'gnss_lon': gnss_lon,
        'gnss_speed': gnss_speed,
        'gnss_bearing': gnss_bearing,
        'gnss_accel': gnss_accel,
        'n_sat': n_sat,
        'mean_cn0': mean_cn0,
        'std_cn0': std_cn0,
        'agc': agc,
        'd_imu': d_imu,
        'omega_imu': omega_imu,
        'j_wifi': j_wifi,
        'j_cell': j_cell,
        'rho_rssi': rho_rssi,
        'delta_tau': delta_tau
    })
    
    return df

if __name__ == "__main__":
    df = generate_benign_trace(num_epochs=100)
    print("Generated shape:", df.shape)
    print(df.head())
