import numpy as np
import pandas as pd

def haversine_dist(lat1, lon1, lat2, lon2):
    """Calculate distance in meters between two lat/lon points."""
    R = 6378137.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

def add_rf_artifacts(df, is_spoofed):
    """Add C/N0 increase and AGC drop during spoofing."""
    df.loc[is_spoofed, 'mean_cn0'] += np.random.uniform(5.0, 10.0, sum(is_spoofed))
    df.loc[is_spoofed, 'std_cn0'] = np.clip(df.loc[is_spoofed, 'std_cn0'] - np.random.uniform(1.0, 2.0, sum(is_spoofed)), 0.5, None)
    df.loc[is_spoofed, 'agc'] -= np.random.uniform(5.0, 15.0, sum(is_spoofed))
    return df

def enforce_threat_model(df_in, df_out):
    """Strictly assert that IMU and Radio channels remain bit-identical."""
    protected_cols = ['d_imu', 'omega_imu', 'j_wifi', 'j_cell', 'rho_rssi']
    for col in protected_cols:
        if col in df_in.columns and col in df_out.columns:
            np.testing.assert_array_equal(df_in[col].values, df_out[col].values, err_msg=f"Threat model violated: {col} was modified!")

def attack_wrapper(func):
    def wrapper(df_in, *args, **kwargs):
        df_out = func(df_in, *args, **kwargs)
        enforce_threat_model(df_in, df_out)
        return df_out
    return wrapper

@attack_wrapper
def a1_jump(df_in, t_start=None, jump_distance=500.0, fs=1.0):
    df = df_in.copy()
    num_epochs = len(df)
    if t_start is None:
        t_start = int(num_epochs * 0.3)
        
    is_spoofed = np.zeros(num_epochs, dtype=bool)
    is_spoofed[t_start:] = True
    
    # Jump by translating all points after t_start
    lat_shift = (jump_distance / 6378137.0) * (180 / np.pi)
    
    df.loc[is_spoofed, 'gnss_lat'] += lat_shift
    # Add a huge speed spike at the transition epoch to simulate the jump dynamics
    if t_start < num_epochs:
        df.loc[t_start, 'gnss_speed'] = jump_distance * fs
        df.loc[t_start, 'gnss_accel'] = jump_distance * fs * fs
    
    df = add_rf_artifacts(df, is_spoofed)
    df['is_spoofed'] = is_spoofed
    
    df['spoofed_delta_pos'] = 0.0
    df.loc[is_spoofed, 'spoofed_delta_pos'] = jump_distance
    
    return df

@attack_wrapper
def a2_drift(df_in, t_start=None, drift_rate=1.0, fs=1.0):
    """Drift/carry-off attack. Diverges from true path at drift_rate (m/s)."""
    df = df_in.copy()
    num_epochs = len(df)
    if t_start is None:
        t_start = int(num_epochs * 0.3)
        
    is_spoofed = np.zeros(num_epochs, dtype=bool)
    is_spoofed[t_start:] = True
    
    # Calculate accumulated drift distance
    time_since_start = np.zeros(num_epochs)
    time_since_start[t_start:] = np.arange(num_epochs - t_start) / fs
    drift_distance = time_since_start * drift_rate
    
    lat_shift = (drift_distance / 6378137.0) * (180 / np.pi)
    
    df.loc[is_spoofed, 'gnss_lat'] += lat_shift[is_spoofed]
    df.loc[is_spoofed, 'gnss_speed'] += drift_rate # speed is higher by drift rate
    
    df = add_rf_artifacts(df, is_spoofed)
    df['is_spoofed'] = is_spoofed
    df['spoofed_delta_pos'] = drift_distance
    
    return df

@attack_wrapper
def a3_intermittent(df_in, t_start=None, burst_len=30, off_len=60, jump_distance=500.0, fs=1.0):
    df = df_in.copy()
    num_epochs = len(df)
    if t_start is None:
        t_start = int(num_epochs * 0.2)
        
    is_spoofed = np.zeros(num_epochs, dtype=bool)
    
    curr = t_start
    while curr < num_epochs:
        end = min(curr + burst_len, num_epochs)
        is_spoofed[curr:end] = True
        curr += burst_len + off_len
        
    lat_shift = (jump_distance / 6378137.0) * (180 / np.pi)
    df.loc[is_spoofed, 'gnss_lat'] += lat_shift
    
    df = add_rf_artifacts(df, is_spoofed)
    df['is_spoofed'] = is_spoofed
    
    df['spoofed_delta_pos'] = 0.0
    df.loc[is_spoofed, 'spoofed_delta_pos'] = jump_distance
    
    return df

@attack_wrapper
def a4_time_bias(df_in, t_start=None, slope=0.01, fs=1.0):
    """GNSS time offset grows. Network time remains correct. delta_tau = tau_net - tau_gnss"""
    df = df_in.copy()
    num_epochs = len(df)
    if t_start is None:
        t_start = int(num_epochs * 0.3)
        
    is_spoofed = np.zeros(num_epochs, dtype=bool)
    is_spoofed[t_start:] = True
    
    time_since_start = np.zeros(num_epochs)
    time_since_start[t_start:] = np.arange(num_epochs - t_start) / fs
    time_bias = time_since_start * slope
    
    df.loc[is_spoofed, 'delta_tau'] += time_bias[is_spoofed]
    
    df = add_rf_artifacts(df, is_spoofed)
    df['is_spoofed'] = is_spoofed
    df['spoofed_delta_pos'] = 0.0 # Pos doesn't necessarily change in pure time attack
    
    return df

@attack_wrapper
def a5_adaptive(df_in, t_start=None, drift_rate=1.0, fs=1.0):
    """Adaptive Attack (A5): Adversary perturbs GNSS while perfectly matching true local dynamics (IMU plausible).
    The spoofed position drifts globally, but local displacement |d_pos| exactly matches the true displacement,
    so the motion-consistency residual r_mc = |d_pos| - d_imu remains near zero.
    """
    df = df_in.copy()
    num_epochs = len(df)
    if t_start is None:
        t_start = int(num_epochs * 0.3)
        
    is_spoofed = np.zeros(num_epochs, dtype=bool)
    is_spoofed[t_start:] = True
    
    # Adversary uses the true speed (displacement) but applies it in a slightly diverging heading.
    # Drift is achieved by rotating the true velocity vector slightly, keeping magnitude identical.
    true_lat = df['true_lat'].values
    true_lon = df['true_lon'].values
    
    spoofed_lat = df['gnss_lat'].values.copy()
    spoofed_lon = df['gnss_lon'].values.copy()
    
    earth_radius = 6378137.0
    
    divergence_angle = 10.0 # degrees of heading offset
    
    for i in range(t_start, num_epochs):
        # Calculate true local displacement vector from i-1 to i
        dlat_true = true_lat[i] - true_lat[i-1]
        dlon_true = true_lon[i] - true_lon[i-1]
        
        # Apply rotation matrix to the displacement vector
        theta = np.radians(divergence_angle)
        c, s = np.cos(theta), np.sin(theta)
        
        # Simple local cartesian rotation
        dlat_spoof = dlat_true * c - dlon_true * s
        dlon_spoof = dlat_true * s + dlon_true * c
        
        spoofed_lat[i] = spoofed_lat[i-1] + dlat_spoof
        spoofed_lon[i] = spoofed_lon[i-1] + dlon_spoof
        
    df.loc[is_spoofed, 'gnss_lat'] = spoofed_lat[is_spoofed]
    df.loc[is_spoofed, 'gnss_lon'] = spoofed_lon[is_spoofed]
    
    # gnss_speed matches true_speed exactly, thus matching IMU exactly
    df.loc[is_spoofed, 'gnss_speed'] = df['gnss_speed'].values[is_spoofed]
    
    df = add_rf_artifacts(df, is_spoofed)
    df['is_spoofed'] = is_spoofed
    
    # Calculate drift distance (difference between spoofed and true)
    drift_distance = np.zeros(num_epochs)
    drift_distance[is_spoofed] = haversine_dist(
        true_lat[is_spoofed], true_lon[is_spoofed],
        spoofed_lat[is_spoofed], spoofed_lon[is_spoofed]
    )
    df['spoofed_delta_pos'] = drift_distance
    
    return df

