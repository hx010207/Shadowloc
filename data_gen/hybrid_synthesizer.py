import numpy as np
import pandas as pd
from data_gen.attacks import haversine_dist

def synthesize_hybrid_modalities(df, fs=1.0):
    """
    Takes a dataframe containing raw GNSS and Ground Truth trajectories,
    and synthesizes statistically consistent IMU, Radio, and Clock context.
    """
    num_epochs = len(df)
    
    # 1. IMU Synthesis (Derived from TRUE trajectory)
    # Re-calculate true speed and turn rate if ground truth speed is not provided directly
    true_lat = df['true_lat'].values
    true_lon = df['true_lon'].values
    
    true_d_pos = np.zeros(num_epochs)
    true_d_pos[1:] = haversine_dist(true_lat[:-1], true_lon[:-1], true_lat[1:], true_lon[1:])
    
    true_speed = true_d_pos * fs
    
    # IMU displacement = true_displacement + sensor noise
    df['d_imu'] = np.clip(true_d_pos + np.random.normal(0, 0.05, num_epochs), 0, None)
    
    # Approximate true bearing from true positions
    true_bearing = np.zeros(num_epochs)
    for i in range(1, num_epochs):
        lat1, lon1 = np.radians(true_lat[i-1]), np.radians(true_lon[i-1])
        lat2, lon2 = np.radians(true_lat[i]), np.radians(true_lon[i])
        dLon = lon2 - lon1
        y = np.sin(dLon) * np.cos(lat2)
        x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dLon)
        brng = np.degrees(np.arctan2(y, x))
        true_bearing[i] = (brng + 360) % 360
        
    true_turn_rate = np.append([0], np.diff(true_bearing))
    true_turn_rate = (true_turn_rate + 180) % 360 - 180
    
    df['omega_imu'] = true_turn_rate * fs + np.random.normal(0, 1.0, num_epochs)
    
    # 2. Radio Context (Wi-Fi/Cell Jaccard, RSSI)
    # Jaccard should be high when stationary (speed < 0.5), lower when moving
    is_stationary = true_speed < 0.5
    df['j_wifi'] = np.where(is_stationary, 
                            np.random.uniform(0.9, 1.0, num_epochs), 
                            np.random.uniform(0.7, 0.95, num_epochs))
    df['j_cell'] = np.where(is_stationary, 
                            np.random.uniform(0.95, 1.0, num_epochs), 
                            np.random.uniform(0.8, 1.0, num_epochs))
    df['rho_rssi'] = np.random.normal(0, 1.0, num_epochs)
    
    # 3. If delta_tau not present in GSDC log (clock offset), simulate it
    if 'delta_tau' not in df.columns:
        df['delta_tau'] = np.random.normal(0, 0.01, num_epochs) + np.cumsum(np.random.normal(0, 0.0001, num_epochs))
        
    # Mark real vs synthetic channels in metadata or conceptually
    # Real: GNSS, Timing (if from GSDC)
    # Synthetic: IMU, Radio
    
    return df
