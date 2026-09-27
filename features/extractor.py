import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from data_gen.attacks import haversine_dist

class FeatureExtractor:
    def __init__(self, W=10, stride=1, fs=1.0):
        self.W = W
        self.stride = stride
        self.fs = fs
        self.scaler = StandardScaler()
        self.is_fitted = False

    def extract_raw_features(self, df):
        """Extract pointwise features for the entire dataframe."""
        df_feat = pd.DataFrame(index=df.index)
        
        # GNSS dynamics
        lat = df['gnss_lat'].values
        lon = df['gnss_lon'].values
        
        # d_pos (Δposition)
        d_pos = np.zeros(len(df))
        d_pos[1:] = haversine_dist(lat[:-1], lon[:-1], lat[1:], lon[1:])
        df_feat['d_pos'] = d_pos
        
        df_feat['d_speed'] = df['gnss_speed'].diff().fillna(0).abs()
        
        # d_bearing
        d_bearing = df['gnss_bearing'].diff().fillna(0)
        # wrap to -180, 180
        d_bearing = (d_bearing + 180) % 360 - 180
        df_feat['d_bearing'] = d_bearing.abs()
        
        df_feat['a_gnss'] = df['gnss_accel'].abs()
        
        # Signal stats
        df_feat['mean_cn0'] = df['mean_cn0']
        df_feat['std_cn0'] = df['std_cn0']
        df_feat['n_sat'] = df['n_sat']
        
        # Interference
        df_feat['agc'] = df['agc']
        df_feat['agc_slope'] = df['agc'].diff().fillna(0) * self.fs
        
        # Timing
        df_feat['delta_tau'] = df['delta_tau']
        df_feat['delta_tau_rate'] = df['delta_tau'].diff().fillna(0) * self.fs
        
        # Inertial
        df_feat['d_imu'] = df['d_imu']
        df_feat['omega_imu'] = df['omega_imu'].abs()
        
        # Radio context
        df_feat['j_wifi'] = df['j_wifi']
        df_feat['j_cell'] = df['j_cell']
        df_feat['rho_rssi'] = df['rho_rssi']
        
        # These are 16 features.
        return df_feat

    def window_features(self, df_feat):
        """Convert pointwise features to overlapping windows of size W."""
        # For simplicity, we just flatten the window. Or keep it as (N, W, D) for LSTM
        # and we can flatten it for Mahalanobis.
        # We will return (N_windows, W, D) and (N_windows, D_flat).
        
        data = df_feat.values
        N, D = data.shape
        
        # Calculate number of windows
        num_windows = (N - self.W) // self.stride + 1
        
        windows = np.zeros((num_windows, self.W, D))
        for i in range(num_windows):
            start = i * self.stride
            windows[i] = data[start : start + self.W]
            
        return windows

    def fit(self, df_benign):
        """Fit scaler on benign training data ONLY. Fits on pointwise features."""
        df_feat = self.extract_raw_features(df_benign)
        self.scaler.fit(df_feat.values)
        self.is_fitted = True

    def transform(self, df):
        """Extract, scale, and window."""
        if not self.is_fitted:
            raise ValueError("Scaler not fitted. Call fit() on benign data first.")
            
        df_feat = self.extract_raw_features(df)
        scaled_data = self.scaler.transform(df_feat.values)
        
        # Put back into DF to use window_features
        df_scaled = pd.DataFrame(scaled_data, columns=df_feat.columns, index=df_feat.index)
        windows = self.window_features(df_scaled)
        
        # Create corresponding labels for windows
        # A window is considered spoofed if ANY epoch in the window is spoofed
        # Wait, usually if the LAST epoch is spoofed, or ANY. Let's use ANY to be safe, 
        # or we just take the label of the last epoch in the window. The delay is counted 
        # from the first spoofed epoch. Let's use the label of the last epoch in the window for causal detection.
        
        is_spoofed_pt = df['is_spoofed'].values if 'is_spoofed' in df.columns else np.zeros(len(df), dtype=bool)
        
        N = len(df)
        num_windows = (N - self.W) // self.stride + 1
        
        window_labels = np.zeros(num_windows, dtype=bool)
        window_delta_pos = np.zeros(num_windows)
        
        if 'is_spoofed' in df.columns:
            for i in range(num_windows):
                start = i * self.stride
                # Spoofed if any epoch in window is spoofed
                window_labels[i] = np.any(is_spoofed_pt[start : start + self.W])
                # Max delta pos in window
                window_delta_pos[i] = np.max(df['spoofed_delta_pos'].values[start : start + self.W])
                
        return windows, window_labels, window_delta_pos
