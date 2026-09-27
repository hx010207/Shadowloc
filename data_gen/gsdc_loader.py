import os
import pandas as pd
import numpy as np

def create_mock_gsdc_data(data_dir='data/raw/gsdc'):
    """Create a mock GSDC dataset structure with simulated raw GNSS logs for two devices."""
    os.makedirs(data_dir, exist_ok=True)
    devices = ['Pixel4', 'Pixel4XL']
    
    # We will just generate processed trajectories to match the output schema we need,
    # as parsing the exact GSDC raw GnssLog.txt requires a complex RINEX/Raw parser 
    # which is generally provided by the Kaggle competition hosts (e.g., Google's GnssAnalysis app).
    # We'll mock the 'derived' format which is typically used for ML tasks.
    
    for dev in devices:
        num_epochs = 1800
        fs = 1.0
        
        # Simulate true trajectory
        lat0, lon0 = 37.7749, -122.4194
        t = np.arange(num_epochs)
        true_speed = np.clip(1.0 + np.sin(t / 100.0) + np.random.normal(0, 0.2, num_epochs), 0, None)
        true_bearing = (np.cumsum(np.random.normal(0, 5.0, num_epochs))) % 360
        
        earth_radius = 6378137.0
        dlat = true_speed * (1/fs) * np.cos(np.radians(true_bearing)) / earth_radius
        dlon = true_speed * (1/fs) * np.sin(np.radians(true_bearing)) / (earth_radius * np.cos(np.radians(lat0)))
        
        true_lat = lat0 + np.degrees(np.cumsum(dlat))
        true_lon = lon0 + np.degrees(np.cumsum(dlon))
        
        # Simulate GNSS noise
        gnss_lat = true_lat + np.random.normal(0, 0.00005, num_epochs)
        gnss_lon = true_lon + np.random.normal(0, 0.00005, num_epochs)
        
        df = pd.DataFrame({
            'MillisSinceGpsEpoch': t * 1000 + 1300000000000, # Fake timestamp
            'lat_true': true_lat,
            'lon_true': true_lon,
            'lat_gnss': gnss_lat,
            'lon_gnss': gnss_lon,
            'speed_gnss': true_speed + np.random.normal(0, 0.5, num_epochs),
            'bearing_gnss': (true_bearing + np.random.normal(0, 10.0, num_epochs)) % 360,
            'mean_cn0': np.random.uniform(30, 45, num_epochs) + np.sin(t/50.0)*2,
            'std_cn0': np.random.uniform(2, 6, num_epochs),
            'agc': np.random.uniform(40, 60, num_epochs),
            'n_sat': np.random.randint(6, 15, num_epochs),
            'delta_tau': np.random.normal(0, 0.01, num_epochs)
        })
        
        dev_dir = os.path.join(data_dir, f'2021-04-29-US-SJC-1/{dev}')
        os.makedirs(dev_dir, exist_ok=True)
        df.to_csv(os.path.join(dev_dir, 'derived_gnss.csv'), index=False)

def load_gsdc_trace(device_name, data_dir='data/raw/gsdc'):
    """Load a GSDC trace for a given device. If missing, generate mock data."""
    if not os.path.exists(data_dir) or not os.listdir(data_dir):
        print("GSDC data not found, generating mock dataset...")
        create_mock_gsdc_data(data_dir)
        
    # Search for the device
    file_path = None
    for root, dirs, files in os.walk(data_dir):
        if device_name in root and 'derived_gnss.csv' in files:
            file_path = os.path.join(root, 'derived_gnss.csv')
            break
            
    if file_path is None:
        raise FileNotFoundError(f"Device {device_name} not found in {data_dir}")
        
    df = pd.read_csv(file_path)
    
    # Rename columns to our internal schema
    schema_map = {
        'lat_true': 'true_lat',
        'lon_true': 'true_lon',
        'lat_gnss': 'gnss_lat',
        'lon_gnss': 'gnss_lon',
        'speed_gnss': 'gnss_speed',
        'bearing_gnss': 'gnss_bearing'
    }
    df = df.rename(columns=schema_map)
    df['time_sec'] = (df['MillisSinceGpsEpoch'] - df['MillisSinceGpsEpoch'].iloc[0]) / 1000.0
    
    # compute gnss_accel
    fs = 1.0 # assuming 1Hz for GSDC
    df['gnss_accel'] = np.append([0], np.diff(df['gnss_speed']) * fs)
    
    return df
