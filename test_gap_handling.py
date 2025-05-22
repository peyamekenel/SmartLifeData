import pandas as pd
import numpy as np
from energy_analysis import calculate_time_weighted_consumption, CONFIG

def test_gap_handling():
    """Test the gap handling logic with synthetic data."""
    print("Testing gap handling with synthetic data...")
    
    timestamps = [
        pd.Timestamp("2025-03-10 12:00:00"),
        pd.Timestamp("2025-03-11 12:00:00"),
        pd.Timestamp("2025-03-12 12:00:00"),
        pd.Timestamp("2025-03-18 12:00:00")
    ]
    
    energy_values = [100.0, 101.0, 102.0, 112.0]
    
    df = pd.DataFrame({
        "timestamp": timestamps,
        "energy": energy_values,
        "device_id": ["test_device_1"] * 4,
        "device_class": ["DB16"] * 4
    })
    
    df["energy_diff"] = df["energy"].diff()
    df["time_diff"] = df["timestamp"].diff().dt.total_seconds() / 3600  # in hours
    
    print("\nDataFrame with differences:")
    print(df)
    
    if "anomaly_detection" not in CONFIG:
        CONFIG["anomaly_detection"] = {}
    original_jump_threshold = CONFIG["anomaly_detection"].get("jump_threshold_factor", 2.0)
    CONFIG["anomaly_detection"]["jump_threshold_factor"] = 1.5  # Lower threshold to ensure normalization
    
    print("\nTesting with uniform distribution:")
    consumption, duration, anomalies = calculate_time_weighted_consumption(
        df, max_gap_hours=24, device_id="test_device_1"
    )
    
    print(f"Total consumption: {consumption:.2f} kWh")
    print(f"Duration: {duration:.2f} hours")
    print(f"Anomalies detected: {len(anomalies)}")
    for a in anomalies:
        print(f"  {a['anomaly_type']}: {a['original_value']:.2f} -> {a['adjusted_value']:.2f}")
    
    expected_consumption = 2.0 + 6.0  # 2 kWh from normal days + normalized 6 kWh from gap
    print(f"Expected consumption: ~{expected_consumption:.2f} kWh")
    
    is_reasonable = abs(consumption - expected_consumption) < 1.0
    print(f"Results reasonable: {is_reasonable}")
    
    print("\nTesting with time-weighted distribution:")
    original_setting = CONFIG.get("anomaly_detection", {}).get("use_time_weighted_distribution", False)
    if "anomaly_detection" not in CONFIG:
        CONFIG["anomaly_detection"] = {}
    CONFIG["anomaly_detection"]["use_time_weighted_distribution"] = True
    
    consumption_tw, duration_tw, anomalies_tw = calculate_time_weighted_consumption(
        df, max_gap_hours=24, device_id="test_device_1"
    )
    
    CONFIG["anomaly_detection"]["use_time_weighted_distribution"] = original_setting
    CONFIG["anomaly_detection"]["jump_threshold_factor"] = original_jump_threshold
    
    print(f"Total consumption: {consumption_tw:.2f} kWh")
    print(f"Duration: {duration_tw:.2f} hours")
    print(f"Anomalies detected: {len(anomalies_tw)}")
    for a in anomalies_tw:
        print(f"  {a['anomaly_type']}: {a['original_value']:.2f} -> {a['adjusted_value']:.2f}")
    
    is_reasonable_tw = abs(consumption_tw - expected_consumption) < 1.0
    print(f"Results reasonable: {is_reasonable_tw}")
    
    print("\nTest completed.")

if __name__ == "__main__":
    test_gap_handling()
