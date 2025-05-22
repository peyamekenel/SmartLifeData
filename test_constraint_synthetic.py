"""
Test to verify that total consumption constraint works correctly using synthetic data.
"""
import pandas as pd
import numpy as np
from energy_analysis import per_device_usage, CONFIG, daily_device, log_anomaly

def test_constraint_with_synthetic_data():
    """Test constraint enforcement with synthetic data."""
    print("Testing total constraint with synthetic data...")
    
    if "energy_analysis" not in CONFIG:
        CONFIG["energy_analysis"] = {}
    CONFIG["energy_analysis"]["enforce_total_constraint"] = True
    CONFIG["energy_analysis"]["constraint_tolerance"] = 0.05
    
    device_id = "test_device_1"
    timestamps = pd.date_range(start="2025-03-01", periods=10, freq="D")
    
    first_energy = 100.0
    last_energy = 150.0  # Actual difference: 50.0
    
    energy_values = [
        100.0,  # First reading
        110.0,  # +10
        105.0,  # Reset to 105 (would be -5 without reset detection)
        115.0,  # +10
        125.0,  # +10
        120.0,  # Reset to 120 (would be -5 without reset detection)
        130.0,  # +10
        140.0,  # +10
        145.0,  # +5
        150.0   # +5, Last reading
    ]
    
    
    df = pd.DataFrame({
        "timestamp": timestamps,
        "energy": energy_values,
        "device_id": [device_id] * 10,
        "device_class": ["DB16"] * 10
    })
    
    print("\nSynthetic data:")
    print(df[["timestamp", "energy", "device_id"]])
    
    print(f"\nFirst reading: {first_energy:.2f} kWh")
    print(f"Last reading: {last_energy:.2f} kWh")
    print(f"Actual difference: {last_energy - first_energy:.2f} kWh")
    print(f"Expected unconstrained consumption: 60.00 kWh")
    
    result_df, _, anomalies = per_device_usage(df)
    
    if not result_df.empty:
        calculated_total = result_df[result_df['device_id'] == device_id]['consumption'].values[0]
        print(f"\nCalculated total: {calculated_total:.2f} kWh")
        
        tolerance = CONFIG["energy_analysis"].get("constraint_tolerance", 0.05)
        max_allowed = (last_energy - first_energy) * (1 + tolerance)
        
        print(f"Max allowed (with {tolerance*100:.1f}% tolerance): {max_allowed:.2f} kWh")
        print(f"Constraint enforced: {calculated_total <= max_allowed}")
        
        total_constraint_anomalies = [a for a in anomalies if a.get('anomaly_type') == 'total_constraint']
        print(f"Total constraint anomalies detected: {len(total_constraint_anomalies)}")
        for anomaly in total_constraint_anomalies:
            print(f"  Original: {anomaly['original_value']:.2f} kWh, Adjusted: {anomaly['adjusted_value']:.2f} kWh")
        
        print("\nTesting daily consumption constraint...")
        
        import os
        os.makedirs("csv_output", exist_ok=True)
        
        result_df.to_csv("csv_output/device_usage.csv", index=False)
        
        daily_df, daily_anomalies = daily_device(df)
        
        if not daily_df.empty:
            device_daily = daily_df[daily_df['device_id'] == device_id]
            daily_sum = device_daily['consumption'].sum()
            print(f"Daily consumption sum: {daily_sum:.2f} kWh")
            print(f"Daily sum matches calculated total: {abs(daily_sum - calculated_total) < 0.1}")
            
            daily_constraint_anomalies = [a for a in daily_anomalies if a.get('anomaly_type') == 'daily_constraint']
            print(f"Daily constraint anomalies detected: {len(daily_constraint_anomalies)}")
            
            print("\nSample daily consumption values:")
            print(device_daily.head(5))
        else:
            print("No daily results for device")
    else:
        print("No results for device")

if __name__ == "__main__":
    test_constraint_with_synthetic_data()
