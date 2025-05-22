"""
Test to verify that total consumption matches the actual difference 
between first and last readings for device 0xa4c138309b3a04fe.
"""
import pandas as pd
import os
import json
from energy_analysis import load_data, per_device_usage, CONFIG, daily_device

def test_total_constraint():
    """Test that total consumption matches actual difference."""
    if "energy_analysis" not in CONFIG:
        CONFIG["energy_analysis"] = {}
    CONFIG["energy_analysis"]["enforce_total_constraint"] = True
    
    print("Loading data...")
    df, _ = load_data('db16_db18_smartlife_feb15_apr15.json')
    
    device_id = '0xa4c138309b3a04fe'
    device_df = df[df['device_id'] == device_id]
    
    if device_df.empty:
        print(f"No data found for device {device_id}")
        return
    
    device_df_sorted = device_df.sort_values('timestamp')
    first_reading = device_df_sorted['energy'].iloc[0]
    last_reading = device_df_sorted['energy'].iloc[-1]
    actual_difference = last_reading - first_reading
    
    print(f"\nDevice {device_id}:")
    print(f"First reading: {first_reading:.2f} kWh")
    print(f"Last reading: {last_reading:.2f} kWh")
    print(f"Actual difference: {actual_difference:.2f} kWh")
    
    result_df, _, anomalies = per_device_usage(device_df)
    
    if not result_df.empty:
        calculated_total = result_df[result_df['device_id'] == device_id]['consumption'].values[0]
        print(f"\nCalculated total: {calculated_total:.2f} kWh")
        
        tolerance = CONFIG["energy_analysis"].get("constraint_tolerance", 0.05)
        max_allowed = actual_difference * (1 + tolerance)
        
        print(f"Max allowed (with {tolerance*100:.1f}% tolerance): {max_allowed:.2f} kWh")
        print(f"Constraint enforced: {calculated_total <= max_allowed}")
        
        total_constraint_anomalies = [a for a in anomalies if a.get('anomaly_type') == 'total_constraint']
        print(f"Total constraint anomalies detected: {len(total_constraint_anomalies)}")
        for anomaly in total_constraint_anomalies:
            print(f"  Original: {anomaly['original_value']:.2f} kWh, Adjusted: {anomaly['adjusted_value']:.2f} kWh")
        
        print("\nTesting daily consumption constraint...")
        os.makedirs("csv_output", exist_ok=True)
        
        result_df.to_csv("csv_output/device_usage.csv", index=False)
        
        daily_df, daily_anomalies = daily_device(device_df)
        
        if not daily_df.empty:
            device_daily = daily_df[daily_df['device_id'] == device_id]
            daily_sum = device_daily['consumption'].sum()
            print(f"Daily consumption sum: {daily_sum:.2f} kWh")
            print(f"Daily sum matches calculated total: {abs(daily_sum - calculated_total) < 0.1}")
            print(f"Daily sum matches actual difference: {abs(daily_sum - actual_difference) < max_allowed}")
            
            daily_constraint_anomalies = [a for a in daily_anomalies if a.get('anomaly_type') == 'daily_constraint']
            print(f"Daily constraint anomalies detected: {len(daily_constraint_anomalies)}")
            
            print("\nSample daily consumption values:")
            print(device_daily.head(5))
        else:
            print("No daily results for device")
    else:
        print("No device usage results for device")

if __name__ == "__main__":
    test_total_constraint()
