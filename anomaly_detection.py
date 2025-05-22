"""
Anomaly detection for energy consumption data.

This module provides functions to detect outliers and anomalies in energy consumption data.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Any
import logging

# Setup logging
logger = logging.getLogger(__name__)

def detect_consumption_anomalies(df: pd.DataFrame, threshold: float = 3.0) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Detect anomalies in device energy consumption.
    
    Args:
        df: DataFrame with device_id, timestamp, energy columns
        threshold: Z-score threshold for anomaly detection
        
    Returns:
        DataFrame with anomalies, dictionary with anomaly statistics
    """
    if df.empty:
        return pd.DataFrame(), {"devices_with_anomalies": [], "total_anomalies": 0}
    
    results = []
    anomaly_stats = {
        "devices_with_anomalies": [],
        "total_anomalies": 0,
        "anomalies_by_device": {},
        "anomaly_dates": []
    }
    
    # Group by device
    for device, group in df.groupby("device_id"):
        if len(group) < 10:  # Need enough data points
            continue
            
        # Sort by timestamp
        group = group.sort_values("timestamp")
        
        # Calculate hourly energy consumption
        group["hour"] = group["timestamp"].dt.floor("H")
        hourly = group.groupby("hour").agg(
            energy_start=("energy", "first"),
            energy_end=("energy", "last"),
            device_class=("device_class", "first")
        ).reset_index()
        
        hourly["consumption"] = (hourly["energy_end"] - hourly["energy_start"]).clip(lower=0)
        
        # Detect anomalies using Z-score
        mean = hourly["consumption"].mean()
        std = hourly["consumption"].std()
        
        if std == 0:  # Skip if standard deviation is zero
            continue
            
        hourly["z_score"] = (hourly["consumption"] - mean) / std
        anomalies = hourly[abs(hourly["z_score"]) > threshold].copy()
        
        if not anomalies.empty:
            anomaly_stats["devices_with_anomalies"].append(device)
            anomaly_stats["total_anomalies"] += len(anomalies)
            anomaly_stats["anomalies_by_device"][device] = len(anomalies)
            anomaly_stats["anomaly_dates"].extend(anomalies["hour"].dt.date.tolist())
            
            # Mark anomaly type
            anomalies["anomaly_type"] = np.where(
                anomalies["z_score"] > 0, "High Consumption", "Low Consumption"
            )
            
            # Add device ID
            anomalies["device_id"] = device
            
            results.append(anomalies)
    
    if not results:
        return pd.DataFrame(), anomaly_stats
        
    anomalies_df = pd.concat(results, ignore_index=True)
    return anomalies_df, anomaly_stats


def detect_pattern_anomalies(daily_df: pd.DataFrame, window_size: int = 7, threshold: float = 2.5) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Detect anomalies in consumption patterns using rolling window.
    
    Args:
        daily_df: DataFrame with date and consumption by device class
        window_size: Size of rolling window for pattern analysis
        threshold: Z-score threshold for anomaly detection
        
    Returns:
        DataFrame with anomalies, dictionary with anomaly statistics
    """
    if daily_df.empty or "date" not in daily_df.columns:
        return pd.DataFrame(), {"total_pattern_anomalies": 0}
    
    # Ensure we have at least window_size + 1 data points
    if len(daily_df) <= window_size:
        return pd.DataFrame(), {"total_pattern_anomalies": 0}
    
    # Convert to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(daily_df["date"]):
        daily_df["date"] = pd.to_datetime(daily_df["date"])
    
    # Make a copy for analysis    
    daily = daily_df.set_index("date").copy()
    
    # Get consumption columns (exclude non-numeric columns)
    consumption_cols = [col for col in daily.columns if pd.api.types.is_numeric_dtype(daily[col])]
    
    if not consumption_cols:
        return pd.DataFrame(), {"total_pattern_anomalies": 0}
    
    # Calculate rolling statistics
    rolling_mean = daily[consumption_cols].rolling(window=window_size, min_periods=window_size//2).mean()
    rolling_std = daily[consumption_cols].rolling(window=window_size, min_periods=window_size//2).std()
    
    # Replace zeros in std with mean std to avoid division by zero
    for col in consumption_cols:
        mean_std = rolling_std[col].mean()
        rolling_std[col] = rolling_std[col].replace(0, mean_std)
    
    # Calculate z-scores
    z_scores = (daily[consumption_cols] - rolling_mean) / rolling_std
    
    # Identify anomalies
    anomalies = (abs(z_scores) > threshold)
    
    # Create result DataFrame
    results = []
    for col in consumption_cols:
        col_anomalies = anomalies[col]
        if col_anomalies.any():
            # Get anomaly dates
            anomaly_dates = col_anomalies[col_anomalies].index
            
            # Create result dataframe
            for date in anomaly_dates:
                actual = daily.loc[date, col]
                expected = rolling_mean.loc[date, col]
                z = z_scores.loc[date, col]
                
                results.append({
                    "date": date,
                    "device_class": col,
                    "actual_consumption": actual,
                    "expected_consumption": expected,
                    "z_score": z,
                    "anomaly_type": "High Consumption" if z > 0 else "Low Consumption"
                })
    
    if not results:
        return pd.DataFrame(), {"total_pattern_anomalies": 0}
    
    result_df = pd.DataFrame(results)
    
    # Calculate statistics
    stats = {
        "total_pattern_anomalies": len(result_df),
        "anomalies_by_class": result_df["device_class"].value_counts().to_dict(),
        "anomalies_by_type": result_df["anomaly_type"].value_counts().to_dict(),
        "anomaly_dates": result_df["date"].dt.date.unique().tolist()
    }
    
    return result_df, stats


def plot_anomalies(daily_df: pd.DataFrame, anomalies_df: pd.DataFrame, output_dir: str = "visualizations"):
    """
    Plot consumption data with highlighted anomalies.
    
    Args:
        daily_df: DataFrame with date and consumption by device class
        anomalies_df: DataFrame with detected anomalies
        output_dir: Directory to save plots
    """
    if daily_df.empty or anomalies_df.empty:
        logger.warning("No data for anomaly plotting")
        return
    
    # Convert to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(daily_df["date"]):
        daily_df["date"] = pd.to_datetime(daily_df["date"])
    
    # Set date as index
    daily = daily_df.set_index("date").copy()
    
    # Get consumption columns
    consumption_cols = [col for col in daily.columns if pd.api.types.is_numeric_dtype(daily[col])]
    
    if not consumption_cols:
        logger.warning("No consumption columns found")
        return
    
    # Create a plot for each device class
    for col in consumption_cols:
        if col not in anomalies_df["device_class"].values:
            continue
            
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot consumption
        daily[col].plot(ax=ax, label=f"{col} Consumption", linewidth=2)
        
        # Get anomalies for this class
        class_anomalies = anomalies_df[anomalies_df["device_class"] == col]
        
        # Plot high anomalies in red
        high_anomalies = class_anomalies[class_anomalies["anomaly_type"] == "High Consumption"]
        if not high_anomalies.empty:
            dates = pd.to_datetime(high_anomalies["date"])
            values = high_anomalies["actual_consumption"]
            ax.scatter(dates, values, color='red', s=50, label="High Consumption Anomaly")
        
        # Plot low anomalies in blue
        low_anomalies = class_anomalies[class_anomalies["anomaly_type"] == "Low Consumption"]
        if not low_anomalies.empty:
            dates = pd.to_datetime(low_anomalies["date"])
            values = low_anomalies["actual_consumption"]
            ax.scatter(dates, values, color='blue', s=50, label="Low Consumption Anomaly")
        
        # Add labels and legend
        ax.set_title(f"{col} Consumption with Anomalies")
        ax.set_xlabel("Date")
        ax.set_ylabel("Consumption (kWh)")
        ax.legend()
        
        # Save figure
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{col}_anomalies.png", dpi=300)
        plt.close()


def main(input_csv: str = "csv_output/daily_consumption.csv", output_dir: str = "visualizations"):
    """
    Main function to detect and visualize anomalies.
    
    Args:
        input_csv: Path to daily consumption CSV
        output_dir: Directory to save output plots
    """
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s · %(levelname)s · %(message)s"
    )
    
    # Load daily consumption data
    try:
        daily_df = pd.read_csv(input_csv)
        logger.info(f"Loaded {len(daily_df)} rows from {input_csv}")
    except Exception as e:
        logger.error(f"Error loading {input_csv}: {e}")
        return
    
    # Detect pattern anomalies
    anomalies_df, stats = detect_pattern_anomalies(daily_df)
    
    if anomalies_df.empty:
        logger.info("No pattern anomalies detected")
        return
    
    logger.info(f"Detected {stats['total_pattern_anomalies']} pattern anomalies")
    
    # Print anomaly statistics
    print("\n" + "="*80)
    print(" "*30 + "ANOMALY DETECTION RESULTS")
    print("="*80)
    
    print(f"\nTotal anomalies detected: {stats['total_pattern_anomalies']}")
    
    print("\nAnomalies by device class:")
    for cls, count in stats.get("anomalies_by_class", {}).items():
        print(f"  {cls}: {count}")
    
    print("\nAnomalies by type:")
    for typ, count in stats.get("anomalies_by_type", {}).items():
        print(f"  {typ}: {count}")
    
    # Plot anomalies
    plot_anomalies(daily_df, anomalies_df, output_dir)
    logger.info(f"Anomaly plots saved to {output_dir}")
    
    # Save anomalies to CSV
    anomalies_df.to_csv(f"{output_dir}/detected_anomalies.csv", index=False)
    logger.info(f"Anomalies saved to {output_dir}/detected_anomalies.csv")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Anomaly detection for energy consumption data")
    parser.add_argument("--input", "-i", default="csv_output/daily_consumption.csv",
                      help="Input CSV file with daily consumption data")
    parser.add_argument("--output", "-o", default="visualizations",
                      help="Output directory for plots")
    args = parser.parse_args()
    
    main(args.input, args.output) 