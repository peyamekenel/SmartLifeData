"""
Data quality module for energy analysis.

This module provides functions for assessing and reporting on data quality.
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class DataQualityMetrics:
    """Data quality metrics container."""
    total_records: int = 0
    valid_records: int = 0
    missing_timestamp: int = 0
    missing_energy: int = 0
    missing_device_id: int = 0
    records_outside_window: int = 0
    devices_with_negative_consumption: List[str] = None
    devices_with_insufficient_readings: List[str] = None
    devices_by_class: Dict[str, int] = None
    readings_per_device: Dict[str, int] = None
    reading_frequency: Dict[str, pd.Timedelta] = None
    _validity_ratio: float = None  # Added to handle serialized data
    
    def __init__(self, **kwargs):
        """
        Initialize metrics with kwargs for flexible deserialization.
        
        This allows loading from cache with additional fields.
        """
        # Set default values
        self.total_records = 0
        self.valid_records = 0
        self.missing_timestamp = 0
        self.missing_energy = 0
        self.missing_device_id = 0
        self.records_outside_window = 0
        self.devices_with_negative_consumption = []
        self.devices_with_insufficient_readings = []
        self.devices_by_class = {}
        self.readings_per_device = {}
        self.reading_frequency = {}
        self._validity_ratio = None
        
        # Apply any values from kwargs
        for key, value in kwargs.items():
            if key == 'validity_ratio':
                self._validity_ratio = value
            # Ignore avg_reading_frequency as it's derived
            elif key != 'avg_reading_frequency' and hasattr(self, key):
                setattr(self, key, value)
    
    @property
    def validity_ratio(self) -> float:
        """Calculate the ratio of valid records to total records."""
        if self._validity_ratio is not None:
            return self._validity_ratio
        return self.valid_records / self.total_records if self.total_records > 0 else 0.0
        
    @validity_ratio.setter
    def validity_ratio(self, value: float):
        """Set validity ratio (for deserialization)."""
        self._validity_ratio = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to a dictionary."""
        return {
            "total_records": self.total_records,
            "valid_records": self.valid_records,
            "_validity_ratio": self.validity_ratio,  # Store as private field for deserialization
            "missing_timestamp": self.missing_timestamp,
            "missing_energy": self.missing_energy,
            "missing_device_id": self.missing_device_id,
            "records_outside_window": self.records_outside_window,
            "devices_with_negative_consumption": self.devices_with_negative_consumption,
            "devices_with_insufficient_readings": self.devices_with_insufficient_readings,
            "devices_by_class": self.devices_by_class,
            "readings_per_device": self.readings_per_device,
            "avg_reading_frequency": {
                device: freq.total_seconds() / 3600  # Convert to hours
                for device, freq in self.reading_frequency.items()
            }
        }
    
    def print_summary(self) -> None:
        """Print a summary of data quality metrics."""
        print("\n" + "="*80)
        print(" "*30 + "DATA QUALITY SUMMARY")
        print("="*80)
        
        print(f"\nData Completeness:")
        print(f"  Total records: {self.total_records}")
        print(f"  Valid records: {self.valid_records} ({self.validity_ratio:.2%})")
        print(f"  Missing timestamp: {self.missing_timestamp}")
        print(f"  Missing energy: {self.missing_energy}")
        print(f"  Missing device ID: {self.missing_device_id}")
        print(f"  Records outside analysis window: {self.records_outside_window}")
        
        print(f"\nDevice Coverage:")
        print(f"  Devices by class:")
        for cls, count in self.devices_by_class.items():
            print(f"    {cls}: {count} devices")
        
        if self.devices_with_negative_consumption:
            print(f"\nDevices with negative consumption (excluded): {len(self.devices_with_negative_consumption)}")
            
        if self.devices_with_insufficient_readings:
            print(f"\nDevices with insufficient readings (excluded): {len(self.devices_with_insufficient_readings)}")
        
        print(f"\nReading Frequency:")
        avg_readings = np.mean(list(self.readings_per_device.values())) if self.readings_per_device else 0
        print(f"  Average readings per device: {avg_readings:.1f}")
        
        if self.reading_frequency:
            avg_freq = np.mean([freq.total_seconds() / 3600 for freq in self.reading_frequency.values()])
            print(f"  Average reading frequency: {avg_freq:.2f} hours")
        
        print("\n" + "="*80)


def calculate_device_reading_frequency(df: pd.DataFrame) -> Dict[str, pd.Timedelta]:
    """
    Calculate the average time between readings for each device.
    
    Args:
        df: DataFrame with device_id and timestamp columns
        
    Returns:
        Dictionary mapping device_id to average reading frequency
    """
    freqs = {}
    for device, group in df.groupby('device_id'):
        # Sort by timestamp
        group = group.sort_values('timestamp')
        # Calculate time differences
        time_diffs = group['timestamp'].diff().dropna()
        if not time_diffs.empty:
            # Calculate mean frequency (excluding outliers)
            q1, q3 = time_diffs.quantile([0.25, 0.75])
            iqr = q3 - q1
            valid_diffs = time_diffs[(time_diffs >= q1 - 1.5 * iqr) & (time_diffs <= q3 + 1.5 * iqr)]
            if not valid_diffs.empty:
                freqs[device] = valid_diffs.mean()
    return freqs


def validate_raw_data(raw_data: List[Dict[str, Any]], window_start: pd.Timestamp, 
                      window_end: pd.Timestamp, timezone_str: str) -> DataQualityMetrics:
    """
    Validate raw data and calculate quality metrics.
    
    Args:
        raw_data: List of raw data records
        window_start: Start of analysis window
        window_end: End of analysis window
        timezone_str: Timezone string
        
    Returns:
        DataQualityMetrics
    """
    metrics = DataQualityMetrics(total_records=len(raw_data))
    
    device_readings = {}
    
    for rec in raw_data:
        # Check device ID
        if not rec.get("devId"):
            metrics.missing_device_id += 1
            continue
            
        device_id = rec.get("devId")
        device_readings[device_id] = device_readings.get(device_id, 0) + 1
        
        # Check timestamp
        ts_str = rec.get("timestamp")
        if not ts_str:
            metrics.missing_timestamp += 1
            continue
            
        try:
            ts = pd.to_datetime(ts_str, format='ISO8601')
            # Convert to target timezone
            ts = ts.tz_convert(timezone_str)
        except (ValueError, AttributeError):
            metrics.missing_timestamp += 1
            continue
            
        # Check if timestamp is within analysis window
        if not (window_start <= ts <= window_end):
            metrics.records_outside_window += 1
            
        # Check energy value
        energy = None
        for s in rec.get("status", []):
            if s.get("code") == "energy":
                energy = s.get("value")
                break
        
        if energy is None:
            metrics.missing_energy += 1
            continue
            
        # Count valid record
        metrics.valid_records += 1
    
    # Store readings per device
    metrics.readings_per_device = device_readings
    
    return metrics


def validate_processed_data(df: pd.DataFrame, min_readings: int = 5) -> DataQualityMetrics:
    """
    Validate processed DataFrame and calculate quality metrics.
    
    Args:
        df: Processed DataFrame
        min_readings: Minimum number of readings for a device to be considered valid
        
    Returns:
        DataQualityMetrics
    """
    metrics = DataQualityMetrics()
    
    # Count records by device class
    class_counts = df['device_class'].value_counts().to_dict()
    metrics.devices_by_class = class_counts
    
    # Count readings per device
    device_counts = df.groupby('device_id').size().to_dict()
    metrics.readings_per_device = device_counts
    
    # Find devices with insufficient readings
    insufficient = [dev for dev, count in device_counts.items() if count < min_readings]
    metrics.devices_with_insufficient_readings = insufficient
    
    # Calculate reading frequency
    metrics.reading_frequency = calculate_device_reading_frequency(df)
    
    return metrics


def validate_consumption_data(df: pd.DataFrame) -> DataQualityMetrics:
    """
    Validate device consumption data and calculate quality metrics.
    
    Args:
        df: DataFrame with device consumption data
        
    Returns:
        DataQualityMetrics
    """
    metrics = DataQualityMetrics()
    
    # Count records by device class
    class_counts = df['device_class'].value_counts().to_dict()
    metrics.devices_by_class = class_counts
    
    # Find devices with negative consumption
    negative = df[df['consumption'] < 0]['device_id'].tolist()
    metrics.devices_with_negative_consumption = negative
    
    return metrics 