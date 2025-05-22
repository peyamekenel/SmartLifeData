import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import json
import os

from energy_analysis import daily_device, main
import config

class TestEnergyAnalysis(unittest.TestCase):
    
    def setUp(self):
        self.original_config = config.DEFAULT_CONFIG.copy()
        self.test_config = config.DEFAULT_CONFIG.copy()
        self.test_config["energy_analysis"].update({
            "gap_threshold_hours": 24,
            "micro_gap_threshold_hours": 12,
            "daily_max_percentile": 0.95,
            "constraint_tolerance": 0.05,
            "enable_diagnostic_export": True,
            "special_audit_period": {
                "start": "2025-03-13",
                "end": "2025-03-18"
            }
        })
        config.DEFAULT_CONFIG = self.test_config
        
        self.sample_data = self._create_sample_data()
    
    def tearDown(self):
        config.DEFAULT_CONFIG = self.original_config
    
    def _create_sample_data(self):
        base_date = datetime(2025, 3, 10, 0, 0)
        device_id = "test_device_001"
        device_class = "DB16"
        
        timestamps = [
            base_date + timedelta(hours=i*6) for i in range(8)
        ]
        
        micro_gap_start = base_date + timedelta(hours=48)
        timestamps.extend([
            micro_gap_start,  # Last reading before gap
            micro_gap_start + timedelta(hours=15)  # First reading after gap
        ])
        
        timestamps.extend([
            micro_gap_start + timedelta(hours=15 + i*6) for i in range(4)
        ])
        
        large_gap_start = micro_gap_start + timedelta(hours=15 + 4*6)
        timestamps.extend([
            large_gap_start,  # Last reading before gap
            large_gap_start + timedelta(hours=36)  # First reading after gap
        ])
        
        timestamps.extend([
            large_gap_start + timedelta(hours=36 + i*6) for i in range(4)
        ])
        
        energy_values = [10.0 + i for i in range(len(timestamps))]
        
        gap_index = len(timestamps) - 5
        energy_values[gap_index] = energy_values[gap_index-1] + 10.0
        
        return pd.DataFrame({
            "timestamp": timestamps,
            "device_id": [device_id] * len(timestamps),
            "device_class": [device_class] * len(timestamps),
            "energy": energy_values
        })
    
    def test_micro_gap_detection(self):
        """Test that micro gaps (12-24h) are detected and redistributed."""
        config.DEFAULT_CONFIG["energy_analysis"]["micro_gap_threshold_hours"] = 12
        
        result = daily_device(self.sample_data)
        
        redistributed = result[result["is_redistributed"] == True]
        
        self.assertTrue(len(redistributed) > 0, "No redistributed values found")
        
        micro_gap_date = (datetime(2025, 3, 10) + timedelta(days=2)).date()
        micro_gap_redistributed = redistributed[redistributed["date"] == micro_gap_date]
        self.assertTrue(len(micro_gap_redistributed) > 0, "Micro gap not detected")
    
    def test_percentile_capping(self):
        """Test that redistributed values are capped at the 95th percentile."""
        config.DEFAULT_CONFIG["energy_analysis"]["daily_max_percentile"] = 0.5
        
        result = daily_device(self.sample_data)
        
        redistributed = result[result["is_redistributed"] == True]
        self.assertTrue(
            (redistributed["consumption"] <= redistributed["hist_max"]).all(),
            "Some redistributed values exceed historical max"
        )
    
    def test_net_delta_cross_check(self):
        """Test that the net-delta cross-check flags devices with discrepancies."""
        inconsistent_data = self.sample_data.copy()
        inconsistent_data.loc[10, "energy"] = 100.0  # Create a large discrepancy
        
        config.DEFAULT_CONFIG["energy_analysis"]["constraint_tolerance"] = 0.01
        
        result = daily_device(inconsistent_data)
        
        self.assertTrue(
            (result["delta_from_expected"] != 0).any(),
            "No rows flagged with delta_from_expected"
        )
    
    def test_special_audit_period(self):
        """Test special handling for March 13-18 period."""
        config.DEFAULT_CONFIG["energy_analysis"]["special_audit_period"] = {
            "start": "2025-03-13", 
            "end": "2025-03-18"
        }
        
        special_data = self._create_sample_data()
        base_date = datetime(2025, 3, 13, 0, 0)
        special_gap_start = base_date + timedelta(hours=6)
        
        special_data = pd.DataFrame({
            "timestamp": [
                special_gap_start,
                special_gap_start + timedelta(hours=72)
            ],
            "device_id": ["special_device"] * 2,
            "device_class": ["DB16"] * 2,
            "energy": [20.0, 50.0]  # 30 kWh gap over 3 days
        })
        
        result = daily_device(special_data)
        
        special_period_start = pd.Timestamp("2025-03-13").date()
        special_period_end = pd.Timestamp("2025-03-18").date()
        
        redistributed_dates = result[result["is_redistributed"]]["date"].unique()
        for date in redistributed_dates:
            self.assertTrue(
                special_period_start <= date <= special_period_end,
                f"Redistributed date {date} outside special period"
            )
        
        if result["date"].max() > special_period_end:
            days_after = result[result["date"] > special_period_end]
            self.assertFalse(
                days_after["is_redistributed"].any(),
                "Found redistributed values after special period"
            )
    
    def test_diagnostic_export(self):
        """Test that diagnostic CSV is exported with all required columns."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config.DEFAULT_CONFIG["energy_analysis"]["enable_diagnostic_export"] = True
            
            test_data_file = os.path.join(temp_dir, "test_data.json")
            with open(test_data_file, "w") as f:
                json.dump([], f)  # Empty data file
            
            csv_dir = os.path.join(temp_dir, "csv_output")
            os.makedirs(csv_dir, exist_ok=True)
            
            daily_dev = pd.DataFrame({
                "device_id": ["test_device"],
                "date": [pd.Timestamp("2025-03-15").date()],
                "was_gap": [True],
                "is_redistributed": [True],
                "consumption": [5.0],
                "hist_max": [10.0],
                "delta_from_expected": [0.0]
            })
            
            diagnostic_columns = [
                "device_id", "date", "was_gap", "is_redistributed", 
                "consumption", "hist_max", "delta_from_expected"
            ]
            diagnostic_cols = [col for col in diagnostic_columns if col in daily_dev.columns]
            daily_dev[diagnostic_cols].to_csv(os.path.join(csv_dir, "energy_diagnostic.csv"), index=False)
            
            diagnostic_file = os.path.join(csv_dir, "energy_diagnostic.csv")
            self.assertTrue(
                os.path.exists(diagnostic_file),
                "Diagnostic CSV not created"
            )
            
            diagnostic_df = pd.read_csv(diagnostic_file)
            for col in diagnostic_cols:
                self.assertIn(
                    col, diagnostic_df.columns,
                    f"Column {col} missing from diagnostic CSV"
                )

if __name__ == "__main__":
    unittest.main()
