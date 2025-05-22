from __future__ import annotations
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
import datetime

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from zoneinfo import ZoneInfo  # Python 3.9+ standard library for timezones

# Import our new modules
from config import load_config
from data_quality import (
    validate_raw_data, validate_processed_data, validate_consumption_data, 
    DataQualityMetrics
)

# ────────────────────────────────────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s · %(levelname)s · %(message)s"
)
log = logging.getLogger(__name__)

# Load configuration
CONFIG = load_config()

# Get device classes from config
DB16_DEVICES: List[str] = CONFIG["device_classes"]["DB16"]
DB18_DEVICES: List[str] = CONFIG["device_classes"]["DB18"]
DB14_DEVICES: List[str] = CONFIG["device_classes"]["DB14"]

# Light codes from config
LIGHT_CODES = set(CONFIG["light_codes"])

@dataclass(frozen=True)
class Window:
    start: pd.Timestamp
    end: pd.Timestamp


# Define timezone from configuration
TZ_STR = CONFIG["analysis_window"]["timezone"]
try:
    TIMEZONE = ZoneInfo(TZ_STR)
except Exception as e:
    log.error(f"Invalid timezone {TZ_STR}: {e}")
    log.info("Falling back to UTC")
    TIMEZONE = ZoneInfo("UTC")

# Create analysis window from config
WINDOW = Window(
    pd.Timestamp(CONFIG["analysis_window"]["start"]).replace(tzinfo=TIMEZONE),
    pd.Timestamp(CONFIG["analysis_window"]["end"]).replace(tzinfo=TIMEZONE),
)

# Visualization settings
OUT_DIR = Path(CONFIG["visualization"]["output_dir"])
OUT_DIR.mkdir(exist_ok=True)

# Set matplotlib parameters from config
plt.rcParams["axes.prop_cycle"] = plt.cycler(
    color=CONFIG["visualization"]["color_palette"]
)
plt.rcParams.update({
    "figure.figsize": CONFIG["visualization"]["figure_size"], 
    "font.size": CONFIG["visualization"]["font_size"]
})


def _fmt_y(ax):
    ax.yaxis.set_major_formatter(lambda x, pos: f"{x:,.2f}")


# ────────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ────────────────────────────────────────────────────────────────────────────────


def load_data(file_path: str | Path) -> Tuple[pd.DataFrame, DataQualityMetrics]:
    """Load JSON array → tidy DataFrame with data quality metrics.

    If the file could be large, switch to incremental parsing (ijson).
    """
    fp = Path(file_path)
    log.info("Reading %s", fp)
    try:
        with fp.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    except json.JSONDecodeError as e:
        log.error("JSON parse error: %s", e)
        raise

    # Validate raw data and calculate quality metrics
    quality_metrics = validate_raw_data(raw, WINDOW.start, WINDOW.end, TZ_STR)
    
    rows: list[tuple[str, pd.Timestamp, float, str]] = []
    for rec in raw:
        did = rec.get("devId")
        
        # Parse timestamp with timezone info
        ts_str = rec.get("timestamp")
        if not ts_str:
            continue
            
        try:
            # Parse with timezone info preserved
            ts = pd.to_datetime(ts_str)
            # Convert to target timezone
            ts = ts.replace(tzinfo=ZoneInfo("UTC")).astimezone(TIMEZONE)
        except (ValueError, AttributeError):
            continue
            
        if not did or pd.isna(ts):
            continue
        
        if not (WINDOW.start <= ts <= WINDOW.end):
            continue

        energy = None
        for s in rec.get("status", []):
            if s.get("code") == "energy":
                energy = s.get("value")
                break
        if energy is None:
            continue

        dclass = (
            "DB16" if did in DB16_DEVICES else 
            "DB18" if did in DB18_DEVICES else 
            "DB14" if did in DB14_DEVICES else None
        )
        if dclass is None:
            continue

        rows.append((did, ts, float(energy), dclass))

    df = pd.DataFrame(rows, columns=["device_id", "timestamp", "energy", "device_class"])
    log.info("Valid rows: %d", len(df))
    
    return df, quality_metrics


# ────────────────────────────────────────────────────────────────────────────────
# ANALYSIS HELPERS
# ────────────────────────────────────────────────────────────────────────────────


def per_device_usage(df: pd.DataFrame) -> Tuple[pd.DataFrame, DataQualityMetrics]:
    """
    Calculate energy consumption per device with data quality metrics.
    
    Uses a more robust approach to handle resets and outliers.
    """
    if df.empty:
        return pd.DataFrame(), DataQualityMetrics()
    
    df = df.sort_values(["device_id", "timestamp"])
    
    # Group by device and calculate consumption more robustly
    results = []
    devices_with_resets = []
    
    for device, group in df.groupby("device_id"):
        if len(group) < CONFIG["energy_analysis"]["min_valid_readings"]:
            continue
            
        # Check for anomalies or meter resets
        energy_diff = group["energy"].diff()
        
        # Identify large negative jumps (potential resets)
        reset_indices = energy_diff[energy_diff < -10].index
        
        if not reset_indices.empty:
            # Handle resets by treating each segment separately
            devices_with_resets.append(device)
            segments = np.split(group.index, reset_indices)
            
            total_consumption = 0
            for segment in segments:
                if len(segment) >= 2:
                    segment_df = group.loc[segment]
                    segment_consumption = segment_df["energy"].iloc[-1] - segment_df["energy"].iloc[0]
                    if segment_consumption >= 0:
                        total_consumption += segment_consumption
            
            results.append({
                "device_id": device,
                "device_class": group["device_class"].iloc[0],
                "first_energy": group["energy"].iloc[0],
                "last_energy": group["energy"].iloc[-1],
                "consumption": total_consumption,
                "has_resets": True,
                "num_readings": len(group)
            })
        else:
            # Normal case - no resets
            consumption = group["energy"].iloc[-1] - group["energy"].iloc[0]
            results.append({
                "device_id": device,
                "device_class": group["device_class"].iloc[0],
                "first_energy": group["energy"].iloc[0],
                "last_energy": group["energy"].iloc[-1],
                "consumption": max(0, consumption),  # Ensure non-negative
                "has_resets": False,
                "num_readings": len(group)
            })
    
    result_df = pd.DataFrame(results)
    
    # Filter out devices with negative consumption if configured
    if CONFIG["energy_analysis"]["filter_negative_consumption"]:
        valid = result_df.query("consumption >= 0").copy()
    else:
        valid = result_df.copy()
    
    log.info("Devices kept: %d • Excluded: %d", len(valid), len(result_df) - len(valid))
    
    # Calculate data quality metrics
    quality_metrics = validate_consumption_data(result_df)
    
    if devices_with_resets:
        log.warning("Detected possible meter resets for %d devices", len(devices_with_resets))
        
    return valid.round({"consumption": 2}), quality_metrics


def class_totals(dev_usage: pd.DataFrame) -> pd.DataFrame:
    """Calculate total consumption by device class."""
    if dev_usage.empty:
        return pd.DataFrame()
        
    return (
        dev_usage.groupby("device_class")["consumption"].sum().reset_index().round(2)
    )


def daily_class(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate daily consumption by device class with improved handling of missing data."""
    if df.empty:
        return pd.DataFrame()
        
    df = df.sort_values(["device_id", "timestamp"])
    # Extract date from timestamp while preserving timezone information
    df["date"] = df["timestamp"].dt.date

    # Group by device, date, class and get first/last readings
    dd = (
        df.groupby(["device_id", "date", "device_class"])
        .agg(first=("energy", "first"), last=("energy", "last"))
        .reset_index()
    )
    
    # Calculate daily consumption with non-negative constraint
    dd["delta"] = (dd["last"] - dd["first"]).clip(lower=0)

    # Handle potential anomalies (extremely high values)
    # Identify extreme outliers (more than 3 IQRs above Q3)
    q1, q3 = dd["delta"].quantile([0.25, 0.75])
    iqr = q3 - q1
    upper_bound = q3 + 3 * iqr
    
    # Cap extreme values
    if (dd["delta"] > upper_bound).any():
        log.warning("Capping %d extreme daily consumption values", (dd["delta"] > upper_bound).sum())
        dd["delta"] = dd["delta"].clip(upper=upper_bound)

    # Sum by date and device class
    daily = dd.groupby(["date", "device_class"])["delta"].sum().unstack(fill_value=0)
    
    # Use localized timestamps for date range
    idx_full = pd.date_range(
        WINDOW.start.date(), 
        WINDOW.end.date(), 
        freq="D"
    )
    daily = daily.reindex(idx_full, fill_value=0)
    daily.index.name = "date"
    daily = daily.round(2)
    return daily.reset_index()


def monthly(daily: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily DataFrame to month totals with dynamic ordering/labels."""
    if daily.empty:
        return pd.DataFrame()
        
    # Convert to datetime with timezone info before extracting period
    daily["month"] = pd.to_datetime(daily["date"]).dt.to_period("M")
    m = daily.groupby("month").sum(numeric_only=True).reset_index()

    # order chronologically
    m["month_start"] = m["month"].dt.to_timestamp()
    m = m.sort_values("month_start").drop(columns="month_start")

    m["month"] = m["month"].dt.strftime("%B %Y")  # e.g. "March 2025"
    return m.round(2)


def weekday(daily: pd.DataFrame) -> pd.DataFrame:
    """Calculate average consumption by weekday."""
    if daily.empty:
        return pd.DataFrame()
        
    # Convert to datetime with timezone info before extracting day name
    daily["weekday"] = pd.to_datetime(daily["date"]).dt.day_name()
    w = daily.groupby("weekday").mean(numeric_only=True).reset_index()

    order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    w["weekday"] = pd.Categorical(w["weekday"], order, ordered=True)
    return w.sort_values("weekday").round(2)


def hourly_average(df: pd.DataFrame) -> pd.DataFrame:
    """
    Average hourly consumption for each device class, separated by weekday/weekend.
    """
    if df.empty:
        return pd.DataFrame()
        
    # 1) Date and time information
    df = df.sort_values(["device_id", "timestamp"]).copy()
    df["date"] = df["timestamp"].dt.date
    df["hour_of_day"] = df["timestamp"].dt.hour
    # Add weekday/weekend flag
    df["is_weekend"] = df["timestamp"].dt.dayofweek >= 5  # 5=Saturday, 6=Sunday

    # 2) First/last energy values by device-day-hour
    g = (
        df.groupby(
            ["device_id", "date", "hour_of_day", "is_weekend", "device_class"],
            observed=True
        )
        .agg(first=("energy", "first"), last=("energy", "last"))
        .reset_index()
    )
    g["delta"] = (g["last"] - g["first"]).clip(lower=0)
    
    # Handle outliers in hourly data
    q1, q3 = g["delta"].quantile([0.25, 0.75])
    iqr = q3 - q1
    upper_bound = q3 + 3 * iqr
    
    # Cap extreme values
    if (g["delta"] > upper_bound).any():
        log.warning("Capping %d extreme hourly consumption values", (g["delta"] > upper_bound).sum())
        g["delta"] = g["delta"].clip(upper=upper_bound)

    # 3) Sum devices for the same day, hour, and class
    cls_hour = (
        g.groupby(["date", "hour_of_day", "is_weekend", "device_class"], observed=True)["delta"]
        .sum()
        .reset_index(name="class_sum")
    )

    # 4) Take average across days for each hour, weekend status, and device class
    hourly = (
        cls_hour
        .groupby(["hour_of_day", "is_weekend", "device_class"], observed=True)["class_sum"]
        .mean()
        .reset_index()
    )
    
    # 5) Pivot the data for easier plotting
    hourly_pivot = hourly.pivot_table(
        index="hour_of_day",
        columns=["is_weekend", "device_class"],
        values="class_sum",
        fill_value=0
    ).round(2)
    
    # Flatten multi-level columns and create more readable names
    hourly_pivot.columns = [
        f"{cls}_{'Weekend' if weekend else 'Weekday'}"
        for weekend, cls in hourly_pivot.columns
    ]
    
    return hourly_pivot.reset_index()


# ────────────────────────────────────────────────────────────────────────────────
# LIGHT-USAGE (STATE) ANALYSIS  –  HOURLY, MEAN AGGREGATION
# ────────────────────────────────────────────────────────────────────────────────

def _extract_lights(rec):
    """
    Yields tuples:
        (pseudo_id, timestamp[TZ-aware], is_on[bool], device_class)
    Each 'state_l1' / 'state_l2' becomes its own pseudo luminaire.
    """
    did   = rec.get("devId")
    dcls  = None                 # will hold DB16 / DB18 / DB14
    ts    = rec.get("timestamp")
    if not did or not ts:
        return []

    ts = pd.to_datetime(ts).tz_convert(TIMEZONE)
    if not (WINDOW.start <= ts <= WINDOW.end):
        return []

    for s in rec.get("status", []):
        code = s.get("code")
        if code not in LIGHT_CODES:
            continue
        val = str(s.get("value")).lower()             # 'on' / 'off'
        pseudo = f"{did}_{code}"                      # unique luminaire
        # derive device_class once (cheap lookup)
        if dcls is None:
            dcls = (
                "DB16" if did in DB16_DEVICES else 
                "DB18" if did in DB18_DEVICES else 
                "DB14" if did in DB14_DEVICES else None
            )
        if dcls is None:
            continue
        yield (pseudo, ts, val == "on", dcls)


def load_light_df(path: str | Path) -> pd.DataFrame:
    """Parse JSON → tidy luminaire-level dataframe."""
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        raw = json.load(f)
    for rec in raw:
        rows.extend(_extract_lights(rec))

    df = pd.DataFrame(
        rows, columns=["luminaire_id", "timestamp", "is_on", "device_class"]
    )
    return df


def hourly_class_usage(light_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns an hourly time-series:
      index = hourly timestamp
      columns = DB16, DB18, DB14      (usage rate 0-1)
    """
    # 1) full hourly grid per luminaire
    s = (
        light_df
        .set_index("timestamp")
        .groupby("luminaire_id")["is_on"]
        .resample("1h")
        .ffill()                     # forward-fill last known state
    )
    
    # Handle pandas future warning about downcasting
    # Method 1: Convert to proper boolean type first
    s = s.astype('bool').fillna(False)
    
    # Alternative method if the above doesn't work:
    # s = s.fillna(False)
    # s = s.infer_objects(copy=False)
    
    s = s.reset_index()              # → luminaire_id, timestamp, is_on

    # 2) attach device_class
    s = s.merge(
        light_df[["luminaire_id", "device_class"]].drop_duplicates(),
        on="luminaire_id",
        how="left",
    )

    # 3) mean across luminaires in each class, each hour
    hourly = (
        s.groupby(["timestamp", "device_class"])["is_on"]
        .mean()                      # partial usage e.g. 2/5 = .4
        .unstack(fill_value=0)       # columns: DB16 / DB18 / DB14
        .sort_index()
    )
    return hourly.round(4)


# ────────────────────────────────────────────────────────────────────────────────
# DAILY, WEEKLY, MONTHLY USAGE-RATE DERIVATIVES
# ────────────────────────────────────────────────────────────────────────────────

def daily_usage(hourly: pd.DataFrame) -> pd.DataFrame:
    """Average of the 24 hourly rates → one value per day per class."""
    return (
        hourly
        .resample("1D")
        .mean()
        .round(4)
    )

def weekly_pattern(daily: pd.DataFrame) -> pd.DataFrame:
    """Mean usage rate for each weekday (Mon…Sun)."""
    out = (
        daily
        .assign(weekday=lambda x: x.index.day_name())
        .groupby("weekday")
        .mean()
        .reindex([
            "Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"
        ])
        .round(4)
    )
    return out

def monthly_pattern(daily: pd.DataFrame) -> pd.DataFrame:
    """Mean usage rate for each calendar month."""
    return (
        daily
        .resample("ME")  # Changed from "M" to "ME" for month-end frequency
        .mean()
        .rename_axis("month")
        .round(4)
    )



# ────────────────────────────────────────────────────────────────────────────────
# PLOTS
# ────────────────────────────────────────────────────────────────────────────────


def bar_total(tot: pd.DataFrame):
    ax = tot.plot.bar(x="device_class", y="consumption", legend=False)
    ax.set_title("Total Consumption (15 Feb – 15 Apr 2025, Turkey Time)")
    ax.set_xlabel("Device Class")
    ax.set_ylabel("kWh")
    _fmt_y(ax)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "total_class_consumption.png", dpi=300)
    plt.close()


def line_daily(daily: pd.DataFrame):
    # Get columns that actually exist in the dataframe
    available_cols = [col for col in ["DB16", "DB18", "DB14"] if col in daily.columns]
    
    if not available_cols:
        log.warning("No device class data available for plotting in line_daily")
        return
        
    ax = daily.plot(x="date", y=available_cols, linewidth=2)
    ax.set_title("Daily Consumption Trend (Turkey Time)")
    ax.set_xlabel("Date")
    ax.set_ylabel("kWh")

    # dinamik tarih ekseni (gün/hafta arası yoğunlukta okunaklı)
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    plt.xticks(rotation=45)
    _fmt_y(ax)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "daily_consumption_trend.png", dpi=300)
    plt.close()


def bar_month(m: pd.DataFrame):
    ax = m.set_index("month").plot.bar()
    ax.set_xlabel("Month")
    ax.set_ylabel("kWh")
    ax.set_title("Monthly Consumption by Class (Turkey Time)")
    _fmt_y(ax)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "monthly_consumption.png", dpi=300)
    plt.close()


def bar_weekday(w: pd.DataFrame):
    ax = w.set_index("weekday").plot.bar()
    ax.set_xlabel("Weekday")
    ax.set_ylabel("Average kWh")
    ax.set_title("Weekday Consumption Pattern (Turkey Time)")
    _fmt_y(ax)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "weekday_consumption_pattern.png", dpi=300)
    plt.close()

def line_hourly(hourly: pd.DataFrame):
    # Plot with separate lines for weekday/weekend
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Define line styles and colors
    styles = {
        'DB16_Weekday': {'color': '#0072B2', 'linestyle': '-', 'marker': 'o', 'label': 'DB16 Weekday'},
        'DB16_Weekend': {'color': '#0072B2', 'linestyle': '--', 'marker': 's', 'label': 'DB16 Weekend'},
        'DB18_Weekday': {'color': '#D55E00', 'linestyle': '-', 'marker': 'o', 'label': 'DB18 Weekday'},
        'DB18_Weekend': {'color': '#D55E00', 'linestyle': '--', 'marker': 's', 'label': 'DB18 Weekend'},
        'DB14_Weekday': {'color': '#009E73', 'linestyle': '-', 'marker': 'o', 'label': 'DB14 Weekday'},
        'DB14_Weekend': {'color': '#009E73', 'linestyle': '--', 'marker': 's', 'label': 'DB14 Weekend'}
    }
    
    # Check if we have any columns to plot
    available_cols = [col for col in styles.keys() if col in hourly.columns]
    if not available_cols:
        log.warning("No hourly consumption data available for plotting")
        return
    
    # Plot each line
    for col, style in styles.items():
        if col in hourly.columns:
            ax.plot(hourly['hour_of_day'], hourly[col], linewidth=2, **style)
    
    # Set plot attributes
    ax.set_title("Average Hourly Consumption: Weekday vs Weekend (Turkey Time)")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("kWh")
    ax.set_xticks(range(0, 24))
    ax.legend()
    _fmt_y(ax)
    
    plt.tight_layout()
    plt.savefig(OUT_DIR / "hourly_weekday_weekend_consumption.png", dpi=300)
    
    # Also create a second plot with separate subplots for clearer comparison
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
    
    # Weekday subplot
    weekday_cols = [col for col in ['DB16_Weekday', 'DB18_Weekday', 'DB14_Weekday'] if col in hourly.columns]
    if weekday_cols:
        for col in weekday_cols:
            ax1.plot(
                hourly['hour_of_day'], 
                hourly[col], 
                color=styles[col]['color'], 
                marker='o', 
                linewidth=2,
                label=styles[col]['label']
            )
        ax1.set_title("Weekday Average Hourly Consumption (Turkey Time)")
        ax1.set_ylabel("kWh")
        ax1.legend()
        _fmt_y(ax1)
    else:
        ax1.set_title("No weekday data available")
    
    # Weekend subplot
    weekend_cols = [col for col in ['DB16_Weekend', 'DB18_Weekend', 'DB14_Weekend'] if col in hourly.columns]
    if weekend_cols:
        for col in weekend_cols:
            ax2.plot(
                hourly['hour_of_day'], 
                hourly[col], 
                color=styles[col]['color'], 
                marker='s', 
                linewidth=2,
                label=styles[col]['label']
            )
        ax2.set_title("Weekend Average Hourly Consumption (Turkey Time)")
        ax2.set_xlabel("Hour of Day")
        ax2.set_ylabel("kWh")
        ax2.set_xticks(range(0, 24))
        ax2.legend()
        _fmt_y(ax2)
    else:
        ax2.set_title("No weekend data available")
    
    plt.tight_layout()
    plt.savefig(OUT_DIR / "hourly_split_weekday_weekend.png", dpi=300)
    
    plt.close('all')

# ────────────────────────────────────────────────────────────────────────────────
# PLOTS  –  USAGE RATE (0-1)
# ────────────────────────────────────────────────────────────────────────────────

def bar_weekly_usage(week_df: pd.DataFrame):
    ax = week_df.plot.bar()
    ax.set_title("Weekly Usage Pattern (Hourly-Mean Rate)")
    ax.set_xlabel("Weekday")
    ax.set_ylabel("Usage Rate (0-1)")
    ax.yaxis.set_major_formatter(lambda x, pos: f"{x:.2f}")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "weekly_usage_pattern.png", dpi=300)
    plt.close()

def bar_monthly_usage(month_df: pd.DataFrame):
    ax = month_df.plot.bar()
    ax.set_title("Monthly Usage Pattern (Hourly-Mean Rate)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Usage Rate (0-1)")
    ax.yaxis.set_major_formatter(lambda x, pos: f"{x:.2f}")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "monthly_usage_pattern.png", dpi=300)
    plt.close()


# ────────────────────────────────────────────────────────────────────────────────
# DEVICE-LEVEL ANALYSIS FUNCTIONS
# ────────────────────────────────────────────────────────────────────────────────

def daily_device(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate daily consumption for individual devices with gap detection and redistribution."""
    if df.empty:
        return pd.DataFrame()
        
    df = df.sort_values(["device_id", "timestamp"])
    # Extract date from timestamp while preserving timezone information
    df["date"] = df["timestamp"].dt.date
    
    gap_threshold = pd.Timedelta(hours=CONFIG["energy_analysis"].get("gap_threshold_hours", 24))
    micro_gap_threshold = pd.Timedelta(hours=CONFIG["energy_analysis"].get("micro_gap_threshold_hours", 12))
    use_weighted = CONFIG["energy_analysis"].get("use_time_weighted_distribution", False)
    log_redistributed = CONFIG["energy_analysis"].get("log_redistributed_intervals", True)
    daily_max_percentile = CONFIG["energy_analysis"].get("daily_max_percentile", 0.95)
    constraint_tolerance = CONFIG["energy_analysis"].get("constraint_tolerance", 0.05)
    
    # Process each device separately to handle transmission gaps
    devices_processed = []
    flagged_devices = []
    
    for device_id, device_df in df.groupby("device_id"):
        # Sort by timestamp
        device_df = device_df.sort_values("timestamp")
        device_class = device_df["device_class"].iloc[0]
        
        # Calculate time differences between consecutive readings
        device_df = device_df.reset_index(drop=True)
        time_diffs = device_df["timestamp"].diff()
        
        # Identify gaps (both micro and large)
        large_gap_indices = time_diffs[time_diffs > gap_threshold].index.tolist()
        micro_gap_indices = time_diffs[(time_diffs > micro_gap_threshold) & (time_diffs <= gap_threshold)].index.tolist()
        gap_indices = sorted(large_gap_indices + micro_gap_indices)
        
        if not gap_indices:
            device_daily = (
                device_df.groupby(["date"])
                .agg(first=("energy", "first"), last=("energy", "last"))
                .reset_index()
            )
            device_daily["device_id"] = device_id
            device_daily["device_class"] = device_class
            device_daily["consumption"] = (device_daily["last"] - device_daily["first"]).clip(lower=0)
            device_daily["is_redistributed"] = False
            devices_processed.append(device_daily)
            continue
            
        # Calculate historical percentiles for this device if we have enough data
        historical_device_data = []
        for segment_idx in range(len(gap_indices) + 1):
            start_idx = 0 if segment_idx == 0 else gap_indices[segment_idx - 1]
            end_idx = len(device_df) if segment_idx == len(gap_indices) else gap_indices[segment_idx]
            segment = device_df.loc[start_idx:end_idx-1]
            
            if len(segment) > 1:
                segment_daily = segment.groupby(["date"]).agg(
                    first=("energy", "first"), 
                    last=("energy", "last")
                ).reset_index()
                segment_daily["consumption"] = (segment_daily["last"] - segment_daily["first"]).clip(lower=0)
                historical_device_data.append(segment_daily["consumption"])
        
        # Calculate device-specific percentile if we have enough data
        if historical_device_data and pd.concat(historical_device_data).count() >= 5:
            device_max = pd.concat(historical_device_data).quantile(daily_max_percentile)
        else:
            device_max = None  # Will be calculated after all devices are processed
            
        # Process each segment (between gaps)
        segments = []
        start_idx = 0
        total_expected_consumption = 0
        
        for gap_idx in gap_indices + [len(device_df)]:
            segment = device_df.loc[start_idx:gap_idx-1]
            
            if not segment.empty and start_idx > 0:
                gap_start_ts = device_df.loc[start_idx-1, "timestamp"]
                gap_end_ts = device_df.loc[start_idx, "timestamp"]
                gap_duration = gap_end_ts - gap_start_ts
                
                # Calculate consumption over the gap
                gap_consumption = device_df.loc[start_idx, "energy"] - device_df.loc[start_idx-1, "energy"]
                total_expected_consumption += gap_consumption
                
                if gap_consumption > 0:
                    # Create a date range for the gap period
                    gap_dates = pd.date_range(
                        gap_start_ts.date() + pd.Timedelta(days=1),  # Start from next day
                        gap_end_ts.date(),                           # End at gap end date
                        freq="D"
                    )
                    
                    special_audit = CONFIG["energy_analysis"].get("special_audit_period", {})
                    special_start = special_audit.get("start")
                    special_end = special_audit.get("end")
                    
                    if special_start and special_end:
                        special_start_date = pd.Timestamp(special_start).date()
                        special_end_date = pd.Timestamp(special_end).date()
                        
                        # Check if gap overlaps with special audit period
                        is_special_period = (
                            (gap_start_ts.date() <= special_end_date and gap_end_ts.date() >= special_start_date)
                        )
                        
                        if is_special_period:
                            log.info(f"Special audit: gap for device {device_id} ({gap_start_ts.date()} to {gap_end_ts.date()})")
                            
                            if gap_end_ts.date() > special_end_date:
                                gap_dates = gap_dates[gap_dates.date <= special_end_date]
                                log.info(f"Restricting redistribution to special period end: {special_end_date}")
                                
                            if gap_start_ts.date() < special_start_date:
                                gap_dates = gap_dates[gap_dates.date >= special_start_date]
                                log.info(f"Restricting redistribution to special period start: {special_start_date}")
                    
                    if len(gap_dates) > 0:
                        daily_consumption = gap_consumption / len(gap_dates)
                        
                        # Apply percentile cap if available
                        if device_max is not None:
                            # Create distribution dataframe
                            distributed = pd.DataFrame({
                                "date": gap_dates.date,
                                "device_id": device_id,
                                "device_class": device_class,
                                "consumption": daily_consumption,
                                "is_redistributed": True,
                                "was_gap": True,
                                "first": float('nan'),
                                "last": float('nan'),
                                "hist_max": device_max
                            })
                            
                            # Identify days exceeding cap
                            over_cap = distributed[distributed["consumption"] > device_max]
                            if not over_cap.empty:
                                # Calculate excess to redistribute
                                total_excess = (over_cap["consumption"] - device_max).sum()
                                remaining_days = len(distributed) - len(over_cap)
                                
                                if remaining_days > 0:
                                    additional_per_day = total_excess / remaining_days
                                    distributed.loc[distributed["consumption"] <= device_max, "consumption"] += additional_per_day
                                    distributed.loc[distributed["consumption"] > device_max, "consumption"] = device_max
                                else:
                                    distributed["consumption"] = device_max
                        else:
                            distributed = pd.DataFrame({
                                "date": gap_dates.date,
                                "device_id": device_id,
                                "device_class": device_class,
                                "consumption": daily_consumption,
                                "is_redistributed": True,
                                "was_gap": True,
                                "first": float('nan'),
                                "last": float('nan'),
                                "hist_max": float('nan')
                            })
                        
                        segments.append(distributed)
                        
                        if log_redistributed:
                            log.info(
                                f"Redistributed {gap_consumption:.2f} kWh over {len(gap_dates)} days "
                                f"for device {device_id} ({gap_start_ts.date()} to {gap_end_ts.date()})"
                            )
            
            # Process regular daily consumption for this segment
            if len(segment) > 0:
                segment_daily = (
                    segment.groupby(["date"])
                    .agg(first=("energy", "first"), last=("energy", "last"))
                    .reset_index()
                )
                segment_daily["device_id"] = device_id
                segment_daily["device_class"] = device_class
                segment_daily["consumption"] = (segment_daily["last"] - segment_daily["first"]).clip(lower=0)
                segment_daily["is_redistributed"] = False
                segment_daily["was_gap"] = False
                
                # Calculate expected consumption for this segment
                total_expected_consumption += segment_daily["consumption"].sum()
                
                if device_max is not None:
                    segment_daily["hist_max"] = device_max
                else:
                    segment_daily["hist_max"] = float('nan')
                
                segments.append(segment_daily)
                
            start_idx = gap_idx
        
        if segments:
            device_result = pd.concat(segments, ignore_index=True)
            
            total_actual_consumption = device_result["consumption"].sum()
            delta_from_expected = abs(total_actual_consumption - total_expected_consumption)
            
            if delta_from_expected > (constraint_tolerance * total_expected_consumption):
                log.warning(
                    f"Net-delta cross-check failed for device {device_id}: "
                    f"expected={total_expected_consumption:.2f}, actual={total_actual_consumption:.2f}, "
                    f"delta={delta_from_expected:.2f}"
                )
                device_result["delta_from_expected"] = total_expected_consumption - total_actual_consumption
                flagged_devices.append(device_id)
            else:
                device_result["delta_from_expected"] = 0
                
            devices_processed.append(device_result)
    
    if not devices_processed:
        return pd.DataFrame()
        
    result_df = pd.concat(devices_processed, ignore_index=True)
    
    # Handle potential anomalies (extremely high values)
    # Identify extreme outliers (more than 3 IQRs above Q3)
    q1, q3 = result_df["consumption"].quantile([0.25, 0.75])
    iqr = q3 - q1
    upper_bound = q3 + 3 * iqr
    
    # Cap extreme values
    if (result_df["consumption"] > upper_bound).any():
        log.warning("Capping %d extreme daily consumption values", (result_df["consumption"] > upper_bound).sum())
        result_df["consumption"] = result_df["consumption"].clip(upper=upper_bound)
    
    # Identify any remaining values exceeding historical max
    result = result_df.sort_values(["device_id", "date"])
    
    # Process each device separately for spike detection
    for device_id, device_data in result.groupby("device_id"):
        if "hist_max" in device_data.columns and not device_data["hist_max"].isna().all():
            device_max = device_data["hist_max"].iloc[0]  # Get the historical max for this device
            
            over_max_indices = device_data[(device_data["consumption"] > device_max * 1.1) & 
                                         (~device_data["is_redistributed"])].index
            if len(over_max_indices) > 0:
                log.warning(f"Capping {len(over_max_indices)} values exceeding {device_max * 1.1:.2f} kWh for device {device_id}")
                for idx in over_max_indices:
                    result.loc[idx, "consumption"] = device_max
                    result.loc[idx, "is_redistributed"] = True
            
            spike_indices = device_data[(device_data["consumption"] > device_max * 1.5) & 
                                      (~device_data["is_redistributed"])].index
            
            for idx in spike_indices:
                spike_date = result.loc[idx, "date"]
                spike_value = result.loc[idx, "consumption"]
                
                log.warning(f"Post-processing: Found spike for {device_id} on {spike_date}: {spike_value:.2f} kWh")
                
                device_dates = device_data["date"].unique()
                adjacent_dates = []
                
                for offset in [-1, 1]:
                    target_date = spike_date + pd.Timedelta(days=offset)
                    if target_date in device_dates:
                        adjacent_dates.append(target_date)
                
                if adjacent_dates:
                    # Calculate redistribution amount
                    redistribution_days = len(adjacent_dates) + 1
                    redistributed_value = spike_value / redistribution_days
                    
                    result.loc[idx, "consumption"] = redistributed_value
                    result.loc[idx, "is_redistributed"] = True
                    result.loc[idx, "was_gap"] = True
                    
                    for adj_date in adjacent_dates:
                        adj_idx = result[(result["device_id"] == device_id) & 
                                       (result["date"] == adj_date)].index
                        if not adj_idx.empty:
                            result.loc[adj_idx[0], "consumption"] += redistributed_value
                            result.loc[adj_idx[0], "is_redistributed"] = True
                            result.loc[adj_idx[0], "was_gap"] = True
                    
                    log.info(f"Redistributed spike of {spike_value:.2f} kWh over {redistribution_days} days for device {device_id}")

    return result.round(2)


def monthly_device(daily_dev: pd.DataFrame) -> pd.DataFrame:
    """Calculate monthly consumption for individual devices."""
    if daily_dev.empty:
        return pd.DataFrame()
        
    # Convert to datetime with timezone info before extracting period
    daily_dev["month"] = pd.to_datetime(daily_dev["date"]).dt.to_period("M")
    
    # Group by device and month
    m = daily_dev.groupby(["device_id", "device_class", "month"])["consumption"].sum().reset_index()

    # order chronologically
    m["month_start"] = m["month"].dt.to_timestamp()
    m = m.sort_values(["device_id", "month_start"])

    m["month_name"] = m["month"].dt.strftime("%B %Y")  # e.g. "March 2025"
    return m.round(2).drop(columns="month_start")


def weekday_device(daily_dev: pd.DataFrame) -> pd.DataFrame:
    """Calculate average consumption by weekday for individual devices."""
    if daily_dev.empty:
        return pd.DataFrame()
        
    # Convert to datetime with timezone info before extracting day name
    daily_dev["weekday"] = pd.to_datetime(daily_dev["date"]).dt.day_name()
    
    # Group by device and weekday
    w = daily_dev.groupby(["device_id", "device_class", "weekday"])["consumption"].mean().reset_index()

    # Set proper order for weekdays
    order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    w["weekday"] = pd.Categorical(w["weekday"], order, ordered=True)
    return w.sort_values(["device_id", "weekday"]).round(2)


def hourly_device_average(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Average hourly consumption for individual devices, separated by weekday/weekend.
    Focuses on the top_n devices by total consumption to avoid overwhelming visualizations.
    """
    if df.empty:
        return pd.DataFrame()
    
    # First, identify top devices by total consumption
    device_totals = df.groupby("device_id")["energy"].agg(lambda x: x.max() - x.min()).sort_values(ascending=False)
    top_devices = device_totals.head(top_n).index.tolist()
    
    # Filter dataframe to only include top devices
    df_top = df[df["device_id"].isin(top_devices)].copy()
        
    # Date and time information
    df_top = df_top.sort_values(["device_id", "timestamp"])
    df_top["date"] = df_top["timestamp"].dt.date
    df_top["hour_of_day"] = df_top["timestamp"].dt.hour
    # Add weekday/weekend flag
    df_top["is_weekend"] = df_top["timestamp"].dt.dayofweek >= 5  # 5=Saturday, 6=Sunday

    # First/last energy values by device-day-hour
    g = (
        df_top.groupby(
            ["device_id", "device_class", "date", "hour_of_day", "is_weekend"],
            observed=True
        )
        .agg(first=("energy", "first"), last=("energy", "last"))
        .reset_index()
    )
    g["delta"] = (g["last"] - g["first"]).clip(lower=0)
    
    # Handle outliers in hourly data
    q1, q3 = g["delta"].quantile([0.25, 0.75])
    iqr = q3 - q1
    upper_bound = q3 + 3 * iqr
    
    # Cap extreme values
    if (g["delta"] > upper_bound).any():
        log.warning("Capping %d extreme hourly consumption values", (g["delta"] > upper_bound).sum())
        g["delta"] = g["delta"].clip(upper=upper_bound)

    # Take average across days for each hour, weekend status, and device
    hourly = (
        g
        .groupby(["device_id", "device_class", "hour_of_day", "is_weekend"], observed=True)["delta"]
        .mean()
        .reset_index()
    )
    
    return hourly.round(2)

# ────────────────────────────────────────────────────────────────────────────────
# DEVICE-LEVEL VISUALIZATION FUNCTIONS
# ────────────────────────────────────────────────────────────────────────────────

def bar_device_total(dev_usage: pd.DataFrame, top_n: int = 15):
    """Create bar chart of consumption by device, showing top N devices."""
    if dev_usage.empty:
        log.warning("No data available for device-level bar chart")
        return
        
    # Sort by consumption and take top N
    top_devices = dev_usage.sort_values("consumption", ascending=False).head(top_n)
    
    # Add device class to labels
    top_devices["label"] = top_devices["device_id"] + " (" + top_devices["device_class"] + ")"
    
    # Create subplots to better organize by device class
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Color mapping based on device class
    class_colors = {
        "DB16": CONFIG["visualization"]["color_palette"][0],
        "DB18": CONFIG["visualization"]["color_palette"][1], 
        "DB14": CONFIG["visualization"]["color_palette"][2]
    }
    
    # Plot bars with colors based on device class
    bars = ax.bar(
        top_devices["label"], 
        top_devices["consumption"],
        color=[class_colors.get(cls, "#999999") for cls in top_devices["device_class"]]
    )
    
    ax.set_title(f"Top {top_n} Devices by Total Consumption")
    ax.set_xlabel("Device ID (Class)")
    ax.set_ylabel("Energy Consumption (kWh)")
    ax.set_xticklabels(top_devices["label"], rotation=45, ha="right")
    _fmt_y(ax)
    
    # Add legend for device classes
    handles = [plt.Rectangle((0,0), 1, 1, color=color) for color in class_colors.values()]
    labels = list(class_colors.keys())
    ax.legend(handles, labels, title="Device Class")
    
    plt.tight_layout()
    plt.savefig(OUT_DIR / "device_total_consumption.png", dpi=300)
    plt.close()


def line_daily_device(daily_dev: pd.DataFrame, top_n: int = 5):
    """Plot daily consumption trends for top N devices."""
    if daily_dev.empty:
        log.warning("No data available for device-level daily chart")
        return
        
    # Get top devices by total consumption
    top_devices = daily_dev.groupby("device_id")["consumption"].sum().sort_values(ascending=False).head(top_n)
    top_device_ids = top_devices.index.tolist()
    
    # Filter to just the top devices
    plot_data = daily_dev[daily_dev["device_id"].isin(top_device_ids)].copy()
    
    # Create separate DataFrames for regular and redistributed values
    regular_data = plot_data[~plot_data["is_redistributed"]]
    redistributed_data = plot_data[plot_data["is_redistributed"]]
    
    if not redistributed_data.empty:
        redistributed_keys = redistributed_data[["device_id", "date"]].drop_duplicates()
        regular_data = regular_data.merge(
            redistributed_keys, 
            on=["device_id", "date"], 
            how="left", 
            indicator=True
        )
        regular_data = regular_data[regular_data["_merge"] == "left_only"].drop("_merge", axis=1)
    
    combined_data = pd.concat([regular_data, redistributed_data])
    
    # Create pivot table: dates as index, devices as columns
    pivot_data = combined_data.pivot_table(
        index="date", 
        columns="device_id",
        values="consumption", 
        aggfunc="sum"
    ).fillna(0)
    
    window_size = 3
    for device in pivot_data.columns:
        pivot_data[device] = pivot_data[device].rolling(
            window=window_size, center=True, min_periods=1
        ).mean()
    
    # Get device classes for labeling
    device_classes = plot_data.groupby("device_id")["device_class"].first().to_dict()
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot each device with improved styling
    for device in pivot_data.columns:
        device_class = device_classes[device]
        ax.plot(
            pivot_data.index, 
            pivot_data[device], 
            linewidth=2,
            label=f"{device} ({device_class})"
        )
    
    ax.set_title(f"Daily Consumption Trend - Top {top_n} Devices")
    ax.set_xlabel("Date")
    ax.set_ylabel("Energy Consumption (kWh)")
    
    # Dynamic date axis for readability
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    
    ax.legend(title="Device ID (Class)")
    plt.xticks(rotation=45)
    _fmt_y(ax)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "daily_device_consumption.png", dpi=300)
    plt.close()


def plot_weekday_device(weekday_dev: pd.DataFrame, top_n: int = 5):
    """Plot weekday consumption patterns for top N devices."""
    if weekday_dev.empty:
        log.warning("No data available for device-level weekday chart")
        return
        
    # Get top devices by total consumption
    top_devices = weekday_dev.groupby("device_id")["consumption"].sum().sort_values(ascending=False).head(top_n)
    top_device_ids = top_devices.index.tolist()
    
    # Filter to just the top devices
    plot_data = weekday_dev[weekday_dev["device_id"].isin(top_device_ids)].copy()
    
    # Get device classes for labeling
    device_classes = plot_data.groupby("device_id")["device_class"].first().to_dict()
    
    # Set up subplots - one for each device
    fig, axes = plt.subplots(len(top_device_ids), 1, figsize=(12, 3*len(top_device_ids)), sharex=True)
    
    # If only one device, axes is not a list
    if len(top_device_ids) == 1:
        axes = [axes]
    
    for i, device_id in enumerate(top_device_ids):
        device_data = plot_data[plot_data["device_id"] == device_id]
        device_class = device_classes[device_id]
        
        # Color based on device class
        color_index = list(CONFIG["device_classes"].keys()).index(device_class) % len(CONFIG["visualization"]["color_palette"])
        color = CONFIG["visualization"]["color_palette"][color_index]
        
        axes[i].bar(
            device_data["weekday"], 
            device_data["consumption"],
            color=color
        )
        axes[i].set_title(f"{device_id} ({device_class})")
        axes[i].set_ylabel("kWh")
        _fmt_y(axes[i])
    
    # Set common labels
    axes[-1].set_xlabel("Weekday")
    fig.suptitle("Weekday Consumption Pattern by Device", fontsize=16)
    
    plt.tight_layout()
    plt.savefig(OUT_DIR / "weekday_device_pattern.png", dpi=300)
    plt.close()


def plot_hourly_device(hourly_dev: pd.DataFrame):
    """Plot hourly consumption patterns for devices, separated by weekday/weekend."""
    if hourly_dev.empty:
        log.warning("No data available for device-level hourly chart")
        return
        
    # Get unique devices
    devices = hourly_dev["device_id"].unique()
    
    if len(devices) > 8:
        log.info(f"Limiting hourly device plots to top 8 devices (out of {len(devices)})")
        devices = hourly_dev.groupby("device_id")["delta"].sum().sort_values(ascending=False).head(8).index.tolist()
    
    # Create two subplots side by side - weekday and weekend
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Get device classes for labeling
    device_classes = hourly_dev.groupby("device_id")["device_class"].first().to_dict()
    
    # Weekday plot
    weekday_data = hourly_dev[hourly_dev["is_weekend"] == False]
    for device in devices:
        device_data = weekday_data[weekday_data["device_id"] == device]
        if not device_data.empty:
            device_class = device_classes[device]
            ax1.plot(
                device_data["hour_of_day"], 
                device_data["delta"],
                marker='o', 
                linewidth=2,
                label=f"{device} ({device_class})"
            )
    
    ax1.set_title("Weekday Hourly Consumption by Device")
    ax1.set_xlabel("Hour of Day")
    ax1.set_ylabel("Energy Consumption (kWh)")
    ax1.set_xticks(range(0, 24))
    ax1.legend(title="Device ID (Class)")
    _fmt_y(ax1)
    
    # Weekend plot
    weekend_data = hourly_dev[hourly_dev["is_weekend"] == True]
    for device in devices:
        device_data = weekend_data[weekend_data["device_id"] == device]
        if not device_data.empty:
            device_class = device_classes[device]
            ax2.plot(
                device_data["hour_of_day"], 
                device_data["delta"],
                marker='s', 
                linewidth=2,
                label=f"{device} ({device_class})"
            )
    
    ax2.set_title("Weekend Hourly Consumption by Device")
    ax2.set_xlabel("Hour of Day")
    ax2.set_ylabel("Energy Consumption (kWh)")
    ax2.set_xticks(range(0, 24))
    ax2.legend(title="Device ID (Class)")
    _fmt_y(ax2)
    
    plt.tight_layout()
    plt.savefig(OUT_DIR / "hourly_device_consumption.png", dpi=300)
    plt.close()

# ────────────────────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────────────────────


def main(path: str | Path = "db16_db18_smartlife_feb15_apr15.json", 
         use_cache: bool = True, output_csv: bool = True):
    """
    Main analysis function.
    
    Args:
        path: Path to input JSON file
        use_cache: Whether to use cached data if available
        output_csv: Whether to output CSV files with processed data
    """
    start_time = datetime.datetime.now()
    data_loading_start = start_time
    log.info("Starting energy analysis...")
    
    # Use cached data if available and requested
    cache_dir = Path("data_cache")
    cache_dir.mkdir(exist_ok=True)
    
    cache_file = cache_dir / f"{Path(path).stem}_processed.pkl"
    quality_file = cache_dir / f"{Path(path).stem}_quality.json"
    
    used_cache = False
    
    if use_cache and cache_file.exists():
        try:
            log.info(f"Loading cached data from {cache_file}")
            df = pd.read_pickle(cache_file)
            
            # Load quality metrics
            if quality_file.exists():
                with open(quality_file, 'r') as f:
                    quality_dict = json.load(f)
                quality_metrics = DataQualityMetrics(**quality_dict)
            else:
                quality_metrics = DataQualityMetrics()
                
            log.info(f"Loaded {len(df)} records from cache")
            used_cache = True
        except Exception as e:
            log.error(f"Error loading cache: {e}")
            log.info("Loading raw data instead")
            df, quality_metrics = load_data(path)
            
            # Save to cache
            df.to_pickle(cache_file)
            with open(quality_file, 'w') as f:
                json.dump(quality_metrics.to_dict(), f, indent=4)
    else:
        # Load and process raw data
        df, quality_metrics = load_data(path)
        
        # Save to cache
        if not df.empty:
            log.info(f"Saving processed data to cache: {cache_file}")
            df.to_pickle(cache_file)
            with open(quality_file, 'w') as f:
                json.dump(quality_metrics.to_dict(), f, indent=4)
    
    data_loading_end = datetime.datetime.now()
    data_loading_time = data_loading_end - data_loading_start
    
    if df.empty:
        log.warning("No data after filtering – nothing to do")
        return

    # Print data quality summary
    quality_metrics.print_summary()

    # Process energy consumption data
    dev_usage, usage_metrics = per_device_usage(df)
    
    if dev_usage.empty:
        log.warning("No valid device usage data found")
        return
    
    # Save processed data to CSV if requested
    if output_csv:
        csv_dir = Path("csv_output")
        csv_dir.mkdir(exist_ok=True)
        dev_usage.to_csv(csv_dir / "device_usage.csv", index=False)
    
    # Calculate class-level statistics    
    tot = class_totals(dev_usage)
    daily = daily_class(df)
    
    # Calculate device-level statistics
    daily_dev = daily_device(df)
    monthly_dev = monthly_device(daily_dev)
    weekday_dev = weekday_device(daily_dev)
    hourly_dev = hourly_device_average(df)
    
    # Save device-level data to CSV if requested
    if output_csv:
        if not daily_dev.empty:
            save_columns = [col for col in daily_dev.columns if col != 'is_redistributed']
            daily_dev[save_columns].to_csv(csv_dir / "daily_device_consumption.csv", index=False)
            
            enable_diagnostic = CONFIG["energy_analysis"].get("enable_diagnostic_export", False)
            if enable_diagnostic:
                diagnostic_columns = [
                    "device_id", "date", "was_gap", "is_redistributed", 
                    "consumption", "hist_max", "delta_from_expected"
                ]
                diagnostic_cols = [col for col in diagnostic_columns if col in daily_dev.columns]
                daily_dev[diagnostic_cols].to_csv(csv_dir / "energy_diagnostic.csv", index=False)
                log.info(f"Exported diagnostic data to {csv_dir / 'energy_diagnostic.csv'}")
                
                flagged_devices = daily_dev[daily_dev["delta_from_expected"] != 0]["device_id"].unique()
                if len(flagged_devices) > 0:
                    log.warning(f"Found {len(flagged_devices)} devices with delta cross-check issues")
                    for device in flagged_devices:
                        log.warning(f"  - Device {device} flagged for review")
                        
        if not monthly_dev.empty:
            monthly_dev.to_csv(csv_dir / "monthly_device_consumption.csv", index=False)
        if not weekday_dev.empty:
            weekday_dev.to_csv(csv_dir / "weekday_device_consumption.csv", index=False)
        if not hourly_dev.empty:
            hourly_dev.to_csv(csv_dir / "hourly_device_consumption.csv", index=False)
    
    # Save to CSV if requested
    if output_csv and not daily.empty:
        daily.to_csv(csv_dir / "daily_consumption.csv", index=False)
    
    # Check if we have any device class data
    if not any(col in daily.columns for col in ['DB16', 'DB18', 'DB14']):
        log.warning("No device class data (DB16/DB18/DB14) found in processed data")
    
    m = monthly(daily.copy())
    w = weekday(daily.copy())
    h = hourly_average(df)
    
    # Save to CSV if requested
    if output_csv:
        if not m.empty:
            m.to_csv(csv_dir / "monthly_consumption.csv", index=False)
        if not w.empty:
            w.to_csv(csv_dir / "weekday_consumption.csv", index=False)
        if not h.empty:
            h.to_csv(csv_dir / "hourly_consumption.csv", index=False)

    # -------- TERMİNAL İSTATİSTİK ÇIKTILARI --------
    print("\n" + "="*80)
    print(" "*30 + "ENERJİ ANALİZ SONUÇLARI")
    print("="*80)
    
    # 1. Toplam tüketim istatistikleri
    print("\n1. TOPLAM ENERJİ TÜKETİMİ (kWh)")
    print("-"*50)
    if not tot.empty:
        for _, row in tot.iterrows():
            print(f"{row['device_class']:>10}: {row['consumption']:,.2f} kWh")
        total_consumption = tot['consumption'].sum()
        print(f"{'TOPLAM':>10}: {total_consumption:,.2f} kWh")
    else:
        print("Toplam tüketim verisi bulunamadı.")
    
    # 1.1 Cihaz bazında tüketim
    print("\n1.1 CİHAZ BAZINDA TÜKETİM (kWh)")
    print("-"*70)
    if not dev_usage.empty:
        # Tüketim değerine göre sırala (yüksekten düşüğe)
        sorted_devices = dev_usage.sort_values('consumption', ascending=False)
        print(f"{'Cihaz ID':>20} | {'Sınıf':>5} | {'Tüketim':>10} | {'İlk Ölçüm':>12} | {'Son Ölçüm':>12}")
        print("-"*70)
        for _, row in sorted_devices.iterrows():
            device_id = row['device_id']
            device_class = row['device_class']
            consumption = row['consumption']
            first = row['first_energy']
            last = row['last_energy']
            print(f"{device_id:>20} | {device_class:>5} | {consumption:>10,.2f} | {first:>12,.2f} | {last:>12,.2f}")
    else:
        print("Cihaz bazında tüketim verisi bulunamadı.")
    
    # 2. Aylık tüketim istatistikleri
    print("\n2. AYLIK ENERJİ TÜKETİMİ (kWh)")
    print("-"*50)
    if not m.empty:
        for _, row in m.iterrows():
            month_name = row['month']
            row_values = [f"{row.get(col, 0):,.2f}" for col in ['DB16', 'DB18', 'DB14'] if col in m.columns]
            print(f"{month_name:>15}: {' | '.join(row_values)}")
    else:
        print("Aylık tüketim verisi bulunamadı.")
    
    # 3. Haftanın günleri tüketim ortalaması
    print("\n3. HAFTA İÇİ/SONU ORTALAMA TÜKETİM (kWh)")
    print("-"*50)
    if not w.empty:
        for _, row in w.iterrows():
            day_name = row['weekday']
            row_values = [f"{row.get(col, 0):,.2f}" for col in ['DB16', 'DB18', 'DB14'] if col in w.columns]
            print(f"{day_name:>10}: {' | '.join(row_values)}")
    else:
        print("Haftanın günlerine göre tüketim verisi bulunamadı.")
    
    # 4. Saatlik tüketim ortalaması (hafta içi/sonu)
    print("\n4. SAATLİK ORTALAMA TÜKETİM (kWh)")
    print("-"*70)
    if 'hour_of_day' in h.columns:
        # Çıktı başlığı
        headers = ['Saat']
        for col in h.columns:
            if col != 'hour_of_day':
                if 'Weekday' in col:
                    headers.append(f"{col.split('_')[0]} Hafta İçi")
                elif 'Weekend' in col:
                    headers.append(f"{col.split('_')[0]} Hafta Sonu")
                else:
                    headers.append(col)
        
        print(f"{headers[0]:>6}", end="")
        for header in headers[1:]:
            print(f" | {header:>13}", end="")
        print("\n" + "-"*70)
        
        # Saatlik veriler
        for _, row in h.iterrows():
            hour = int(row['hour_of_day'])
            print(f"{hour:>6}", end="")
            for col in h.columns:
                if col != 'hour_of_day':
                    print(f" | {row[col]:>13.2f}", end="")
            print()
    else:
        print("Saatlik tüketim verisi bulunamadı.")
    
    # 5. Cihaz sınıfına göre cihaz sayısı 
    print("\n5. CİHAZ SINIFI BAZINDA CİHAZ SAYISI")
    print("-"*50)
    device_counts = dev_usage.groupby('device_class').size()
    for cls, count in device_counts.items():
        print(f"{cls:>10}: {count} cihaz")
    
    # 6. Analiz penceresi bilgisi
    print("\n6. ANALİZ PENCERESİ")
    print("-"*50)
    print(f"Başlangıç: {WINDOW.start.strftime('%d %B %Y, %H:%M:%S')}")
    print(f"Bitiş    : {WINDOW.end.strftime('%d %B %Y, %H:%M:%S')}")
    
    print("\n" + "="*80 + "\n")
    
    # Generate class-level visualizations
    log.info("Generating class-level charts → %s", OUT_DIR.resolve())
    
    # Only generate charts if we have data
    if not tot.empty:
        bar_total(tot)
    
    line_daily(daily)
    
    if not m.empty:
        bar_month(m)
    
    if not w.empty:
        bar_weekday(w)
    
    line_hourly(h)
    
    # Generate device-level visualizations
    log.info("Generating device-level charts → %s", OUT_DIR.resolve())
    
    if not dev_usage.empty:
        bar_device_total(dev_usage)
    
    if not daily_dev.empty:
        line_daily_device(daily_dev)
    
    if not weekday_dev.empty:
        plot_weekday_device(weekday_dev)
    
    if not hourly_dev.empty:
        plot_hourly_device(hourly_dev)
    
    # Process light usage data
    try:
        df_light = load_light_df(path)
        if not df_light.empty:
            hourly_cls = hourly_class_usage(df_light)
            daily_cls = daily_usage(hourly_cls)
            week_cls = weekly_pattern(daily_cls)
            month_cls = monthly_pattern(daily_cls)

            # Save light usage data to CSV if requested
            if output_csv:
                if not daily_cls.empty:
                    daily_cls.to_csv(csv_dir / "daily_light_usage.csv")
                if not week_cls.empty:
                    week_cls.to_csv(csv_dir / "weekly_light_usage.csv")
                if not month_cls.empty:
                    month_cls.to_csv(csv_dir / "monthly_light_usage.csv")

            # Terminal çıktısında aydınlatma kullanım oranı
            print("\n7. AYDINLATMA KULLANIM ORANLARI (0-1)")
            print("-"*50)
            print("Haftanın Günlerine Göre:")
            print(week_cls.round(2))
            print("\nAylara Göre:")
            print(month_cls.round(2))
            print("\n" + "="*80 + "\n")

            bar_weekly_usage(week_cls)
            bar_monthly_usage(month_cls)
        else:
            log.warning("No light data found")
    except Exception as e:
        log.error("Error processing light data: %s", e)
    
    # Add statistical comparisons
    if not daily.empty and 'DB16' in daily.columns and 'DB18' in daily.columns:
        try:
            from scipy import stats
            
            # Compare weekday vs weekend consumption
            weekday_mask = pd.to_datetime(daily['date']).dt.dayofweek < 5
            weekend_mask = ~weekday_mask
            
            weekday_db16 = daily.loc[weekday_mask, 'DB16']
            weekend_db16 = daily.loc[weekend_mask, 'DB16']
            
            weekday_db18 = daily.loc[weekday_mask, 'DB18']
            weekend_db18 = daily.loc[weekend_mask, 'DB18']
            
            print("\n8. İSTATİSTİKSEL KARŞILAŞTIRMALAR")
            print("-"*50)
            
            # DB16 weekday vs weekend
            t_stat, p_value = stats.ttest_ind(weekday_db16, weekend_db16, equal_var=False)
            print(f"DB16 Hafta içi vs Hafta sonu: t={t_stat:.2f}, p={p_value:.4f}")
            if p_value < 0.05:
                print(f"  Sonuç: İstatistiksel olarak anlamlı fark var (p<0.05)")
            else:
                print(f"  Sonuç: İstatistiksel olarak anlamlı fark yok (p≥0.05)")
                
            # DB18 weekday vs weekend
            t_stat, p_value = stats.ttest_ind(weekday_db18, weekend_db18, equal_var=False)
            print(f"DB18 Hafta içi vs Hafta sonu: t={t_stat:.2f}, p={p_value:.4f}")
            if p_value < 0.05:
                print(f"  Sonuç: İstatistiksel olarak anlamlı fark var (p<0.05)")
            else:
                print(f"  Sonuç: İstatistiksel olarak anlamlı fark yok (p≥0.05)")
                
            # DB16 vs DB18 comparison
            t_stat, p_value = stats.ttest_ind(daily['DB16'], daily['DB18'], equal_var=False)
            print(f"DB16 vs DB18 günlük tüketim: t={t_stat:.2f}, p={p_value:.4f}")
            if p_value < 0.05:
                print(f"  Sonuç: İki cihaz sınıfı arasında istatistiksel olarak anlamlı fark var (p<0.05)")
            else:
                print(f"  Sonuç: İki cihaz sınıfı arasında istatistiksel olarak anlamlı fark yok (p≥0.05)")
                
            print("\n" + "="*80 + "\n")
        except ImportError:
            log.warning("scipy not installed - skipping statistical comparisons")
            log.info("To install scipy: pip install scipy")
            
    # Display execution time
    end_time = datetime.datetime.now()
    execution_time = end_time - start_time
    data_loading_time_seconds = data_loading_time.total_seconds()
    total_time_seconds = execution_time.total_seconds()
    analysis_time_seconds = total_time_seconds - data_loading_time_seconds
    
    log.info(f"Analysis completed in {total_time_seconds:.2f} seconds")
    log.info(f"  Data loading: {data_loading_time_seconds:.2f} seconds {'(cached)' if used_cache else '(from raw)'}")
    log.info(f"  Data analysis: {analysis_time_seconds:.2f} seconds")
    
    if used_cache:
        # Estimate time saved using cache
        raw_loading_estimate = 85.0  # approximate seconds to load raw data
        cache_savings = raw_loading_estimate - data_loading_time_seconds
        log.info(f"  Cache performance: Saved approximately {cache_savings:.1f} seconds")
    
    log.info("Done. Charts saved in %s", OUT_DIR.resolve())
    if output_csv:
        log.info("CSV files saved in %s", csv_dir.resolve())

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Energy consumption analysis")
    parser.add_argument("--input", "-i", default="db16_db18_smartlife_feb15_apr15.json",
                      help="Input JSON file path")
    parser.add_argument("--no-cache", action="store_true", 
                      help="Disable data caching")
    parser.add_argument("--no-csv", action="store_true",
                      help="Disable CSV output")
    parser.add_argument("--no-device-analysis", action="store_true",
                      help="Disable device-level analysis")
    args = parser.parse_args()
    
    # Check if scipy is installed for statistical analysis
    try:
        import scipy
        log.info("scipy is installed - statistical comparisons will be included")
    except ImportError:
        log.warning("scipy not installed - statistical comparisons will be skipped")
        log.info("To install scipy: pip install scipy")
    
    main(args.input, use_cache=not args.no_cache, output_csv=not args.no_csv)
