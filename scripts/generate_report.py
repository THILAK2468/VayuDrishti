"""
Step 12: Comprehensive Reporting Module for VayuDrishti.
Generates:
  - metadata/dataset_summary.txt (human-readable executive summary)
  - metadata/class_distribution_report.csv (multi-split class distribution analysis)
"""
import os
import sys
import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import load_config, get_path, BASE_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("generate_report")

def generate_reports(config=None):
    if config is None:
        config = load_config()
        
    logger.info("Compiling comprehensive dataset quality and summary reports...")
    
    # File paths
    matched_path = get_path("matched_samples", config)
    quality_path = get_path("data_quality_report", config)
    summary_path = get_path("dataset_summary", config)
    class_dist_path = get_path("class_distribution_report", config)
    splits_dir = get_path("splits_dir", config)
    era5_catalog_path = get_path("era5_catalog", config)
    image_catalog_path = get_path("image_catalog", config)
    norm_params_path = get_path("normalization_params", config)
    
    # Load dataframes
    df_matched = pd.read_csv(matched_path)
    df_quality = pd.read_csv(quality_path)
    df_era5_cat = pd.read_csv(era5_catalog_path)
    df_sat_cat = pd.read_csv(image_catalog_path) if os.path.exists(image_catalog_path) else pd.DataFrame()
    
    train_df = pd.read_csv(os.path.join(splits_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(splits_dir, "validation.csv"))
    test_df = pd.read_csv(os.path.join(splits_dir, "test.csv"))
    
    # 1. Class Distribution Analysis
    all_classes = config.get("intensity_policy", {}).get("thresholds", [])
    class_order = [t["class"] for t in all_classes] + ["UNKNOWN"]
    
    dist_rows = []
    total_usable = len(df_matched[df_matched["quality_status"] != "INVALID"])
    
    for cls in class_order:
        tot_cnt = int((df_matched[df_matched["quality_status"] != "INVALID"]["intensity_class"] == cls).sum())
        tr_cnt = int((train_df["intensity_class"] == cls).sum())
        va_cnt = int((val_df["intensity_class"] == cls).sum())
        te_cnt = int((test_df["intensity_class"] == cls).sum())
        
        if tot_cnt > 0:
            dist_rows.append({
                "intensity_class": cls,
                "total_count": tot_cnt,
                "total_percentage": round((tot_cnt / total_usable) * 100.0, 2),
                "train_count": tr_cnt,
                "train_percentage": round((tr_cnt / len(train_df)) * 100.0, 2) if len(train_df) else 0.0,
                "val_count": va_cnt,
                "val_percentage": round((va_cnt / len(val_df)) * 100.0, 2) if len(val_df) else 0.0,
                "test_count": te_cnt,
                "test_percentage": round((te_cnt / len(test_df)) * 100.0, 2) if len(test_df) else 0.0
            })
            
    df_class_dist = pd.DataFrame(dist_rows)
    df_class_dist.to_csv(class_dist_path, index=False)
    logger.info(f"Saved class distribution report to: {class_dist_path}")
    
    # 2. Compile metrics for dataset_summary.txt
    total_storms_tracked = int(df_matched["storm_name"].nunique())
    total_candidate_samples = len(df_matched)
    invalid_samples = int((df_matched["quality_status"] == "INVALID").sum())
    warning_samples = int((df_matched["quality_status"] == "WARNING").sum())
    valid_samples = int((df_matched["quality_status"] == "VALID").sum())
    
    # Samples per storm
    storm_stats = []
    for sname, grp in df_matched.groupby("storm_name"):
        tot_s = len(grp)
        use_s = len(grp[grp["quality_status"] != "INVALID"])
        inv_s = len(grp[grp["quality_status"] == "INVALID"])
        w_min = grp["wind_speed"].min()
        w_max = grp["wind_speed"].max()
        p_min = grp["pressure"].min()
        p_max = grp["pressure"].max()
        storm_stats.append({
            "name": sname,
            "total": tot_s,
            "usable": use_s,
            "invalid": inv_s,
            "wind_range": f"{w_min:.0f}-{w_max:.0f} kt" if pd.notna(w_min) else "N/A",
            "pres_range": f"{p_min:.0f}-{p_max:.0f} hPa" if pd.notna(p_min) else "N/A",
            "time_start": grp["datetime"].min()[:16],
            "time_end": grp["datetime"].max()[:16]
        })
        
    # Rejection reasons breakdown
    rejections = df_quality[df_quality["quality_status"] == "INVALID"]["quality_reasons"].value_counts().to_dict()
    warning_reasons = df_quality[df_quality["quality_status"] == "WARNING"]["quality_reasons"].value_counts().to_dict()
    
    # Splits breakdown
    tr_storms = sorted(train_df["storm_name"].unique())
    va_storms = sorted(val_df["storm_name"].unique())
    te_storms = sorted(test_df["storm_name"].unique())
    
    # Load normalization params
    norm_info = {}
    if os.path.exists(norm_params_path):
        with open(norm_params_path, "r", encoding="utf-8") as f:
            norm_info = json.load(f)
            
    # Geographic coverage
    lat_min = df_matched["latitude"].min()
    lat_max = df_matched["latitude"].max()
    lon_min = df_matched["longitude"].min()
    lon_max = df_matched["longitude"].max()
    
    # Format dataset_summary.txt
    lines = []
    lines.append("=" * 80)
    lines.append("                     VAYUDRISHTI DATASET SUMMARY REPORT")
    lines.append("=" * 80)
    lines.append(f"Project Name:        {config.get('project_name', 'VayuDrishti')}")
    lines.append(f"Version:             {config.get('version', '1.0.0')}")
    lines.append(f"Basin:               North Indian Ocean (Arabian Sea & Bay of Bengal)")
    lines.append(f"Authoritative Track: IBTrACS v04r01 (WMO / IMD / JTWC consensus)")
    lines.append(f"Atmospheric Source:  ECMWF ERA5 Reanalysis (0.25 deg, hourly)")
    lines.append(f"Satellite Source:    INSAT-3D Imager (Supported: HDF5, NetCDF, GeoTIFF, JPG)")
    lines.append("-" * 80)
    lines.append("")
    lines.append("1. EXECUTIVE SAMPLE COUNTS")
    lines.append("--------------------------")
    lines.append(f"Total Cyclones Tracked:       {total_storms_tracked}")
    lines.append(f"Total Candidate Samples:      {total_candidate_samples}")
    lines.append(f"Strictly Valid Samples:       {valid_samples}")
    lines.append(f"Warning Samples (Usable):     {warning_samples}")
    lines.append(f"Rejected / Invalid Samples:   {invalid_samples}")
    lines.append(f"Total Usable ML Samples:      {total_usable}")
    lines.append("")
    lines.append("2. CYCLONE STORM INVENTORY")
    lines.append("--------------------------")
    lines.append(f"{'Storm':<10} {'Total':<7} {'Usable':<8} {'Invalid':<9} {'Wind Range':<14} {'Pres Range':<16} {'Time Window'}")
    lines.append("-" * 80)
    for s in storm_stats:
        lines.append(f"{s['name']:<10} {s['total']:<7} {s['usable']:<8} {s['invalid']:<9} {s['wind_range']:<14} {s['pres_range']:<16} {s['time_start']} to {s['time_end']}")
    lines.append("")
    lines.append("3. DATASET PARTITIONING (STORM-LEVEL LEAKAGE-FREE SPLIT)")
    lines.append("-------------------------------------------------------")
    lines.append(f"Partition Policy: Strictly Storm-Level (Zero storm or sample overlap)")
    lines.append(f"Target Split:     70.0% Train / 15.0% Validation / 15.0% Test")
    lines.append(f"Train Split:      {len(train_df):>4} samples ({len(train_df)/total_usable*100:5.1f}%) | Storms: {tr_storms}")
    lines.append(f"Validation Split: {len(val_df):>4} samples ({len(val_df)/total_usable*100:5.1f}%) | Storms: {va_storms}")
    lines.append(f"Test Split:       {len(test_df):>4} samples ({len(test_df)/total_usable*100:5.1f}%) | Storms: {te_storms}")
    lines.append("")
    lines.append("4. CLASS DISTRIBUTION & IMBALANCE ANALYSIS")
    lines.append("-------------------------------------------")
    lines.append(f"{'Intensity Class':<18} {'Total':<8} {'Total %':<10} {'Train':<8} {'Train %':<10} {'Val':<8} {'Test'}")
    lines.append("-" * 80)
    for _, r in df_class_dist.iterrows():
        lines.append(f"{r['intensity_class']:<18} {r['total_count']:<8} {r['total_percentage']:>6.1f}%    {r['train_count']:<8} {r['train_percentage']:>6.1f}%    {r['val_count']:<8} {r['test_count']}")
    lines.append("")
    lines.append("5. SPATIAL & TEMPORAL COVERAGE")
    lines.append("------------------------------")
    lines.append(f"Geographic Bounding Box: Latitude [{lat_min:.2f}N to {lat_max:.2f}N], Longitude [{lon_min:.2f}E to {lon_max:.2f}E]")
    lines.append(f"Atmospheric Domain:      Latitude [0.00N to 30.00N], Longitude [40.00E to 100.00E]")
    lines.append(f"Patch Extraction Box:    {config.get('spatial', {}).get('crop_size_deg', 10.0)} deg x {config.get('spatial', {}).get('crop_size_deg', 10.0)} deg synoptic bounding patch")
    lines.append(f"Temporal Extent:         2021-05-14 to 2025-12-02 (covering 2021, 2022, 2023, 2024, 2025 seasons)")
    lines.append(f"Time Synchronization:    UTC Standardized (Max tolerance: {config.get('temporal', {}).get('max_temporal_tolerance_minutes', 60)} minutes)")
    lines.append("")
    lines.append("6. VARIABLES & FEATURES")
    lines.append("-----------------------")
    lines.append("Core ERA5 Atmospheric Variables:")
    lines.append("  - u10: 10m U-component of wind [m/s]")
    lines.append("  - v10: 10m V-component of wind [m/s]")
    lines.append("  - t2m: 2m temperature [K]")
    lines.append("  - msl: Mean sea level pressure [Pa]")
    lines.append("  - sst: Sea surface temperature [K]")
    lines.append("Engineered / Derived Atmospheric Variables:")
    lines.append("  - wind_speed_10m: sqrt(u10^2 + v10^2) [m/s]")
    lines.append("  - wind_direction_10m: atan2(v10, u10) [radians]")
    lines.append("  - wind_direction_10m_deg: Meteorological wind direction [0-360 degrees]")
    lines.append("  - msl_hpa: Mean sea level pressure converted to [hPa] (recorded conversion)")
    lines.append("Ground Truth Cyclone Targets:")
    lines.append("  - Continuous: wind_speed [kt], pressure [hPa]")
    lines.append("  - Categorical: intensity_class (IMD 8-tier North Indian Ocean Scale)")
    lines.append("")
    lines.append("7. NORMALIZATION METHODOLOGY (LEAKAGE PREVENTION)")
    lines.append("-------------------------------------------------")
    lines.append("Policy: Normalization statistics (Mean, Std, Min, Max) calculated STRICTLY on TRAIN.")
    lines.append(f"Fitted on Training Storms: {norm_info.get('splits_source', {}).get('train_storms', [])}")
    lines.append("Features Normalized:")
    for f_name, f_stats in norm_info.get("features", {}).items():
        lines.append(f"  - {f_name:<22}: mean = {f_stats['mean']:>10.2f}, std = {f_stats['std']:>10.2f}, min = {f_stats['min']:>10.2f}, max = {f_stats['max']:>10.2f}")
    lines.append("")
    lines.append("8. QUALITY CONTROL & REJECTION AUDIT")
    lines.append("------------------------------------")
    lines.append("Rejected Samples (INVALID):")
    if rejections:
        for r_name, r_cnt in rejections.items():
            lines.append(f"  - {r_name}: {r_cnt} samples")
    else:
        lines.append("  - None")
    lines.append("Warning Samples (Documented Non-Fatal Notices):")
    if warning_reasons:
        for w_name, w_cnt in warning_reasons.items():
            lines.append(f"  - {w_name}: {w_cnt} samples")
    lines.append("")
    lines.append("=" * 80)
    lines.append("                     END OF VAYUDRISHTI DATASET REPORT")
    lines.append("=" * 80)
    
    summary_text = "\n".join(lines)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text)
    logger.info(f"Saved human-readable dataset summary to: {summary_path}")
    
    return summary_text

if __name__ == "__main__":
    generate_reports()
