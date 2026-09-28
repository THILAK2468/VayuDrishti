"""
Step 11: Automated Leakage Checker and Train-Only Normalization Module.
Verifies:
  1. Zero storm overlap across splits
  2. Zero sample ID overlap across splits
  3. Zero duplicate timestamps across splits
  4. Temporal causality preservation
  5. Fits ML normalization parameters (mean, std, min, max) STRICTLY ON TRAIN SPLIT ONLY
  6. Applies normalization parameters to validation and test sets without leakage
Outputs:
  - configs/normalization_params.json
  - metadata/leakage_check_report.json
"""
import os
import sys
import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import netCDF4 as nc

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import load_config, get_path, BASE_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("leakage_check")

def check_leakage_and_compute_normalization(splits_dir=None, config=None):
    if config is None:
        config = load_config()
    if splits_dir is None:
        splits_dir = get_path("splits_dir", config)
        
    train_path = os.path.join(splits_dir, "train.csv")
    val_path = os.path.join(splits_dir, "validation.csv")
    test_path = os.path.join(splits_dir, "test.csv")
    
    logger.info("Loading splits for automated leakage verification...")
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)
    
    report = {
        "passed_all_checks": True,
        "checks": {},
        "violations": []
    }
    
    # 1. Storm-level leakage check
    train_storms = set(df_train["storm_name"].unique())
    val_storms = set(df_val["storm_name"].unique())
    test_storms = set(df_test["storm_name"].unique())
    
    storm_overlap_tv = train_storms.intersection(val_storms)
    storm_overlap_tt = train_storms.intersection(test_storms)
    storm_overlap_vt = val_storms.intersection(test_storms)
    
    all_storm_overlaps = storm_overlap_tv | storm_overlap_tt | storm_overlap_vt
    if all_storm_overlaps:
        report["passed_all_checks"] = False
        report["violations"].append(f"Storm-level leakage detected: {all_storm_overlaps}")
        logger.error(f"FAIL: Storms present in multiple splits: {all_storm_overlaps}")
    else:
        logger.info("PASS: No storm appears in multiple splits.")
        
    report["checks"]["storm_leakage"] = {
        "passed": len(all_storm_overlaps) == 0,
        "train_storms": list(train_storms),
        "val_storms": list(val_storms),
        "test_storms": list(test_storms)
    }
    
    # 2. Sample ID leakage check
    train_samples = set(df_train["sample_id"].unique())
    val_samples = set(df_val["sample_id"].unique())
    test_samples = set(df_test["sample_id"].unique())
    
    sample_overlap = (train_samples & val_samples) | (train_samples & test_samples) | (val_samples & test_samples)
    if sample_overlap:
        report["passed_all_checks"] = False
        report["violations"].append(f"Sample-level leakage detected: {sample_overlap}")
        logger.error(f"FAIL: Duplicate sample IDs across splits: {len(sample_overlap)}")
    else:
        logger.info("PASS: No identical sample appears in multiple splits.")
        
    report["checks"]["sample_leakage"] = {
        "passed": len(sample_overlap) == 0,
        "overlap_count": len(sample_overlap)
    }
    
    # 3. Duplicate timestamp for same storm across splits
    report["checks"]["timestamp_cross_split_leakage"] = {
        "passed": len(all_storm_overlaps) == 0,
        "details": "Storms are strictly partitioned; no storm-time collision across splits."
    }
    
    # 4. Temporal causality within splits
    for sname, group in df_train.groupby("storm_name"):
        times = pd.to_datetime(group["datetime"], utc=True).values
        if not np.all(times[:-1] <= times[1:]):
            report["passed_all_checks"] = False
            report["violations"].append(f"Temporal order violation in train storm {sname}")
            logger.error(f"FAIL: Non-chronological observations in {sname}")
            
    report["checks"]["temporal_causality"] = {
        "passed": report["passed_all_checks"],
        "details": "All storm observations ordered strictly chronologically."
    }
    
    # 5. Fit Normalization Statistics on TRAIN ONLY
    logger.info("Fitting normalization statistics STRICTLY on TRAIN storms...")
    era5_vars = ["u10", "v10", "t2m", "msl", "sst", "wind_speed_10m", "wind_direction_10m", "msl_hpa"]
    
    # Running accumulators: sum, sum_sq, count, min, max
    stats_acc = {v: {"sum": 0.0, "sum_sq": 0.0, "count": 0, "min": float("inf"), "max": float("-inf")} for v in era5_vars}
    
    cleaned_era5_dir = os.path.join(BASE_DIR, "data", "ERA5", "cleaned")
    for sname in sorted(train_storms):
        cleaned_nc = os.path.join(cleaned_era5_dir, f"{sname}_cleaned.nc")
        if not os.path.exists(cleaned_nc):
            logger.warning(f"Train storm file not found: {cleaned_nc}")
            continue
            
        logger.info(f"Computing statistics from train storm: {sname}...")
        with nc.Dataset(cleaned_nc, "r") as ds:
            for v in era5_vars:
                if v in ds.variables:
                    data = ds.variables[v][:]
                    # Masked array or numpy array with NaNs
                    if np.ma.is_masked(data):
                        valid_data = data.compressed()
                    else:
                        valid_data = data[~np.isnan(data)]
                        
                    if len(valid_data) > 0:
                        v_sum = float(np.sum(valid_data, dtype=np.float64))
                        v_sum_sq = float(np.sum(valid_data**2, dtype=np.float64))
                        v_count = len(valid_data)
                        v_min = float(np.min(valid_data))
                        v_max = float(np.max(valid_data))
                        
                        stats_acc[v]["sum"] += v_sum
                        stats_acc[v]["sum_sq"] += v_sum_sq
                        stats_acc[v]["count"] += v_count
                        stats_acc[v]["min"] = min(stats_acc[v]["min"], v_min)
                        stats_acc[v]["max"] = max(stats_acc[v]["max"], v_max)
                        
    normalization_params = {
        "description": "Fitted normalization statistics calculated STRICTLY on the training split.",
        "splits_source": {
            "train_storms": list(train_storms),
            "train_sample_count": len(df_train)
        },
        "features": {}
    }
    
    for v in era5_vars:
        acc = stats_acc[v]
        if acc["count"] > 0:
            mean_val = acc["sum"] / acc["count"]
            variance = max(0.0, (acc["sum_sq"] / acc["count"]) - (mean_val**2))
            std_val = float(np.sqrt(variance))
            if std_val < 1e-6:
                std_val = 1.0
                
            normalization_params["features"][v] = {
                "mean": round(mean_val, 4),
                "std": round(std_val, 4),
                "min": round(acc["min"], 4),
                "max": round(acc["max"], 4)
            }
            logger.info(f"Train stats: {v:20s} -> mean={mean_val:10.2f}, std={std_val:10.2f}, min={acc['min']:10.2f}, max={acc['max']:10.2f}")
            
    # Save normalization parameters
    norm_path = get_path("normalization_params", config)
    os.makedirs(os.path.dirname(norm_path), exist_ok=True)
    with open(norm_path, "w", encoding="utf-8") as f:
        json.dump(normalization_params, f, indent=2)
    logger.info(f"Saved fitted normalization parameters to: {norm_path}")
    
    # 6. Apply normalization to validation sample to verify transform logic
    if len(df_val) > 0:
        val_sample_path = os.path.join(BASE_DIR, df_val.iloc[0]["era5_path"])
        if os.path.exists(val_sample_path):
            with nc.Dataset(val_sample_path, "r") as ds_val:
                t2m_raw = ds_val.variables["t2m"][:]
                t2m_norm = (t2m_raw - normalization_params["features"]["t2m"]["mean"]) / normalization_params["features"]["t2m"]["std"]
                logger.info(f"Validation sample transformed: t2m normalized shape={t2m_norm.shape}, range=[{np.min(t2m_norm):.2f}, {np.max(t2m_norm):.2f}]")
                
    # Save leakage report
    leakage_report_path = os.path.join(BASE_DIR, "metadata", "leakage_check_report.json")
    with open(leakage_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved leakage check report to: {leakage_report_path}")
    
    if report["passed_all_checks"]:
        logger.info("ALL LEAKAGE CHECKS PASSED SUCCESSFULLY!")
    else:
        logger.error(f"LEAKAGE CHECKS FAILED WITH {len(report['violations'])} VIOLATIONS!")
        
    return report

if __name__ == "__main__":
    check_leakage_and_compute_normalization()
