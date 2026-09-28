"""
Step 9: Train / Validation / Test Split Module for VayuDrishti.
Performs strictly storm-level (cyclone-level) partitioning to prevent
temporal, spatial, and storm correlation data leakage.
Outputs:
  - splits/train.csv
  - splits/validation.csv
  - splits/test.csv
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
logger = logging.getLogger("split_data")

def perform_storm_level_split(matched_csv_path=None, splits_dir=None, config=None):
    if config is None:
        config = load_config()
    if matched_csv_path is None:
        matched_csv_path = get_path("matched_samples", config)
    if splits_dir is None:
        splits_dir = get_path("splits_dir", config)
        
    os.makedirs(splits_dir, exist_ok=True)
    
    logger.info(f"Loading matched samples from {matched_csv_path}")
    df_matched = pd.read_csv(matched_csv_path)
    
    # Exclude INVALID samples from training splits
    df_usable = df_matched[df_matched["quality_status"] != "INVALID"].copy()
    total_usable = len(df_usable)
    logger.info(f"Usable samples: {total_usable} (out of {len(df_matched)} total)")
    
    # Get distinct storms and sample counts
    storm_counts = df_usable.groupby("storm_name")["sample_id"].count().to_dict()
    storms = sorted(list(storm_counts.keys()))
    num_storms = len(storms)
    logger.info(f"Unique storms available: {num_storms} -> {storms}")
    
    # Optimal storm-level allocation closest to 70% / 15% / 15%
    # Deterministic search minimizing squared error of percentages
    target_tr = config.get("splits", {}).get("train_ratio", 0.70)
    target_va = config.get("splits", {}).get("validation_ratio", 0.15)
    target_te = config.get("splits", {}).get("test_ratio", 0.15)
    
    best_loss = float("inf")
    best_partition = None
    
    # Check all valid storm allocations
    for val_s in storms:
        for test_s in storms:
            if val_s == test_s:
                continue
            tr_storms = [s for s in storms if s not in (val_s, test_s)]
            tr_cnt = sum(storm_counts[s] for s in tr_storms)
            va_cnt = storm_counts[val_s]
            te_cnt = storm_counts[test_s]
            
            p_tr = tr_cnt / total_usable
            p_va = va_cnt / total_usable
            p_te = te_cnt / total_usable
            
            loss = (p_tr - target_tr)**2 + (p_va - target_va)**2 + (p_te - target_te)**2
            if loss < best_loss:
                best_loss = loss
                best_partition = {
                    "train_storms": sorted(tr_storms),
                    "val_storms": [val_s],
                    "test_storms": [test_s],
                    "train_samples": tr_cnt,
                    "val_samples": va_cnt,
                    "test_samples": te_cnt,
                    "train_pct": round(p_tr * 100.0, 2),
                    "val_pct": round(p_va * 100.0, 2),
                    "test_pct": round(p_te * 100.0, 2)
                }
                
    logger.info("=== Selected Storm-Level Partition ===")
    logger.info(f"Train Storms: {best_partition['train_storms']} | {best_partition['train_samples']} samples ({best_partition['train_pct']}%)")
    logger.info(f"Val Storms:   {best_partition['val_storms']} | {best_partition['val_samples']} samples ({best_partition['val_pct']}%)")
    logger.info(f"Test Storms:  {best_partition['test_storms']} | {best_partition['test_samples']} samples ({best_partition['test_pct']}%)")
    
    # Partition dataset
    train_df = df_usable[df_usable["storm_name"].isin(best_partition["train_storms"])].copy()
    val_df = df_usable[df_usable["storm_name"].isin(best_partition["val_storms"])].copy()
    test_df = df_usable[df_usable["storm_name"].isin(best_partition["test_storms"])].copy()
    
    # Save split files
    train_path = os.path.join(splits_dir, "train.csv")
    val_path = os.path.join(splits_dir, "validation.csv")
    test_path = os.path.join(splits_dir, "test.csv")
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    logger.info(f"Saved: {train_path} ({len(train_df)} rows)")
    logger.info(f"Saved: {val_path} ({len(val_df)} rows)")
    logger.info(f"Saved: {test_path} ({len(test_df)} rows)")
    
    # Document split details
    summary_path = os.path.join(BASE_DIR, "metadata", "split_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(best_partition, f, indent=2)
        
    return train_df, val_df, test_df, best_partition

if __name__ == "__main__":
    perform_storm_level_split()
