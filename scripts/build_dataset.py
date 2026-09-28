"""
Step 10: Final Dataset Builder for VayuDrishti.
Organizes train, validation, and test splits into standard ML layout:
dataset/
├── train/
│   ├── images/
│   ├── era5/
│   └── labels/
│   └── manifest.csv
├── validation/
│   ├── images/
│   ├── era5/
│   └── labels/
│   └── manifest.csv
└── test/
    ├── images/
    ├── era5/
    └── labels/
    └── manifest.csv
"""
import os
import sys
import json
import shutil
import logging
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import load_config, get_path, BASE_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("build_dataset")

def build_final_dataset(splits_dir=None, dataset_root=None, config=None):
    if config is None:
        config = load_config()
    if splits_dir is None:
        splits_dir = get_path("splits_dir", config)
    if dataset_root is None:
        dataset_root = get_path("dataset_dir", config)
        
    logger.info(f"Assembling final dataset into {dataset_root}...")
    
    splits = ["train", "validation", "test"]
    for split_name in splits:
        split_csv = os.path.join(splits_dir, f"{split_name}.csv")
        if not os.path.exists(split_csv):
            logger.error(f"Split file missing: {split_csv}")
            continue
            
        df_split = pd.read_csv(split_csv)
        logger.info(f"Processing split '{split_name}' ({len(df_split)} samples)...")
        
        split_base = os.path.join(dataset_root, split_name)
        images_dir = os.path.join(split_base, "images")
        era5_dir = os.path.join(split_base, "era5")
        labels_dir = os.path.join(split_base, "labels")
        
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(era5_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        
        manifest_rows = []
        
        for _, row in df_split.iterrows():
            sample_id = row["sample_id"]
            
            # Label JSON
            label_data = {
                "sample_id": sample_id,
                "storm_id": row["storm_id"],
                "storm_name": row["storm_name"],
                "datetime": row["datetime"],
                "latitude": float(row["latitude"]),
                "longitude": float(row["longitude"]),
                "wind_speed_kt": float(row["wind_speed"]) if pd.notna(row["wind_speed"]) else None,
                "pressure_hpa": float(row["pressure"]) if pd.notna(row["pressure"]) else None,
                "intensity_class": row["intensity_class"],
                "basin": row["basin"]
            }
            
            # Save label file
            label_filename = f"{sample_id}_label.json"
            label_path = os.path.join(labels_dir, label_filename)
            with open(label_path, "w", encoding="utf-8") as f:
                json.dump(label_data, f, indent=2)
                
            rel_label = os.path.relpath(label_path, BASE_DIR).replace("\\", "/")
            
            # ERA5 patch handling
            # To avoid duplicating large binary files, we link or copy if small
            era5_src = os.path.join(BASE_DIR, row["era5_path"]) if pd.notna(row["era5_path"]) and row["era5_path"] != "" else None
            era5_dst_rel = ""
            if era5_src and os.path.exists(era5_src):
                # Copy or reference the cropped file
                era5_target = os.path.join(era5_dir, f"{sample_id}_era5.nc")
                if not os.path.exists(era5_target):
                    shutil.copy2(era5_src, era5_target)
                era5_dst_rel = os.path.relpath(era5_target, BASE_DIR).replace("\\", "/")
                
            # Satellite image handling
            sat_src = os.path.join(BASE_DIR, row["satellite_path"]) if pd.notna(row["satellite_path"]) and row["satellite_path"] != "" else None
            sat_dst_rel = ""
            if sat_src and os.path.exists(sat_src):
                sat_ext = os.path.splitext(sat_src)[1]
                sat_target = os.path.join(images_dir, f"{sample_id}_sat{sat_ext}")
                if not os.path.exists(sat_target):
                    shutil.copy2(sat_src, sat_target)
                sat_dst_rel = os.path.relpath(sat_target, BASE_DIR).replace("\\", "/")
                
            manifest_rows.append({
                "sample_id": sample_id,
                "storm_name": row["storm_name"],
                "datetime": row["datetime"],
                "latitude": row["latitude"],
                "longitude": row["longitude"],
                "wind_speed": row["wind_speed"],
                "pressure": row["pressure"],
                "intensity_class": row["intensity_class"],
                "label_path": rel_label,
                "era5_path": era5_dst_rel,
                "satellite_path": sat_dst_rel,
                "quality_status": row["quality_status"]
            })
            
        manifest_df = pd.DataFrame(manifest_rows)
        manifest_path = os.path.join(split_base, "manifest.csv")
        manifest_df.to_csv(manifest_path, index=False)
        logger.info(f"Saved split manifest: {manifest_path} ({len(manifest_df)} records)")
        
    logger.info("Dataset layout completed successfully.")

if __name__ == "__main__":
    build_final_dataset()
