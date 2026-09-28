"""
Feature Caching Script for CPU-First Optimization.
Pre-extracts, resizes, and normalizes all ERA5 and satellite tensors ONCE to disk.
Avoids decoding NetCDF files and re-synthesizing images on every training epoch.
Strictly applies training set normalization parameters only.
"""
import os
import sys
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import xarray as xr
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import BASE_DIR, load_config
from dataset import ERA5_VARS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("cache_features")

def build_feature_cache(resolutions=(160, 192, 224)):
    config = load_config()
    norm_path = Path(BASE_DIR) / config["paths"]["normalization_params"]
    
    with open(norm_path, "r") as f:
        norm_dict = json.load(f)
    norm_era5 = norm_dict.get("era5_features", {})
    
    # Target directories
    era5_feat_dir = Path(BASE_DIR) / "processed" / "era5_features"
    sat_feat_dir = Path(BASE_DIR) / "processed" / "satellite_features"
    meta_dir = Path(BASE_DIR) / "processed" / "metadata"
    
    era5_feat_dir.mkdir(parents=True, exist_ok=True)
    sat_feat_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all samples from master matched_samples
    matched_path = Path(BASE_DIR) / config["paths"]["matched_samples"]
    df = pd.read_csv(matched_path)
    logger.info(f"Loaded {len(df)} total samples from matched registry.")
    
    cached_records = []
    
    for idx, row in df.iterrows():
        sample_id = row["sample_id"]
        era5_rel = row.get("era5_path", "")
        sat_rel = row.get("satellite_path", "")
        
        # 1. Process & Cache ERA5 patch (8, 40, 40)
        era5_cache_file = era5_feat_dir / f"{sample_id}.npy"
        era5_normed_patch = None
        
        if era5_cache_file.exists():
            era5_normed_patch = np.load(era5_cache_file)
        else:
            channels = []
            if era5_rel and pd.notna(era5_rel) and era5_rel != "":
                full_era5_p = Path(BASE_DIR) / era5_rel
                if full_era5_p.exists():
                    try:
                        with xr.open_dataset(full_era5_p) as ds:
                            for var in ERA5_VARS:
                                if var in ds:
                                    v_arr = ds[var].values.astype(np.float32)
                                    if v_arr.ndim == 3:
                                        v_arr = v_arr[0]
                                    mean = norm_era5.get(var, {}).get("mean", 0.0)
                                    std = norm_era5.get(var, {}).get("std", 1.0)
                                    if std < 1e-6:
                                        std = 1.0
                                    normed = (v_arr - mean) / std
                                    channels.append(normed)
                                else:
                                    channels.append(np.zeros((40, 40), dtype=np.float32))
                            era5_normed_patch = np.stack(channels, axis=0).astype(np.float32)
                    except Exception as e:
                        logger.warning(f"Error loading ERA5 for {sample_id}: {e}")
            
            if era5_normed_patch is None:
                era5_normed_patch = np.zeros((len(ERA5_VARS), 40, 40), dtype=np.float32)
                
            np.save(era5_cache_file, era5_normed_patch)
            
        # 2. Process & Cache Satellite Tensors at specified resolutions
        for res in resolutions:
            sat_cache_file = sat_feat_dir / f"{sample_id}_{res}.npy"
            if not sat_cache_file.exists():
                sat_tensor = None
                has_real_sat = False
                
                if sat_rel and pd.notna(sat_rel) and sat_rel != "":
                    full_sat_p = Path(BASE_DIR) / sat_rel
                    if full_sat_p.exists():
                        try:
                            with Image.open(full_sat_p) as img:
                                img = img.convert("RGB").resize((res, res), Image.Resampling.BILINEAR)
                                arr = np.array(img, dtype=np.float32) / 255.0
                                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                                std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                                normed = (arr - mean) / std
                                sat_tensor = np.transpose(normed, (2, 0, 1)).astype(np.float32)
                                has_real_sat = True
                        except Exception:
                            pass
                            
                # Fallback multi-channel environmental proxy
                if sat_tensor is None and era5_normed_patch is not None:
                    ch0 = Image.fromarray(era5_normed_patch[5]).resize((res, res), Image.Resampling.BILINEAR)
                    ch1 = Image.fromarray(era5_normed_patch[2]).resize((res, res), Image.Resampling.BILINEAR)
                    ch2 = Image.fromarray(era5_normed_patch[7]).resize((res, res), Image.Resampling.BILINEAR)
                    sat_tensor = np.stack([np.array(ch0), np.array(ch1), np.array(ch2)], axis=0).astype(np.float32)
                    
                if sat_tensor is None:
                    sat_tensor = np.zeros((3, res, res), dtype=np.float32)
                    
                np.save(sat_cache_file, sat_tensor)
                
        cached_records.append({
            "sample_id": sample_id,
            "era5_cache": str(era5_cache_file.relative_to(BASE_DIR)).replace("\\", "/"),
            "sat_cache_160": str((sat_feat_dir / f"{sample_id}_160.npy").relative_to(BASE_DIR)).replace("\\", "/"),
            "sat_cache_192": str((sat_feat_dir / f"{sample_id}_192.npy").relative_to(BASE_DIR)).replace("\\", "/"),
            "sat_cache_224": str((sat_feat_dir / f"{sample_id}_224.npy").relative_to(BASE_DIR)).replace("\\", "/")
        })
        
    manifest_cache = meta_dir / "cache_manifest.json"
    with open(manifest_cache, "w") as f:
        json.dump(cached_records, f, indent=2)
        
    logger.info(f"Successfully cached preprocessed features for all {len(cached_records)} samples in {era5_feat_dir} and {sat_feat_dir}.")
    return len(cached_records)

if __name__ == "__main__":
    build_feature_cache()
