"""
Step 3: ERA5 Cleaning and Validation Module for VayuDrishti
Reads, validates, extracts derived variables, and catalogs ERA5 NetCDF environmental data.
"""
import os
import sys
import glob
import zipfile
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import load_config, get_path, BASE_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("clean_era5")

def extract_era5_zips(raw_dir, extracted_dir):
    """
    Extracts raw ERA5 zip archives into data/ERA5/extracted/<storm_name>/
    """
    os.makedirs(extracted_dir, exist_ok=True)
    zip_files = sorted(glob.glob(os.path.join(raw_dir, "*.zip")))
    logger.info(f"Found {len(zip_files)} ERA5 zip archives in {raw_dir}")
    
    extracted_files = {}
    for zpath in zip_files:
        storm_tag = os.path.splitext(os.path.basename(zpath))[0]
        # Clean storm tag name, e.g. Asna-1 -> Asna_1
        clean_tag = storm_tag.replace("-", "_")
        target_nc = os.path.join(extracted_dir, f"{clean_tag}.nc")
        
        if not os.path.exists(target_nc):
            logger.info(f"Extracting {os.path.basename(zpath)} -> {target_nc}...")
            with zipfile.ZipFile(zpath, "r") as z:
                # Find the internal nc file
                nc_names = [n for n in z.namelist() if n.endswith(".nc")]
                if not nc_names:
                    logger.warning(f"No .nc file found in {zpath}!")
                    continue
                internal_name = nc_names[0]
                with z.open(internal_name) as src, open(target_nc, "wb") as dst:
                    dst.write(src.read())
        else:
            logger.info(f"Extracted file already exists: {target_nc}")
            
        extracted_files[clean_tag] = target_nc
        
    return extracted_files

def merge_multi_part_storms(extracted_files, extracted_dir):
    """
    Merges storms that were downloaded in multiple parts (e.g. Asna_1 + Asna2, Ditwah1 + Ditwah2).
    """
    # Group by base storm name
    merged_catalog = {}
    storm_groups = {
        "ASNA": ["Asna_1", "Asna2"],
        "DITWAH": ["Ditwah1", "Ditwah2"]
    }
    
    # Process merged storms
    for storm_name, parts in storm_groups.items():
        part_paths = [extracted_files[p] for p in parts if p in extracted_files]
        if len(part_paths) == len(parts):
            merged_nc = os.path.join(extracted_dir, f"{storm_name}.nc")
            if not os.path.exists(merged_nc):
                logger.info(f"Merging multi-part storm {storm_name} ({parts}) -> {merged_nc}...")
                datasets = [xr.open_dataset(p) for p in part_paths]
                time_coord = "valid_time" if "valid_time" in datasets[0].coords else "time"
                ds_merged = xr.concat(datasets, dim=time_coord)
                # Sort by time
                ds_merged = ds_merged.sortby(time_coord)
                # Write to disk
                ds_merged.to_netcdf(merged_nc)
                for d in datasets:
                    d.close()
                ds_merged.close()
            merged_catalog[storm_name] = merged_nc
            
    # Add standalone storms
    for tag, path in extracted_files.items():
        base_name = tag.split("_")[0].upper()
        if base_name not in storm_groups and tag not in ["Asna_1", "Asna2", "Ditwah1", "Ditwah2"]:
            merged_catalog[base_name] = path
            
    return merged_catalog

def clean_and_catalog_era5(raw_dir=None, extracted_dir=None, catalog_out=None, config=None):
    if config is None:
        config = load_config()
    if raw_dir is None:
        raw_dir = get_path("era5_raw_dir", config)
    if extracted_dir is None:
        extracted_dir = get_path("era5_extracted_dir", config)
    if catalog_out is None:
        catalog_out = get_path("era5_catalog", config)
        
    # 1. Extract archives
    extracted_files = extract_era5_zips(raw_dir, extracted_dir)
    
    # 2. Merge multi-part storms
    storm_files = merge_multi_part_storms(extracted_files, extracted_dir)
    
    # 3. Clean and validate each storm dataset
    catalog_rows = []
    cleaned_era5_dir = os.path.join(BASE_DIR, "data", "ERA5", "cleaned")
    os.makedirs(cleaned_era5_dir, exist_ok=True)
    
    for storm_name, nc_path in sorted(storm_files.items()):
        logger.info(f"Validating ERA5 data for storm {storm_name}: {nc_path}")
        ds = xr.open_dataset(nc_path)
        
        # Standardize time dimension
        time_col = "valid_time" if "valid_time" in ds.coords else "time"
        times = pd.to_datetime(ds[time_col].values, utc=True)
        
        # Check time sorting and duplicates
        is_sorted = np.all(times[:-1] <= times[1:])
        time_series = pd.Series(times)
        dup_count = int(time_series.duplicated().sum())
        
        # Check missing timesteps (expected 1h)
        full_hourly_range = pd.date_range(start=times[0], end=times[-1], freq="1h", tz="UTC")
        expected_timesteps = len(full_hourly_range)
        actual_timesteps = len(times)
        missing_timesteps = expected_timesteps - actual_timesteps
        
        # Check coordinates
        lat_vals = ds["latitude"].values
        lon_vals = ds["longitude"].values
        lat_min, lat_max = float(np.min(lat_vals)), float(np.max(lat_vals))
        lon_min, lon_max = float(np.min(lon_vals)), float(np.max(lon_vals))
        
        # Inspect variables and units
        var_info = {}
        vars_available = list(ds.data_vars.keys())
        for v in vars_available:
            units = getattr(ds[v], "units", "unknown")
            long_name = getattr(ds[v], "long_name", v)
            nan_pct = float((np.isnan(ds[v].values).sum() / ds[v].values.size) * 100.0)
            var_info[v] = {
                "units": units,
                "long_name": long_name,
                "nan_percentage": round(nan_pct, 2)
            }
            
        # Check quality
        quality_status = "VALID"
        if dup_count > 0:
            quality_status = "WARNING_DUPLICATES"
        if missing_timesteps > 0:
            quality_status = "WARNING_GAPS"
        if not all(k in vars_available for k in ["u10", "v10", "t2m", "msl"]):
            quality_status = "INVALID_MISSING_VARS"
            
        # Calculate derived variables and save cleaned version
        # Derived: wind_speed_10m = sqrt(u10^2 + v10^2)
        # Derived: wind_direction_10m = atan2(v10, u10)
        # Note: units preserved and conversions recorded in attributes
        cleaned_nc_path = os.path.join(cleaned_era5_dir, f"{storm_name}_cleaned.nc")
        if not os.path.exists(cleaned_nc_path):
            logger.info(f"Computing derived variables for {storm_name}...")
            u10 = ds["u10"]
            v10 = ds["v10"]
            wind_speed = np.sqrt(u10**2 + v10**2)
            wind_speed.attrs["units"] = "m s**-1"
            wind_speed.attrs["long_name"] = "10 metre wind speed derived from sqrt(u10^2 + v10^2)"
            
            wind_dir_rad = np.arctan2(v10, u10)
            wind_dir_rad.attrs["units"] = "radians"
            wind_dir_rad.attrs["long_name"] = "10 metre wind direction derived from atan2(v10, u10)"
            
            # Meteorological direction in degrees (direction FROM which wind blows)
            wind_dir_deg = (270.0 - np.rad2deg(wind_dir_rad)) % 360.0
            wind_dir_deg.attrs["units"] = "degrees"
            wind_dir_deg.attrs["long_name"] = "10 metre meteorological wind direction (0=N, 90=E, 180=S, 270=W)"
            
            # Mean sea level pressure in hPa (conversion from Pa recorded)
            msl_hpa = ds["msl"] / 100.0
            msl_hpa.attrs["units"] = "hPa"
            msl_hpa.attrs["long_name"] = "Mean sea level pressure converted from Pa (msl / 100)"
            msl_hpa.attrs["original_units"] = "Pa"
            
            ds_clean = ds.copy()
            ds_clean["wind_speed_10m"] = wind_speed
            ds_clean["wind_direction_10m"] = wind_dir_rad
            ds_clean["wind_direction_10m_deg"] = wind_dir_deg
            ds_clean["msl_hpa"] = msl_hpa
            
            # Rename time coordinate to 'time'
            if time_col != "time":
                ds_clean = ds_clean.rename({time_col: "time"})
                
            # Drop unnecessary coordinates like expver or number if present
            for extra_coord in ["expver", "number"]:
                if extra_coord in ds_clean.coords:
                    ds_clean = ds_clean.drop_vars(extra_coord)
                    
            ds_clean.to_netcdf(cleaned_nc_path)
            ds_clean.close()
            
        ds.close()
        
        rel_cleaned_path = os.path.relpath(cleaned_nc_path, BASE_DIR).replace("\\", "/")
        rel_raw_path = os.path.relpath(nc_path, BASE_DIR).replace("\\", "/")
        
        catalog_rows.append({
            "storm_name": storm_name,
            "raw_file": rel_raw_path,
            "cleaned_file": rel_cleaned_path,
            "start_time": times[0].isoformat(),
            "end_time": times[-1].isoformat(),
            "timesteps": actual_timesteps,
            "expected_timesteps": expected_timesteps,
            "missing_timesteps": missing_timesteps,
            "duplicate_timestamps": dup_count,
            "latitude_min": lat_min,
            "latitude_max": lat_max,
            "longitude_min": lon_min,
            "longitude_max": lon_max,
            "variables": ",".join(vars_available),
            "derived_variables": "wind_speed_10m,wind_direction_10m,wind_direction_10m_deg,msl_hpa",
            "quality_status": quality_status
        })
        
    df_catalog = pd.DataFrame(catalog_rows)
    os.makedirs(os.path.dirname(catalog_out), exist_ok=True)
    df_catalog.to_csv(catalog_out, index=False)
    logger.info(f"Saved ERA5 catalog to: {catalog_out} ({len(df_catalog)} storms)")
    
    return df_catalog

if __name__ == "__main__":
    clean_and_catalog_era5()
