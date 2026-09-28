"""
Steps 4, 5, 6, 7, 8: Synchronization, Spatial Matching, Quality Control, Labeling, and Master Dataset Builder.
Produces:
  - processed/era5_crops/<sample_id>_era5.nc
  - processed/satellite_crops/<sample_id>_sat.png (when satellite imagery is present)
  - metadata/matched_samples.csv
  - metadata/data_quality_report.csv
"""
import os
import sys
import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import xarray as xr
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import load_config, get_path, BASE_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("sync_and_match")

def classify_intensity(wind_kt, policy_thresholds):
    """
    Classifies cyclone intensity based on maximum sustained wind speed (kt)
    using the defined project labeling policy.
    """
    if pd.isna(wind_kt) or wind_kt is None:
        return "UNKNOWN"
        
    for item in policy_thresholds:
        if item["min_wind_kt"] <= wind_kt <= item["max_wind_kt"]:
            return item["class"]
    return "LOW"

def crop_era5_environment(ds, target_time, center_lat, center_lon, crop_deg=10.0):
    """
    Extracts a spatial region of size crop_deg x crop_deg centered on (center_lat, center_lon)
    at the closest timestamp to target_time.
    """
    # 1. Select nearest time slice
    # ds['time'] is sorted datetime64[ns]
    time_diff = np.abs(pd.to_datetime(ds["time"].values, utc=True) - target_time)
    nearest_idx = int(np.argmin(time_diff))
    diff_minutes = float(time_diff[nearest_idx].total_seconds() / 60.0)
    matched_time = pd.to_datetime(ds["time"].values[nearest_idx], utc=True)
    
    # 2. Compute spatial bounding box
    half_box = crop_deg / 2.0
    lat_min_req = center_lat - half_box
    lat_max_req = center_lat + half_box
    lon_min_req = center_lon - half_box
    lon_max_req = center_lon + half_box
    
    # ERA5 latitudes are 30.0 down to 0.0 (descending), longitudes 40.0 to 100.0 (ascending)
    # Check if center is within ERA5 domain
    domain_lat_min = float(ds["latitude"].min())
    domain_lat_max = float(ds["latitude"].max())
    domain_lon_min = float(ds["longitude"].min())
    domain_lon_max = float(ds["longitude"].max())
    
    is_center_inside = (domain_lat_min <= center_lat <= domain_lat_max and
                        domain_lon_min <= center_lon <= domain_lon_max)
                        
    # Clamp bounding box to domain bounds
    slice_lat_max = min(domain_lat_max, lat_max_req)
    slice_lat_min = max(domain_lat_min, lat_min_req)
    slice_lon_min = max(domain_lon_min, lon_min_req)
    slice_lon_max = min(domain_lon_max, lon_max_req)
    
    # In xarray with descending latitude: slice(slice_lat_max, slice_lat_min)
    patch = ds.isel(time=nearest_idx).sel(
        latitude=slice(slice_lat_max, slice_lat_min),
        longitude=slice(slice_lon_min, slice_lon_max)
    )
    
    bounds = {
        "crop_lat_min": float(slice_lat_min),
        "crop_lat_max": float(slice_lat_max),
        "crop_lon_min": float(slice_lon_min),
        "crop_lon_max": float(slice_lon_max),
        "requested_lat_min": float(lat_min_req),
        "requested_lat_max": float(lat_max_req),
        "requested_lon_min": float(lon_min_req),
        "requested_lon_max": float(lon_max_req),
        "is_center_inside": bool(is_center_inside),
        "matched_time": matched_time.isoformat(),
        "diff_minutes": diff_minutes
    }
    
    return patch, bounds

def synchronize_and_match(config=None):
    if config is None:
        config = load_config()
        
    logger.info("Initializing multi-source synchronization and spatial matching...")
    
    # Paths
    ibtracs_clean_path = get_path("ibtracs_cleaned", config)
    era5_catalog_path = get_path("era5_catalog", config)
    image_catalog_path = get_path("image_catalog", config)
    matched_samples_path = get_path("matched_samples", config)
    quality_report_path = get_path("data_quality_report", config)
    
    era5_crops_dir = os.path.join(BASE_DIR, "processed", "era5_crops")
    satellite_crops_dir = os.path.join(BASE_DIR, "processed", "satellite_crops")
    os.makedirs(era5_crops_dir, exist_ok=True)
    os.makedirs(satellite_crops_dir, exist_ok=True)
    
    # Load catalogs
    df_ibtracs = pd.read_csv(ibtracs_clean_path)
    df_ibtracs["datetime"] = pd.to_datetime(df_ibtracs["datetime"], utc=True)
    
    df_era5_cat = pd.read_csv(era5_catalog_path)
    df_sat_cat = pd.read_csv(image_catalog_path) if os.path.exists(image_catalog_path) else pd.DataFrame()
    if not df_sat_cat.empty and "datetime" in df_sat_cat.columns:
        df_sat_cat["datetime"] = pd.to_datetime(df_sat_cat["datetime"], utc=True)
        
    policy = config.get("intensity_policy", {}).get("thresholds", [])
    max_tol_minutes = config.get("temporal", {}).get("max_temporal_tolerance_minutes", 60)
    crop_size_deg = config.get("spatial", {}).get("crop_size_deg", 10.0)
    
    # Open all cleaned ERA5 datasets
    era5_datasets = {}
    for _, row in df_era5_cat.iterrows():
        sname = row["storm_name"]
        nc_file = os.path.join(BASE_DIR, row["cleaned_file"])
        if os.path.exists(nc_file):
            era5_datasets[sname] = xr.open_dataset(nc_file)
            
    logger.info(f"Loaded {len(era5_datasets)} cleaned ERA5 storm datasets.")
    
    # Filter IBTrACS for the storms available in ERA5
    era5_storm_names = list(era5_datasets.keys())
    matched_samples = []
    quality_records = []
    
    sample_id_counts = {}
    
    for sname, ds_era5 in era5_datasets.items():
        # Match storm in IBTrACS by name
        storm_ib = df_ibtracs[df_ibtracs["NAME"].str.upper() == sname].copy()
        if storm_ib.empty:
            # Fallback search for partial name match
            storm_ib = df_ibtracs[df_ibtracs["NAME"].str.upper().str.contains(sname, na=False)].copy()
            
        logger.info(f"Storm {sname}: found {len(storm_ib)} IBTrACS track points.")
        
        era5_start = pd.to_datetime(ds_era5["time"].values[0], utc=True)
        era5_end = pd.to_datetime(ds_era5["time"].values[-1], utc=True)
        
        for _, track_row in storm_ib.iterrows():
            sid = track_row["SID"]
            obs_time = track_row["datetime"]
            lat = float(track_row["latitude"])
            lon = float(track_row["longitude"])
            wind_kt = track_row["wind_speed"] if pd.notna(track_row["wind_speed"]) else None
            pres_hpa = track_row["pressure"] if pd.notna(track_row["pressure"]) else None
            basin = track_row["BASIN"]
            
            # Formulate canonical sample_id: <STORM>_<YYYYMMDD>_<HHMM>
            time_tag = obs_time.strftime("%Y%m%d_%H%M")
            sample_id = f"{sname}_{time_tag}"
            
            # Track duplicates
            sample_id_counts[sample_id] = sample_id_counts.get(sample_id, 0) + 1
            if sample_id_counts[sample_id] > 1:
                sample_id = f"{sample_id}_dup{sample_id_counts[sample_id]}"
                
            intensity_cls = classify_intensity(wind_kt, policy)
            
            # Quality checks initialized
            q_status = "VALID"
            q_reasons = []
            
            # Check 1: Temporal coverage in ERA5
            if obs_time < (era5_start - pd.Timedelta(minutes=max_tol_minutes)) or \
               obs_time > (era5_end + pd.Timedelta(minutes=max_tol_minutes)):
                q_status = "INVALID"
                q_reasons.append("ERA5_TEMPORAL_OUT_OF_BOUNDS")
                era5_crop_path = None
                era5_diff_min = np.nan
            else:
                # Perform spatial extraction
                patch, bounds = crop_era5_environment(
                    ds_era5, obs_time, lat, lon, crop_deg=crop_size_deg
                )
                era5_diff_min = bounds["diff_minutes"]
                
                # Check 2: Temporal tolerance exceeded
                if era5_diff_min > max_tol_minutes:
                    q_status = "INVALID"
                    q_reasons.append(f"ERA5_TIME_DIFF_{era5_diff_min:.1f}m_EXCEEDS_{max_tol_minutes}m")
                    
                # Check 3: Spatial coverage
                if not bounds["is_center_inside"]:
                    q_status = "INVALID"
                    q_reasons.append(f"CYCLONE_CENTER_OUTSIDE_ERA5_DOMAIN_(lat={lat},lon={lon})")
                    
                # Check 4: Missing ERA5 variables or NaNs in core dynamics
                # u10, v10, t2m, msl should have zero NaNs
                for v in ["u10", "v10", "t2m", "msl"]:
                    if v in patch and np.isnan(patch[v].values).any():
                        q_status = "INVALID"
                        q_reasons.append(f"ERA5_VAR_{v}_CONTAINS_NANS")
                        
                # Save cropped patch to processed/era5_crops/
                era5_crop_filename = f"{sample_id}_era5.nc"
                era5_crop_full = os.path.join(era5_crops_dir, era5_crop_filename)
                if q_status != "INVALID" and not os.path.exists(era5_crop_full):
                    patch.to_netcdf(era5_crop_full)
                    
                era5_crop_path = os.path.relpath(era5_crop_full, BASE_DIR).replace("\\", "/")
                
            # Check 5: Satellite Match
            sat_crop_path = None
            sat_diff_min = np.nan
            if not df_sat_cat.empty:
                # Search for closest satellite acquisition within tolerance
                sat_valid = df_sat_cat[df_sat_cat["quality_status"].isin(["VALID", "WARNING_HIGH_MISSING"])]
                if not sat_valid.empty:
                    sat_times = sat_valid["datetime"]
                    sat_diffs = np.abs(sat_times - obs_time)
                    closest_idx = sat_diffs.idxmin()
                    closest_diff_min = sat_diffs[closest_idx].total_seconds() / 60.0
                    if closest_diff_min <= max_tol_minutes:
                        sat_diff_min = closest_diff_min
                        sat_row = sat_valid.loc[closest_idx]
                        sat_crop_path = sat_row["source_file"]
                    else:
                        if q_status == "VALID":
                            q_status = "WARNING"
                        q_reasons.append(f"SATELLITE_TIME_DIFF_{closest_diff_min:.1f}m_EXCEEDS_{max_tol_minutes}m")
                else:
                    if q_status == "VALID":
                        q_status = "WARNING"
                    q_reasons.append("NO_VALID_SATELLITE_IMAGES_IN_CATALOG")
            else:
                if q_status == "VALID":
                    q_status = "WARNING"
                q_reasons.append("SATELLITE_DATA_ABSENT_IN_SOURCE")
                
            # Check 6: Intensity / Label validity
            if pd.isna(wind_kt):
                if q_status == "VALID":
                    q_status = "WARNING"
                q_reasons.append("MISSING_WIND_LABEL")
            if pd.isna(pres_hpa):
                if q_status == "VALID":
                    q_status = "WARNING"
                q_reasons.append("MISSING_PRESSURE_LABEL")
                
            reason_str = "; ".join(q_reasons) if q_reasons else "ALL_CHECKS_PASSED"
            
            # Append to master dataset
            matched_samples.append({
                "sample_id": sample_id,
                "storm_id": sid,
                "storm_name": sname,
                "datetime": obs_time.isoformat(),
                "latitude": lat,
                "longitude": lon,
                "satellite_path": sat_crop_path if sat_crop_path else "",
                "era5_path": era5_crop_path if era5_crop_path else "",
                "wind_speed": wind_kt if pd.notna(wind_kt) else np.nan,
                "pressure": pres_hpa if pd.notna(pres_hpa) else np.nan,
                "intensity_class": intensity_cls,
                "basin": basin,
                "quality_status": q_status
            })
            
            # Quality report record
            quality_records.append({
                "sample_id": sample_id,
                "storm_id": sid,
                "storm_name": sname,
                "datetime": obs_time.isoformat(),
                "latitude": lat,
                "longitude": lon,
                "wind_speed": wind_kt,
                "pressure": pres_hpa,
                "era5_time_diff_min": round(era5_diff_min, 1) if pd.notna(era5_diff_min) else np.nan,
                "satellite_time_diff_min": round(sat_diff_min, 1) if pd.notna(sat_diff_min) else np.nan,
                "quality_status": q_status,
                "quality_reasons": reason_str
            })
            
    # Close ERA5 datasets
    for ds in era5_datasets.values():
        ds.close()
        
    df_matched = pd.DataFrame(matched_samples)
    df_quality = pd.DataFrame(quality_records)
    
    # Save master dataset
    os.makedirs(os.path.dirname(matched_samples_path), exist_ok=True)
    df_matched.to_csv(matched_samples_path, index=False)
    logger.info(f"Saved master matched dataset to: {matched_samples_path} ({len(df_matched)} samples)")
    
    # Save quality control report
    os.makedirs(os.path.dirname(quality_report_path), exist_ok=True)
    df_quality.to_csv(quality_report_path, index=False)
    logger.info(f"Saved quality control report to: {quality_report_path}")
    
    # Summary of quality status
    status_counts = df_quality["quality_status"].value_counts().to_dict()
    logger.info(f"Quality Status Distribution: {status_counts}")
    
    return df_matched, df_quality

if __name__ == "__main__":
    synchronize_and_match()
