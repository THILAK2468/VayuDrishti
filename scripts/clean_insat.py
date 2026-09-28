"""
Step 2: INSAT-3D Satellite Cleaning and Cataloging Module for VayuDrishti
Supports HDF5, NetCDF, GeoTIFF, and Image formats (JPEG, PNG).
Extracts timestamps (UTC), channels, geospatial bounds, checks dimensions,
identifies missing/corrupt pixels, and generates the image_catalog.csv table.
"""
import os
import sys
import re
import glob
import logging
import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import load_config, get_path, BASE_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("clean_insat")

# Nominal bounds for INSAT-3D North Indian Ocean coverage
NOMINAL_LAT_MIN = -10.0
NOMINAL_LAT_MAX = 45.0
NOMINAL_LON_MIN = 40.0
NOMINAL_LON_MAX = 105.0

# Month mapping for standard INSAT naming (e.g. 02JUN2020)
MONTH_MAP = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12
}

def parse_filename_metadata(filename):
    """
    Parses timestamp and channel from INSAT-3D filename.
    Examples:
      3DIMG_02JUN2020_0400_L1B_STD_IR2.jpg
      3DIMG_17MAY2021_0600_L1B_STD_IR1.h5
      3DIMG_01DEC2017_0300_L1B_STD_IR2 cropped.jpg
    """
    fn = os.path.basename(filename)
    stem, ext = os.path.splitext(fn)
    
    # 1. Regex for standard INSAT timestamp: DDMMMYYYY_HHMM
    # e.g. 02JUN2020_0400
    m_std = re.search(r"(\d{2})([A-Za-z]{3})(\d{4})_(\d{2})(\d{2})", fn)
    dt = None
    if m_std:
        day, mon_str, year, hr, minute = m_std.groups()
        mon = MONTH_MAP.get(mon_str.upper())
        if mon:
            try:
                dt = datetime.datetime(int(year), mon, int(day), int(hr), int(minute), tzinfo=datetime.timezone.utc)
            except ValueError:
                dt = None
                
    # Fallback regex for numeric timestamp: YYYYMMDD_HHMM
    if dt is None:
        m_num = re.search(r"(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})", fn)
        if m_num:
            yr, mo, da, hr, mi = m_num.groups()
            try:
                dt = datetime.datetime(int(yr), int(mo), int(da), int(hr), int(mi), tzinfo=datetime.timezone.utc)
            except ValueError:
                dt = None
                
    # 2. Extract Channel / Spectral band
    channel = "IR" # default
    fn_upper = fn.upper()
    for ch in ["TIR1", "TIR2", "SWIR", "MIR", "IR1", "IR2", "VIS", "WV"]:
        if ch in fn_upper:
            channel = ch
            break
            
    return dt, channel

def inspect_hdf5(filepath):
    """Inspects INSAT-3D HDF5 file."""
    import h5py
    with h5py.File(filepath, "r") as f:
        # Check datasets
        dsets = list(f.keys())
        main_dset = None
        for k in dsets:
            if isinstance(f[k], h5py.Dataset):
                main_dset = f[k]
                break
            elif isinstance(f[k], h5py.Group):
                for sub_k in f[k].keys():
                    if isinstance(f[k][sub_k], h5py.Dataset):
                        main_dset = f[k][sub_k]
                        break
                if main_dset is not None:
                    break
                    
        if main_dset is None:
            raise ValueError("No dataset found in HDF5")
            
        data = main_dset[:]
        h, w = data.shape[:2]
        min_v = float(np.nanmin(data))
        max_v = float(np.nanmax(data))
        nan_count = np.isnan(data).sum() + (data <= 0).sum()
        missing_pct = float((nan_count / data.size) * 100.0)
        
        # Read geospatial bounds if in attrs
        lat_min = float(f.attrs.get("lower_latitude", NOMINAL_LAT_MIN))
        lat_max = float(f.attrs.get("upper_latitude", NOMINAL_LAT_MAX))
        lon_min = float(f.attrs.get("left_longitude", NOMINAL_LON_MIN))
        lon_max = float(f.attrs.get("right_longitude", NOMINAL_LON_MAX))
        
        return w, h, min_v, max_v, missing_pct, lat_min, lat_max, lon_min, lon_max

def inspect_netcdf(filepath):
    """Inspects NetCDF satellite file."""
    import xarray as xr
    ds = xr.open_dataset(filepath)
    vars_list = list(ds.data_vars.keys())
    if not vars_list:
        ds.close()
        raise ValueError("No data variables in NetCDF")
    v = ds[vars_list[0]]
    h, w = v.shape[-2:]
    data = v.values
    min_v = float(np.nanmin(data))
    max_v = float(np.nanmax(data))
    nan_count = np.isnan(data).sum()
    missing_pct = float((nan_count / data.size) * 100.0)
    
    lat_min = float(ds.latitude.min()) if "latitude" in ds else NOMINAL_LAT_MIN
    lat_max = float(ds.latitude.max()) if "latitude" in ds else NOMINAL_LAT_MAX
    lon_min = float(ds.longitude.min()) if "longitude" in ds else NOMINAL_LON_MIN
    lon_max = float(ds.longitude.max()) if "longitude" in ds else NOMINAL_LON_MAX
    ds.close()
    return w, h, min_v, max_v, missing_pct, lat_min, lat_max, lon_min, lon_max

def inspect_image(filepath):
    """Inspects standard image file (JPEG, PNG)."""
    with Image.open(filepath) as img:
        img_arr = np.array(img)
        h, w = img_arr.shape[:2]
        min_v = float(np.min(img_arr))
        max_v = float(np.max(img_arr))
        # Missing pixels are often black (0) or saturated (255) if corrupted
        missing_count = np.sum(img_arr == 0)
        missing_pct = float((missing_count / img_arr.size) * 100.0)
        return w, h, min_v, max_v, missing_pct, NOMINAL_LAT_MIN, NOMINAL_LAT_MAX, NOMINAL_LON_MIN, NOMINAL_LON_MAX

def clean_insat_catalog(raw_dir=None, output_catalog_path=None, config=None):
    if config is None:
        config = load_config()
    if raw_dir is None:
        raw_dir = get_path("insat_raw_dir", config)
    if output_catalog_path is None:
        output_catalog_path = get_path("image_catalog", config)
        
    logger.info(f"Scanning INSAT-3D raw directory: {raw_dir}")
    os.makedirs(raw_dir, exist_ok=True)
    
    # Supported file extensions
    exts = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff", "*.h5", "*.hdf", "*.nc")
    found_files = []
    for ext in exts:
        found_files.extend(glob.glob(os.path.join(raw_dir, "**", ext), recursive=True))
        
    logger.info(f"Discovered {len(found_files)} potential satellite files.")
    
    catalog_rows = []
    seen_identifiers = set()
    
    for filepath in sorted(found_files):
        filename = os.path.basename(filepath)
        rel_path = os.path.relpath(filepath, BASE_DIR).replace("\\", "/")
        
        # Parse timestamp and channel
        dt, channel = parse_filename_metadata(filename)
        
        dt_str = dt.isoformat() if dt else None
        image_id = f"INSAT3D_{dt.strftime('%Y%m%d_%H%M')}_{channel}" if dt else f"INSAT3D_UNK_{filename}"
        
        # Check duplicate files
        is_duplicate = False
        dedup_key = (dt_str, channel) if dt else filename
        if dedup_key in seen_identifiers:
            is_duplicate = True
        else:
            seen_identifiers.add(dedup_key)
            
        quality_status = "VALID"
        w, h = 0, 0
        min_v, max_v, missing_pct = 0.0, 0.0, 0.0
        lat_min, lat_max, lon_min, lon_max = NOMINAL_LAT_MIN, NOMINAL_LAT_MAX, NOMINAL_LON_MIN, NOMINAL_LON_MAX
        
        # Inspect content
        ext_lower = os.path.splitext(filename)[1].lower()
        try:
            if ext_lower in [".h5", ".hdf"]:
                w, h, min_v, max_v, missing_pct, lat_min, lat_max, lon_min, lon_max = inspect_hdf5(filepath)
            elif ext_lower in [".nc"]:
                w, h, min_v, max_v, missing_pct, lat_min, lat_max, lon_min, lon_max = inspect_netcdf(filepath)
            else: # image files
                w, h, min_v, max_v, missing_pct, lat_min, lat_max, lon_min, lon_max = inspect_image(filepath)
        except Exception as e:
            logger.error(f"Corrupted or unreadable satellite file {filename}: {e}")
            quality_status = "CORRUPTED"
            
        # Quality flags
        if quality_status == "VALID":
            if dt is None:
                quality_status = "INVALID_TIMESTAMP"
            elif is_duplicate:
                quality_status = "DUPLICATE"
            elif missing_pct > config.get("quality_control", {}).get("max_missing_pixel_percentage", 20.0):
                quality_status = "WARNING_HIGH_MISSING"
                
        catalog_rows.append({
            "image_id": image_id,
            "source_file": rel_path,
            "datetime": dt_str,
            "channel": channel,
            "width": w,
            "height": h,
            "min_value": round(min_v, 2),
            "max_value": round(max_v, 2),
            "missing_pixel_percentage": round(missing_pct, 2),
            "latitude_min": lat_min,
            "latitude_max": lat_max,
            "longitude_min": lon_min,
            "longitude_max": lon_max,
            "quality_status": quality_status
        })
        
    df_catalog = pd.DataFrame(catalog_rows, columns=[
        "image_id", "source_file", "datetime", "channel",
        "width", "height", "min_value", "max_value",
        "missing_pixel_percentage", "latitude_min", "latitude_max",
        "longitude_min", "longitude_max", "quality_status"
    ])
    
    os.makedirs(os.path.dirname(output_catalog_path), exist_ok=True)
    df_catalog.to_csv(output_catalog_path, index=False)
    logger.info(f"Saved satellite image catalog to: {output_catalog_path} ({len(df_catalog)} records)")
    
    return df_catalog

if __name__ == "__main__":
    clean_insat_catalog()
