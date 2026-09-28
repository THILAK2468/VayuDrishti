"""
Step 1: IBTrACS Cleaning Module for VayuDrishti
Cleans and standardizes the IBTrACS North Indian Ocean cyclone best-track dataset.
"""
import os
import sys
import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import load_config, get_path, BASE_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("clean_ibtracs")

def clean_ibtracs(raw_csv_path=None, output_csv_path=None, config=None):
    if config is None:
        config = load_config()
    
    if raw_csv_path is None:
        raw_csv_path = get_path("ibtracs_raw", config)
    if output_csv_path is None:
        output_csv_path = get_path("ibtracs_cleaned", config)
        
    logger.info(f"Starting IBTrACS cleaning. Input: {raw_csv_path}")
    
    if not os.path.exists(raw_csv_path):
        raise FileNotFoundError(f"Raw IBTrACS file not found at {raw_csv_path}")
        
    # IBTrACS row 0 is header, row 1 is units/metadata (e.g., 'Year', 'mb', 'km')
    # Read raw dataframe
    df_raw = pd.read_csv(raw_csv_path, header=0, skiprows=[1], low_memory=False)
    original_row_count = len(df_raw)
    logger.info(f"Loaded raw IBTrACS data. Total rows: {original_row_count}")

    # Track cleaning metrics
    report = {
        "original_row_count": original_row_count,
        "removed_rows": 0,
        "duplicate_rows": 0,
        "invalid_timestamps": 0,
        "invalid_coordinates": 0,
        "missing_wind": 0,
        "missing_pressure": 0,
        "final_row_count": 0,
        "removal_reasons": {}
    }
    
    # 1. Parse timestamps to UTC datetime
    # IBTrACS format is typically 'YYYY-MM-DD HH:MM:SS'
    logger.info("Parsing ISO_TIME to UTC datetime...")
    df_raw["datetime_utc"] = pd.to_datetime(df_raw["ISO_TIME"], errors="coerce", utc=True)
    invalid_time_mask = df_raw["datetime_utc"].isna()
    report["invalid_timestamps"] = int(invalid_time_mask.sum())
    if report["invalid_timestamps"] > 0:
        logger.warning(f"Found {report['invalid_timestamps']} records with invalid timestamps.")
    
    # 2. Convert coordinates to numeric and validate
    logger.info("Parsing and validating coordinates (LAT, LON)...")
    df_raw["LAT_num"] = pd.to_numeric(df_raw["LAT"], errors="coerce")
    df_raw["LON_num"] = pd.to_numeric(df_raw["LON"], errors="coerce")
    
    # Latitude must be within [-90, 90], Longitude [-180, 180]
    # For North Indian Ocean specifically, latitude is generally between -10 and 45, longitude 30 to 120
    invalid_coord_mask = (
        df_raw["LAT_num"].isna() | 
        df_raw["LON_num"].isna() |
        (df_raw["LAT_num"] < -90) | (df_raw["LAT_num"] > 90) |
        (df_raw["LON_num"] < -180) | (df_raw["LON_num"] > 180)
    )
    report["invalid_coordinates"] = int(invalid_coord_mask.sum())
    if report["invalid_coordinates"] > 0:
        logger.warning(f"Found {report['invalid_coordinates']} records with invalid coordinates.")
        
    # 3. Numeric conversion for wind speed and pressure
    logger.info("Parsing wind speed and pressure metrics...")
    # Priority for North Indian Ocean:
    # IMD (NEWDELHI_WIND) -> WMO (WMO_WIND) -> USA/JTWC (USA_WIND)
    for col in ["NEWDELHI_WIND", "WMO_WIND", "USA_WIND", "NEWDELHI_PRES", "WMO_PRES", "USA_PRES"]:
        if col in df_raw.columns:
            df_raw[f"{col}_num"] = pd.to_numeric(df_raw[col], errors="coerce")
        else:
            df_raw[f"{col}_num"] = np.nan
            
    # Derive consensus wind speed (knots) and source
    def resolve_wind(row):
        if pd.notna(row.get("NEWDELHI_WIND_num")):
            return row["NEWDELHI_WIND_num"], "NEWDELHI"
        elif pd.notna(row.get("WMO_WIND_num")):
            return row["WMO_WIND_num"], "WMO"
        elif pd.notna(row.get("USA_WIND_num")):
            return row["USA_WIND_num"], "USA"
        return np.nan, "NONE"
        
    def resolve_pres(row):
        if pd.notna(row.get("NEWDELHI_PRES_num")):
            return row["NEWDELHI_PRES_num"], "NEWDELHI"
        elif pd.notna(row.get("WMO_PRES_num")):
            return row["WMO_PRES_num"], "WMO"
        elif pd.notna(row.get("USA_PRES_num")):
            return row["USA_PRES_num"], "USA"
        return np.nan, "NONE"

    wind_res = df_raw.apply(resolve_wind, axis=1)
    df_raw["wind_speed_kt"] = [w[0] for w in wind_res]
    df_raw["wind_source"] = [w[1] for w in wind_res]
    
    pres_res = df_raw.apply(resolve_pres, axis=1)
    df_raw["pressure_hpa"] = [p[0] for p in pres_res]
    df_raw["pressure_source"] = [p[1] for p in pres_res]
    
    report["missing_wind"] = int(df_raw["wind_speed_kt"].isna().sum())
    report["missing_pressure"] = int(df_raw["pressure_hpa"].isna().sum())

    # 4. Standardize storm identification
    df_raw["SID"] = df_raw["SID"].astype(str).str.strip()
    df_raw["NAME"] = df_raw["NAME"].astype(str).str.strip().str.upper()
    df_raw["SEASON"] = pd.to_numeric(df_raw["SEASON"], errors="coerce")
    df_raw["BASIN"] = df_raw["BASIN"].astype(str).str.strip()
    df_raw["SUBBASIN"] = df_raw["SUBBASIN"].astype(str).str.strip() if "SUBBASIN" in df_raw.columns else "NI"
    df_raw["NATURE"] = df_raw["NATURE"].astype(str).str.strip() if "NATURE" in df_raw.columns else "TS"

    # Filter out fatal errors (invalid time or invalid coordinates)
    fatal_mask = invalid_time_mask | invalid_coord_mask
    df_clean = df_raw[~fatal_mask].copy()
    
    # 5. Remove duplicates: same storm (SID) and timestamp (datetime_utc)
    # If duplicates exist, keep the one with non-null wind and pressure
    initial_valid_count = len(df_clean)
    df_clean.sort_values(by=["SID", "datetime_utc", "wind_speed_kt"], ascending=[True, True, False], inplace=True)
    duplicate_mask = df_clean.duplicated(subset=["SID", "datetime_utc"], keep="first")
    report["duplicate_rows"] = int(duplicate_mask.sum())
    if report["duplicate_rows"] > 0:
        logger.warning(f"Found and removing {report['duplicate_rows']} duplicate storm-time records.")
        df_clean = df_clean[~duplicate_mask].copy()
        
    df_clean.sort_values(by=["SID", "datetime_utc"], inplace=True)
    df_clean.reset_index(drop=True, inplace=True)
    
    # Select clean output columns
    output_cols = [
        "SID", "NAME", "SEASON", "BASIN", "SUBBASIN", "NATURE",
        "datetime_utc", "LAT_num", "LON_num",
        "wind_speed_kt", "wind_source", "pressure_hpa", "pressure_source",
        "WMO_WIND_num", "WMO_PRES_num", "NEWDELHI_WIND_num", "NEWDELHI_PRES_num", "USA_WIND_num", "USA_PRES_num"
    ]
    # Rename for cleaner access
    rename_dict = {
        "datetime_utc": "datetime",
        "LAT_num": "latitude",
        "LON_num": "longitude",
        "wind_speed_kt": "wind_speed",
        "pressure_hpa": "pressure",
        "WMO_WIND_num": "wmo_wind",
        "WMO_PRES_num": "wmo_pressure",
        "NEWDELHI_WIND_num": "newdelhi_wind",
        "NEWDELHI_PRES_num": "newdelhi_pressure",
        "USA_WIND_num": "usa_wind",
        "USA_PRES_num": "usa_pressure"
    }
    
    df_clean_out = df_clean[output_cols].rename(columns=rename_dict)
    
    final_row_count = len(df_clean_out)
    report["removed_rows"] = original_row_count - final_row_count
    report["final_row_count"] = final_row_count
    
    # Save cleaned CSV
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df_clean_out.to_csv(output_csv_path, index=False)
    logger.info(f"Saved cleaned IBTrACS dataset to: {output_csv_path} ({final_row_count} rows)")
    
    # Save cleaning report JSON
    report_path = Path(BASE_DIR) / "metadata" / "ibtracs_cleaning_report.json"
    os.makedirs(report_path.parent, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Cleaning report saved to: {report_path}")
    
    logger.info("=== IBTrACS Cleaning Summary ===")
    logger.info(f"Original Row Count: {report['original_row_count']}")
    logger.info(f"Removed Rows:       {report['removed_rows']}")
    logger.info(f"Duplicate Rows:     {report['duplicate_rows']}")
    logger.info(f"Invalid Timestamps: {report['invalid_timestamps']}")
    logger.info(f"Invalid Coords:     {report['invalid_coordinates']}")
    logger.info(f"Missing Wind:       {report['missing_wind']}")
    logger.info(f"Missing Pressure:   {report['missing_pressure']}")
    logger.info(f"Final Row Count:    {report['final_row_count']}")
    
    return df_clean_out, report

if __name__ == "__main__":
    clean_ibtracs()
