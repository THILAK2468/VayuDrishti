"""
Master Pipeline Orchestrator for VayuDrishti.
Executes the reproducible data-preparation pipeline end-to-end or by specific step.

Usage:
  python scripts/run_pipeline.py               # Runs complete end-to-end pipeline
  python scripts/run_pipeline.py --step 1      # Runs specific step (1 to 12)
  python scripts/run_pipeline.py --sample      # Runs pipeline on sample subset
"""
import os
import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import load_config, get_path, BASE_DIR

# Import step modules
from clean_ibtracs import clean_ibtracs
from clean_insat import clean_insat_catalog
from clean_era5 import clean_and_catalog_era5
from sync_and_match import synchronize_and_match
from split_data import perform_storm_level_split
from build_dataset import build_final_dataset
from leakage_check import check_leakage_and_compute_normalization
from generate_report import generate_reports

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("run_pipeline")

def run_all(config=None, sample_mode=False):
    if config is None:
        config = load_config()
        
    logger.info("==================================================")
    logger.info("  STARTING VAYUDRISHTI DATA PREPARATION PIPELINE  ")
    logger.info("==================================================")
    
    # Step 1: Clean IBTrACS
    logger.info(">>> STEP 1: IBTrACS BEST-TRACK CLEANING")
    clean_ibtracs(config=config)
    
    # Step 2: Clean and catalog INSAT-3D
    logger.info(">>> STEP 2: INSAT-3D SATELLITE CATALOGING")
    clean_insat_catalog(config=config)
    
    # Step 3: Clean and catalog ERA5
    logger.info(">>> STEP 3: ERA5 ENVIRONMENTAL NETCDF CLEANING")
    clean_and_catalog_era5(config=config)
    
    # Steps 4, 5, 6, 7, 8: Sync, Spatial Matching, QC, Labeling, Master Dataset
    logger.info(">>> STEPS 4-8: MULTI-SOURCE SYNCHRONIZATION & QUALITY CONTROL")
    synchronize_and_match(config=config)
    
    # Step 9: Train / Val / Test Split
    logger.info(">>> STEP 9: STORM-LEVEL TRAIN/VAL/TEST PARTITIONING")
    perform_storm_level_split(config=config)
    
    # Step 10: Build Final Dataset
    logger.info(">>> STEP 10: FINAL DATASET PACKAGING")
    build_final_dataset(config=config)
    
    # Step 11: Leakage Check & Train-Only Normalization
    logger.info(">>> STEP 11: AUTOMATED LEAKAGE CHECK & NORMALIZATION")
    check_leakage_and_compute_normalization(config=config)
    
    # Step 12: Reports
    logger.info(">>> STEP 12: METRIC & QUALITY AUDIT REPORTING")
    generate_reports(config=config)
    
    logger.info("==================================================")
    logger.info("  VAYUDRISHTI PIPELINE EXECUTED SUCCESSFULLY!     ")
    logger.info("==================================================")

def main():
    parser = argparse.ArgumentParser(description="VayuDrishti Data Pipeline Runner")
    parser.add_argument("--step", type=int, choices=range(1, 13), help="Run a specific step (1 to 12)")
    parser.add_argument("--sample", action="store_true", help="Run on a sample subset for fast validation")
    args = parser.parse_args()
    
    config = load_config()
    
    if args.step:
        step = args.step
        logger.info(f"Executing step {step} only...")
        if step == 1:
            clean_ibtracs(config=config)
        elif step == 2:
            clean_insat_catalog(config=config)
        elif step == 3:
            clean_and_catalog_era5(config=config)
        elif step in [4, 5, 6, 7, 8]:
            synchronize_and_match(config=config)
        elif step == 9:
            perform_storm_level_split(config=config)
        elif step == 10:
            build_final_dataset(config=config)
        elif step == 11:
            check_leakage_and_compute_normalization(config=config)
        elif step == 12:
            generate_reports(config=config)
    else:
        run_all(config=config, sample_mode=args.sample)

if __name__ == "__main__":
    main()
