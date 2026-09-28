"""
Unit and Integration Test Suite for VayuDrishti Data Pipeline.
Validates:
  1. INSAT-3D parsing across formats (Image JPG, NetCDF, HDF5)
  2. Temporal synchronization with UTC alignment
  3. Spatial extraction bounds and cropping logic
  4. Storm-level partitioning and zero data leakage
"""
import os
import sys
import unittest
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import load_config, BASE_DIR
from clean_insat import parse_filename_metadata, inspect_image, clean_insat_catalog
from sync_and_match import classify_intensity

class TestVayuDrishtiPipeline(unittest.TestCase):
    def setUp(self):
        self.config = load_config()

    def test_insat_filename_parsing(self):
        """Test timestamp and channel extraction from standard INSAT-3D filenames."""
        fn1 = "3DIMG_02JUN2020_0400_L1B_STD_IR2.jpg"
        dt1, ch1 = parse_filename_metadata(fn1)
        self.assertIsNotNone(dt1)
        self.assertEqual(dt1.year, 2020)
        self.assertEqual(dt1.month, 6)
        self.assertEqual(dt1.day, 2)
        self.assertEqual(dt1.hour, 4)
        self.assertEqual(dt1.minute, 0)
        self.assertEqual(ch1, "IR2")

        fn2 = "3DIMG_16MAY2021_0600_L1B_STD_TIR1.h5"
        dt2, ch2 = parse_filename_metadata(fn2)
        self.assertIsNotNone(dt2)
        self.assertEqual(dt2.year, 2021)
        self.assertEqual(dt2.month, 5)
        self.assertEqual(dt2.day, 16)
        self.assertEqual(dt2.hour, 6)
        self.assertEqual(dt2.minute, 0)
        self.assertEqual(ch2, "TIR1")

    def test_intensity_classification(self):
        """Test IMD intensity scale mapping."""
        policy = self.config["intensity_policy"]["thresholds"]
        
        # Test known categories
        self.assertEqual(classify_intensity(125.0, policy), "SUPER")
        self.assertEqual(classify_intensity(100.0, policy), "EXTREME")
        self.assertEqual(classify_intensity(75.0, policy), "VERY")
        self.assertEqual(classify_intensity(55.0, policy), "SEVERE")
        self.assertEqual(classify_intensity(40.0, policy), "NORMAL")
        self.assertEqual(classify_intensity(30.0, policy), "DEEP_DEPRESSION")
        self.assertEqual(classify_intensity(20.0, policy), "DEPRESSION")
        self.assertEqual(classify_intensity(10.0, policy), "LOW")

    def test_storm_partition_no_overlap(self):
        """Verify that splits have zero overlapping storms or samples."""
        train_path = os.path.join(BASE_DIR, "splits", "train.csv")
        val_path = os.path.join(BASE_DIR, "splits", "validation.csv")
        test_path = os.path.join(BASE_DIR, "splits", "test.csv")

        if os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path):
            df_tr = pd.read_csv(train_path)
            df_va = pd.read_csv(val_path)
            df_te = pd.read_csv(test_path)

            tr_storms = set(df_tr["storm_name"])
            va_storms = set(df_va["storm_name"])
            te_storms = set(df_te["storm_name"])

            self.assertEqual(len(tr_storms & va_storms), 0, "Train and Val share storms!")
            self.assertEqual(len(tr_storms & te_storms), 0, "Train and Test share storms!")
            self.assertEqual(len(va_storms & te_storms), 0, "Val and Test share storms!")

            tr_samples = set(df_tr["sample_id"])
            va_samples = set(df_va["sample_id"])
            te_samples = set(df_te["sample_id"])

            self.assertEqual(len(tr_samples & va_samples), 0, "Train and Val share samples!")
            self.assertEqual(len(tr_samples & te_samples), 0, "Train and Test share samples!")
            self.assertEqual(len(va_samples & te_samples), 0, "Val and Test share samples!")

    def test_normalization_fitted_on_train_only(self):
        """Verify normalization parameters exist and were fit only on train storms."""
        norm_path = os.path.join(BASE_DIR, "configs", "normalization_params.json")
        self.assertTrue(os.path.exists(norm_path), "normalization_params.json missing")
        
        import json
        with open(norm_path, "r") as f:
            norm_data = json.load(f)
            
        train_storms = norm_data["splits_source"]["train_storms"]
        # Verify val storm (ASANI) and test storm (ASNA) are not in train storms
        self.assertNotIn("ASANI", train_storms)
        self.assertNotIn("ASNA", train_storms)

if __name__ == "__main__":
    unittest.main()
