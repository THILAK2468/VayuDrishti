"""
Error Analysis Module for VayuDrishti.
Automatically identifies:
  - False positives / False negatives in cyclone detection
  - Incorrect intensity class predictions (classification errors)
  - Large track prediction errors (> 150 km)
  - Large wind forecast errors (> 15 kt)
Outputs:
  - reports/error_analysis.csv
"""
import os
import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import load_config, BASE_DIR
from dataset import get_dataloaders, IDX_TO_INTENSITY
from models import VayuDrishtiModel
from losses import haversine_distance_np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("error_analysis")

def run_error_analysis(config=None, track_thresh_km=150.0, wind_thresh_kt=15.0):
    if config is None:
        config = load_config()
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models_dir = Path(BASE_DIR) / "models"
    ckpt_path = models_dir / "best_multitask.pt"
    
    if not ckpt_path.exists():
        logger.error("Checkpoint not found at models/best_multitask.pt")
        return
        
    logger.info(f"Loading checkpoint {ckpt_path} for error analysis...")
    ckpt = torch.load(ckpt_path, map_location=device)
    model = VayuDrishtiModel(
        backbone_name=ckpt["config"]["backbone"],
        temporal_type=ckpt["config"]["temporal"],
        use_gating=ckpt["config"]["gating"]
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    
    # Evaluate across Test set
    _, _, test_loader = get_dataloaders(batch_size=8, config=config)
    
    error_records = []
    
    with torch.no_grad():
        for batch in test_loader:
            sat_seq = batch["satellite_sequence"].to(device)
            era5_seq = batch["era5_sequence"].to(device)
            track_seq = batch["track_sequence"].to(device)
            curr_pos = batch["current_position"].to(device)
            curr_int = batch["current_intensity"].to(device)
            
            preds = model(sat_seq, era5_seq, track_seq, curr_pos, curr_int)
            
            det_probs = preds["detection_prob"].cpu().numpy()
            det_trues = batch["detection_label"].numpy()
            
            cls_logits = preds["intensity_logits"].cpu().numpy()
            cls_preds = np.argmax(cls_logits, axis=-1)
            cls_trues = batch["intensity_class"].numpy()
            
            track_preds = preds["future_track"].cpu().numpy() # (B, 3, 2)
            track_trues = batch["target_tracks"].numpy()
            track_masks = batch["target_masks"].numpy()
            
            int_preds = preds["future_intensity"].cpu().numpy() # (B, 3, 2)
            int_trues = batch["target_intensities"].numpy()
            
            sample_ids = batch["sample_id"]
            storm_names = batch["storm_name"]
            datetimes = batch["datetime"]
            
            for i in range(len(sample_ids)):
                sid = sample_ids[i]
                sname = storm_names[i]
                dt = datetimes[i]
                
                # Detection error check
                d_prob = float(det_probs[i])
                d_true = float(det_trues[i])
                is_fp = (d_prob >= 0.5 and d_true == 0.0)
                is_fn = (d_prob < 0.5 and d_true == 1.0)
                
                # Intensity class check
                c_pred_idx = int(cls_preds[i])
                c_true_idx = int(cls_trues[i])
                c_pred_name = IDX_TO_INTENSITY.get(c_pred_idx, "UNKNOWN")
                c_true_name = IDX_TO_INTENSITY.get(c_true_idx, "UNKNOWN")
                is_cls_err = (c_pred_idx != c_true_idx)
                
                # Track error at +1h
                if track_masks[i, 0] > 0.5:
                    d_km = haversine_distance_np(
                        track_trues[i, 0, 0], track_trues[i, 0, 1],
                        track_preds[i, 0, 0], track_preds[i, 0, 1]
                    )
                else:
                    d_km = 0.0
                is_large_track_err = (d_km > track_thresh_km)
                
                # Wind error at +1h
                if track_masks[i, 0] > 0.5:
                    w_err = abs(int_preds[i, 0, 0] - int_trues[i, 0, 0])
                else:
                    w_err = 0.0
                is_large_wind_err = (w_err > wind_thresh_kt)
                
                # Flag record if any error occurred
                error_types = []
                if is_fp: error_types.append("FALSE_POSITIVE")
                if is_fn: error_types.append("FALSE_NEGATIVE")
                if is_cls_err: error_types.append(f"CLASS_MISMATCH:pred_{c_pred_name}_vs_true_{c_true_name}")
                if is_large_track_err: error_types.append(f"LARGE_TRACK_ERR_{d_km:.1f}km")
                if is_large_wind_err: error_types.append(f"LARGE_WIND_ERR_{w_err:.1f}kt")
                
                error_records.append({
                    "sample_id": sid,
                    "storm_name": sname,
                    "datetime": dt,
                    "detection_prob": round(d_prob, 3),
                    "detection_true": int(d_true),
                    "intensity_pred": c_pred_name,
                    "intensity_true": c_true_name,
                    "is_class_error": is_cls_err,
                    "track_1h_error_km": round(float(d_km), 2),
                    "wind_1h_error_kt": round(float(w_err), 2),
                    "has_error_flag": len(error_types) > 0,
                    "error_reasons": "; ".join(error_types) if error_types else "NONE"
                })
                
    df_err = pd.DataFrame(error_records)
    rep_csv = Path(BASE_DIR) / "reports" / "error_analysis.csv"
    df_err.to_csv(rep_csv, index=False)
    
    total_eval = len(df_err)
    n_err = df_err["has_error_flag"].sum()
    n_cls_err = df_err["is_class_error"].sum()
    n_track_err = (df_err["track_1h_error_km"] > track_thresh_km).sum()
    n_wind_err = (df_err["wind_1h_error_kt"] > wind_thresh_kt).sum()
    
    logger.info("=== Error Analysis Audit Summary ===")
    logger.info(f"Total Evaluated Test Samples: {total_eval}")
    logger.info(f"Intensity Classification Errors: {n_cls_err} ({n_cls_err/total_eval*100:.1f}%)")
    logger.info(f"Track Errors (> {track_thresh_km} km): {n_track_err} ({n_track_err/total_eval*100:.1f}%)")
    logger.info(f"Wind Errors (> {wind_thresh_kt} kt):   {n_wind_err} ({n_wind_err/total_eval*100:.1f}%)")
    logger.info(f"Saved detailed error report to: {rep_csv}")
    
    return df_err

if __name__ == "__main__":
    run_error_analysis()
