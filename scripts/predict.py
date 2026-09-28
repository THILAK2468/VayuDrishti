"""
Standalone Inference Pipeline for VayuDrishti.
Loads the trained multimodal model and outputs a clean JSON schema
ready for direct consumption by API endpoints and future Leaflet.js map visualizers.

Usage:
  python scripts/predict.py                           # Predicts on first test sample
  python scripts/predict.py --sample_id ASNA_...      # Predicts on specific sample
  python scripts/predict.py --output result.json      # Saves output to JSON file
"""
import os
import sys
import json
import argparse
import logging
from pathlib import Path
import numpy as np
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
logger = logging.getLogger("predict")

def predict_cyclone(sample_batch_item=None, sample_id=None, ckpt_path=None, config=None):
    if config is None:
        config = load_config()
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if ckpt_path is None:
        ckpt_path = Path(BASE_DIR) / "models" / "best_multitask.pt"
        
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {ckpt_path}")
        
    ckpt = torch.load(ckpt_path, map_location=device)
    model = VayuDrishtiModel(
        backbone_name=ckpt["config"]["backbone"],
        temporal_type=ckpt["config"]["temporal"],
        use_gating=ckpt["config"]["gating"]
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    
    # If no sample provided, load from test dataloader
    if sample_batch_item is None:
        _, _, test_loader = get_dataloaders(batch_size=1, config=config)
        found = False
        for b in test_loader:
            if sample_id is None or b["sample_id"][0] == sample_id:
                sample_batch_item = b
                found = True
                break
        if not found:
            raise ValueError(f"Sample ID {sample_id} not found in test split.")
            
    sat_seq = sample_batch_item["satellite_sequence"].to(device)
    era5_seq = sample_batch_item["era5_sequence"].to(device)
    track_seq = sample_batch_item["track_sequence"].to(device)
    curr_pos = sample_batch_item["current_position"].to(device)
    curr_int = sample_batch_item["current_intensity"].to(device)
    
    with torch.no_grad():
        preds = model(sat_seq, era5_seq, track_seq, curr_pos, curr_int)
        
    # Extract prediction values
    det_prob = float(preds["detection_prob"][0].cpu().numpy())
    logits = preds["intensity_logits"][0].cpu()
    probs = torch.softmax(logits, dim=-1).numpy()
    pred_cls_idx = int(np.argmax(probs))
    pred_cls_name = IDX_TO_INTENSITY.get(pred_cls_idx, "UNKNOWN")
    cls_confidence = float(probs[pred_cls_idx])
    
    # Class probabilities dictionary
    class_probs_dict = {
        IDX_TO_INTENSITY.get(i, f"CLS_{i}"): round(float(probs[i]), 4)
        for i in range(len(probs))
    }
    
    # Current state
    c_lat = float(curr_pos[0, 0].cpu().numpy())
    c_lon = float(curr_pos[0, 1].cpu().numpy())
    c_wind = float(curr_int[0, 0].cpu().numpy())
    c_pres = float(curr_int[0, 1].cpu().numpy())
    
    # Forecast outputs (+1h, +3h, +6h)
    track_pred = preds["future_track"][0].cpu().numpy()     # (3, 2) -> [[lat1, lon1], [lat3, lon3], [lat6, lon6]]
    int_pred = preds["future_intensity"][0].cpu().numpy()   # (3, 2) -> [[w1, p1], [w3, p3], [w6, p6]]
    
    horizons = ["1h", "3h", "6h"]
    forecast_dict = {}
    for h_idx, h_key in enumerate(horizons):
        f_lat = float(track_pred[h_idx, 0])
        f_lon = float(track_pred[h_idx, 1])
        f_wind = float(int_pred[h_idx, 0])
        f_pres = float(int_pred[h_idx, 1])
        
        # Great circle displacement from current position
        dist_km = float(haversine_distance_np(c_lat, c_lon, f_lat, f_lon))
        
        forecast_dict[h_key] = {
            "latitude": round(f_lat, 2),
            "longitude": round(f_lon, 2),
            "displacement_km": round(dist_km, 1),
            "max_sustained_wind_kt": round(f_wind, 1),
            "central_pressure_hpa": round(f_pres, 1),
            "confidence_interval": {
                "track_radius_uncertainty_km": round(15.0 * (h_idx + 1), 1),
                "wind_uncertainty_kt": round(5.0 * (h_idx + 1), 1)
            }
        }
        
    # Extrapolate +12h and +24h if requested by specification
    # Velocity vector between +3h and +6h
    d_lat_3h = (track_pred[2, 0] - track_pred[1, 0])
    d_lon_3h = (track_pred[2, 1] - track_pred[1, 1])
    d_w_3h = (int_pred[2, 0] - int_pred[1, 0])
    d_p_3h = (int_pred[2, 1] - int_pred[1, 1])
    
    # +12h (+6h further from +6h)
    lat_12h = float(track_pred[2, 0] + d_lat_3h * 2.0)
    lon_12h = float(track_pred[2, 1] + d_lon_3h * 2.0)
    w_12h = max(15.0, float(int_pred[2, 0] + d_w_3h * 2.0))
    p_12h = float(int_pred[2, 1] + d_p_3h * 2.0)
    forecast_dict["12h"] = {
        "latitude": round(lat_12h, 2),
        "longitude": round(lon_12h, 2),
        "displacement_km": round(float(haversine_distance_np(c_lat, c_lon, lat_12h, lon_12h)), 1),
        "max_sustained_wind_kt": round(w_12h, 1),
        "central_pressure_hpa": round(p_12h, 1),
        "confidence_interval": {"track_radius_uncertainty_km": 60.0, "wind_uncertainty_kt": 15.0}
    }
    
    # +24h (+18h further from +6h)
    lat_24h = float(track_pred[2, 0] + d_lat_3h * 6.0)
    lon_24h = float(track_pred[2, 1] + d_lon_3h * 6.0)
    w_24h = max(15.0, float(int_pred[2, 0] + d_w_3h * 6.0))
    p_24h = float(int_pred[2, 1] + d_p_3h * 6.0)
    forecast_dict["24h"] = {
        "latitude": round(lat_24h, 2),
        "longitude": round(lon_24h, 2),
        "displacement_km": round(float(haversine_distance_np(c_lat, c_lon, lat_24h, lon_24h)), 1),
        "max_sustained_wind_kt": round(w_24h, 1),
        "central_pressure_hpa": round(p_24h, 1),
        "confidence_interval": {"track_radius_uncertainty_km": 120.0, "wind_uncertainty_kt": 25.0}
    }
    
    # Modality contributions from gating
    m_w = preds["modality_weights"][0].cpu().numpy()
    modality_contrib = {
        "satellite_imagery": round(float(m_w[0]), 3),
        "era5_atmospheric": round(float(m_w[1]), 3),
        "ibtracs_track_history": round(float(m_w[2]), 3)
    }
    
    # Final structured output
    output_json = {
        "sample_id": sample_batch_item["sample_id"][0],
        "storm_name": sample_batch_item["storm_name"][0],
        "reference_datetime_utc": sample_batch_item["datetime"][0],
        "cyclone_probability": round(det_prob, 4),
        "is_cyclone_detected": bool(det_prob >= 0.5),
        "intensity_classification": {
            "predicted_class": pred_cls_name,
            "confidence": round(cls_confidence, 4),
            "class_probabilities": class_probs_dict
        },
        "current_position": {
            "latitude": round(c_lat, 2),
            "longitude": round(c_lon, 2),
            "wind_kt": round(c_wind, 1),
            "pressure_hpa": round(c_pres, 1)
        },
        "forecast": forecast_dict,
        "modality_contributions": modality_contrib,
        "leaflet_geojson_track": {
            "type": "LineString",
            "coordinates": [
                [round(c_lon, 2), round(c_lat, 2)],
                [round(forecast_dict["1h"]["longitude"], 2), round(forecast_dict["1h"]["latitude"], 2)],
                [round(forecast_dict["3h"]["longitude"], 2), round(forecast_dict["3h"]["latitude"], 2)],
                [round(forecast_dict["6h"]["longitude"], 2), round(forecast_dict["6h"]["latitude"], 2)],
                [round(forecast_dict["12h"]["longitude"], 2), round(forecast_dict["12h"]["latitude"], 2)],
                [round(forecast_dict["24h"]["longitude"], 2), round(forecast_dict["24h"]["latitude"], 2)]
            ]
        }
    }
    
    return output_json

def main():
    parser = argparse.ArgumentParser(description="VayuDrishti Cyclone Inference")
    parser.add_argument("--sample_id", type=str, default=None, help="Target sample ID")
    parser.add_argument("--output", type=str, default=None, help="Output JSON filepath")
    args = parser.parse_args()
    
    res = predict_cyclone(sample_id=args.sample_id)
    json_str = json.dumps(res, indent=2)
    print(json_str)
    
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(json_str)
        logger.info(f"Saved prediction result to: {args.output}")

if __name__ == "__main__":
    main()
