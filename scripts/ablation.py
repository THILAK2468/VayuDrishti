"""
Ablation Experiments and Model Comparison Module for VayuDrishti.
Runs controlled experiments A through H:
  - Exp A: Satellite Only (Baseline 1)
  - Exp B: Satellite + ERA5 (Baseline 4)
  - Exp C: Satellite + Track
  - Exp D: Satellite + ERA5 + Track (Concat Fusion)
  - Exp F: Gated Multimodal Fusion (Proposed VayuDrishti)
  - Exp G: Temporal Comparison (GRU vs ConvLSTM)
  - Exp H: Vision Backbone (MobileViT-XS vs ConvNeXt-V2-Atto)
Outputs:
  - reports/ablation_results.csv
  - reports/model_comparison.csv
"""
import os
import sys
import time
import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import load_config, get_path, BASE_DIR
from dataset import get_dataloaders
from models import VayuDrishtiModel
from losses import MultiTaskLoss
from train import train_epoch, evaluate_epoch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("ablation")

def benchmark_inference_latency(model, sample_batch, device, n_runs=10):
    model.eval()
    sat_seq = sample_batch["satellite_sequence"][:1].to(device)
    era5_seq = sample_batch["era5_sequence"][:1].to(device)
    track_seq = sample_batch["track_sequence"][:1].to(device)
    curr_pos = sample_batch["current_position"][:1].to(device)
    curr_int = sample_batch["current_intensity"][:1].to(device)
    
    with torch.no_grad():
        for _ in range(2):
            _ = model(sat_seq, era5_seq, track_seq, curr_pos, curr_int)
            
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = model(sat_seq, era5_seq, track_seq, curr_pos, curr_int)
            times.append(time.perf_counter() - t0)
            
    latency_ms = float(np.mean(times) * 1000.0)
    return round(latency_ms, 2)

def train_ablation_epoch(model, dataloader, criterion, optimizer, device, max_batches=12):
    model.train()
    running_loss = 0.0
    count = 0
    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break
        optimizer.zero_grad()
        sat_seq = batch["satellite_sequence"].to(device)
        era5_seq = batch["era5_sequence"].to(device)
        track_seq = batch["track_sequence"].to(device)
        curr_pos = batch["current_position"].to(device)
        curr_int = batch["current_intensity"].to(device)
        
        preds = model(sat_seq, era5_seq, track_seq, curr_pos=curr_pos, curr_int=curr_int)
        loss_dict = criterion(preds, batch)
        loss_dict["total_loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        bs = sat_seq.shape[0]
        running_loss += loss_dict["total_loss"].item() * bs
        count += bs
    return running_loss / max(1, count)

def run_ablation_experiments(epochs=1, batch_size=8, config=None):
    if config is None:
        config = load_config()
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=batch_size, config=config)
    sample_batch = next(iter(test_loader))
    
    criterion = MultiTaskLoss()
    reports_dir = Path(BASE_DIR) / "reports"
    os.makedirs(reports_dir, exist_ok=True)
    
    experiments = [
        {
            "id": "Exp_A",
            "name": "Satellite Only",
            "backbone": "mobilevit_xs",
            "temporal": "convlstm",
            "use_sat": True, "use_era5": False, "use_track": False,
            "use_gating": False,
            "desc": "Baseline 1: Satellite imagery only with ConvLSTM"
        },
        {
            "id": "Exp_B",
            "name": "Satellite + ERA5",
            "backbone": "mobilevit_xs",
            "temporal": "convlstm",
            "use_sat": True, "use_era5": True, "use_track": False,
            "use_gating": False,
            "desc": "Baseline 4: Satellite and atmospheric data without historical track"
        },
        {
            "id": "Exp_C",
            "name": "Satellite + Track",
            "backbone": "mobilevit_xs",
            "temporal": "convlstm",
            "use_sat": True, "use_era5": False, "use_track": True,
            "use_gating": False,
            "desc": "Satellite and IBTrACS track history without ERA5"
        },
        {
            "id": "Exp_D",
            "name": "Full Concat Fusion",
            "backbone": "mobilevit_xs",
            "temporal": "convlstm",
            "use_sat": True, "use_era5": True, "use_track": True,
            "use_gating": False,
            "desc": "All 3 modalities with standard linear concatenation fusion"
        },
        {
            "id": "Exp_F",
            "name": "Proposed Gated Multimodal",
            "backbone": "mobilevit_xs",
            "temporal": "convlstm",
            "use_sat": True, "use_era5": True, "use_track": True,
            "use_gating": True,
            "desc": "Proposed VayuDrishti model with learnable modality gating"
        },
        {
            "id": "Exp_G",
            "name": "Temporal GRU",
            "backbone": "mobilevit_xs",
            "temporal": "gru",
            "use_sat": True, "use_era5": True, "use_track": True,
            "use_gating": True,
            "desc": "Lightweight GRU baseline instead of ConvLSTM"
        },
        {
            "id": "Exp_H",
            "name": "ConvNeXt-V2-Atto Backbone",
            "backbone": "convnextv2_atto",
            "temporal": "convlstm",
            "use_sat": True, "use_era5": True, "use_track": True,
            "use_gating": True,
            "desc": "Alternative lightweight vision backbone for comparison"
        }
    ]
    
    results = []
    logger.info(f"Starting execution of {len(experiments)} controlled ablation experiments...")
    
    for exp in experiments:
        exp_id = exp["id"]
        exp_name = exp["name"]
        logger.info(f"\n>>> Running {exp_id}: {exp_name} ({exp['desc']})...")
        
        # Build model
        model = VayuDrishtiModel(
            backbone_name=exp["backbone"],
            temporal_type=exp["temporal"],
            use_gating=exp["use_gating"],
            use_sat=exp["use_sat"],
            use_era5=exp["use_era5"],
            use_track=exp["use_track"]
        ).to(device)
        
        if model.sat_encoder is not None:
            model.sat_encoder.freeze_backbone()
            
        params_count = sum(p.numel() for p in model.parameters())
        model_size_mb = (params_count * 4) / (1024 * 1024)
        
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        
        t_start = time.time()
        for ep in range(epochs):
            _ = train_ablation_epoch(model, train_loader, criterion, optimizer, device, max_batches=12)
        train_time_sec = time.time() - t_start
        
        val_metrics, _, _ = evaluate_epoch(model, val_loader, criterion, device)
        test_metrics, _, _ = evaluate_epoch(model, test_loader, criterion, device)
        
        latency_ms = benchmark_inference_latency(model, sample_batch, device)
        
        h1_track_km = test_metrics["track_metrics"].get("+1_haversine_mean_km", 0.0)
        h1_wind_mae = test_metrics["intensity_metrics"].get("+1_wind_mae_kt", 0.0)
        
        res_entry = {
            "experiment_id": exp_id,
            "model_name": exp_name,
            "backbone": exp["backbone"],
            "temporal_module": exp["temporal"],
            "modalities": f"{'Sat+' if exp['use_sat'] else ''}{'ERA5+' if exp['use_era5'] else ''}{'Track' if exp['use_track'] else ''}".rstrip('+'),
            "fusion_type": "Gated" if exp["use_gating"] else "Concat",
            "parameters": params_count,
            "model_size_mb": round(model_size_mb, 2),
            "train_time_sec": round(train_time_sec, 1),
            "inference_latency_ms": latency_ms,
            "val_loss": round(val_metrics["val_loss"], 4),
            "val_accuracy": round(val_metrics["accuracy"] * 100.0, 2),
            "val_macro_f1": round(val_metrics["macro_f1"], 4),
            "test_accuracy": round(test_metrics["accuracy"] * 100.0, 2),
            "test_macro_f1": round(test_metrics["macro_f1"], 4),
            "test_track_1h_km": h1_track_km,
            "test_wind_1h_mae_kt": h1_wind_mae
        }
        results.append(res_entry)
        logger.info(f"[{exp_id}] Params: {params_count:,} | Size: {res_entry['model_size_mb']}MB | Latency: {latency_ms:.1f}ms | Val Acc: {res_entry['val_accuracy']}% | Macro F1: {res_entry['val_macro_f1']} | Track 1h: {h1_track_km}km | Wind MAE: {h1_wind_mae}kt")
        
    df_results = pd.DataFrame(results)
    ablation_csv = reports_dir / "ablation_results.csv"
    df_results.to_csv(ablation_csv, index=False)
    logger.info(f"\nSaved ablation experiment results to: {ablation_csv}")
    
    comp_df = df_results[df_results["experiment_id"].isin(["Exp_F", "Exp_H"])].copy()
    comp_csv = reports_dir / "model_comparison.csv"
    comp_df.to_csv(comp_csv, index=False)
    logger.info(f"Saved model comparison report to: {comp_csv}")
    
    return df_results

if __name__ == "__main__":
    run_ablation_experiments(epochs=1, batch_size=8)
