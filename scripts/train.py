"""
Training Pipeline for VayuDrishti.
Implements:
  - Mixed precision training (GPU) with CPU fallback
  - Gradient clipping, early stopping, LR scheduler
  - Progressive fine-tuning (Phases 1, 2, 3)
  - Multi-task checkpointing (best_detection.pt, best_classification.pt, best_forecasting.pt, best_multitask.pt)
  - Experiment logging with config, metrics, training logs, and loss plots
"""
import os
import sys
import time
import json
import logging
import datetime
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import load_config, get_path, BASE_DIR
from dataset import get_dataloaders, INTENSITY_TO_IDX
from models import VayuDrishtiModel
from losses import MultiTaskLoss, compute_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("train")

def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None, max_grad_norm=1.0):
    model.train()
    running_loss = 0.0
    running_det = 0.0
    running_cls = 0.0
    running_track = 0.0
    running_int = 0.0
    
    for batch in dataloader:
        optimizer.zero_grad()
        
        sat_seq = batch["satellite_sequence"].to(device)
        era5_seq = batch["era5_sequence"].to(device)
        track_seq = batch["track_sequence"].to(device)
        
        curr_pos = batch["current_position"].to(device)
        curr_int = batch["current_intensity"].to(device)
        
        if scaler and device.type == "cuda":
            with torch.cuda.amp.autocast():
                preds = model(sat_seq, era5_seq, track_seq, curr_pos=curr_pos, curr_int=curr_int)
                loss_dict = criterion(preds, batch)
                loss = loss_dict["total_loss"]
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            preds = model(sat_seq, era5_seq, track_seq, curr_pos=curr_pos, curr_int=curr_int)
            loss_dict = criterion(preds, batch)
            loss = loss_dict["total_loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            
        bs = sat_seq.shape[0]
        running_loss += loss.item() * bs
        running_det += loss_dict["loss_det"] * bs
        running_cls += loss_dict["loss_cls"] * bs
        running_track += loss_dict["loss_track"] * bs
        running_int += loss_dict["loss_int"] * bs
        
    n = len(dataloader.dataset)
    return {
        "loss": running_loss / n,
        "loss_det": running_det / n,
        "loss_cls": running_cls / n,
        "loss_track": running_track / n,
        "loss_int": running_int / n
    }

@torch.no_grad()
def evaluate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_det = 0.0
    running_cls = 0.0
    running_track = 0.0
    running_int = 0.0
    
    all_preds = {
        "detection_prob": [],
        "intensity_logits": [],
        "future_track": [],
        "future_intensity": [],
        "modality_weights": []
    }
    all_targets = {
        "detection_label": [],
        "intensity_class": [],
        "target_tracks": [],
        "target_intensities": [],
        "target_masks": []
    }
    
    for batch in dataloader:
        sat_seq = batch["satellite_sequence"].to(device)
        era5_seq = batch["era5_sequence"].to(device)
        track_seq = batch["track_sequence"].to(device)
        
        curr_pos = batch["current_position"].to(device)
        curr_int = batch["current_intensity"].to(device)
        
        preds = model(sat_seq, era5_seq, track_seq, curr_pos=curr_pos, curr_int=curr_int)
        loss_dict = criterion(preds, batch)
        
        bs = sat_seq.shape[0]
        running_loss += loss_dict["total_loss"].item() * bs
        running_det += loss_dict["loss_det"] * bs
        running_cls += loss_dict["loss_cls"] * bs
        running_track += loss_dict["loss_track"] * bs
        running_int += loss_dict["loss_int"] * bs
        
        all_preds["detection_prob"].append(preds["detection_prob"].cpu().numpy())
        all_preds["intensity_logits"].append(preds["intensity_logits"].cpu().numpy())
        all_preds["future_track"].append(preds["future_track"].cpu().numpy())
        all_preds["future_intensity"].append(preds["future_intensity"].cpu().numpy())
        if preds["modality_weights"] is not None:
            all_preds["modality_weights"].append(preds["modality_weights"].cpu().numpy())
            
        all_targets["detection_label"].append(batch["detection_label"].numpy())
        all_targets["intensity_class"].append(batch["intensity_class"].numpy())
        all_targets["target_tracks"].append(batch["target_tracks"].numpy())
        all_targets["target_intensities"].append(batch["target_intensities"].numpy())
        all_targets["target_masks"].append(batch["target_masks"].numpy())
        
    n = len(dataloader.dataset)
    aggregated_preds = {k: np.concatenate(v, axis=0) for k, v in all_preds.items() if v}
    aggregated_targets = {k: np.concatenate(v, axis=0) for k, v in all_targets.items() if v}
    
    metrics = compute_metrics(aggregated_preds, aggregated_targets)
    metrics["val_loss"] = running_loss / n
    metrics["val_loss_det"] = running_det / n
    metrics["val_loss_cls"] = running_cls / n
    metrics["val_loss_track"] = running_track / n
    metrics["val_loss_int"] = running_int / n
    
    return metrics, aggregated_preds, aggregated_targets

def run_training(
    epochs=15,
    batch_size=8,
    lr=1e-4,
    weight_decay=1e-4,
    patience=8,
    backbone_name="mobilevit_xs",
    temporal_type="convlstm",
    num_lstm_layers=1,
    use_gating=True,
    exp_name=None,
    config=None
):
    if config is None:
        config = load_config()
        
    # Experiment directory setup
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_tag = exp_name if exp_name else f"exp_{timestamp}"
    exp_dir = Path(BASE_DIR) / "experiments" / exp_tag
    plots_dir = exp_dir / "plots"
    os.makedirs(plots_dir, exist_ok=True)
    models_dir = Path(BASE_DIR) / "models"
    os.makedirs(models_dir, exist_ok=True)
    
    # Setup experiment file logger
    fh = logging.FileHandler(exp_dir / "training.log")
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device} | Experiment: {exp_tag}")
    
    # 1. Dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=batch_size, config=config)
    logger.info(f"Datasets initialized: Train={len(train_loader.dataset)}, Val={len(val_loader.dataset)}, Test={len(test_loader.dataset)}")
    
    # 2. Compute class weights for intensity imbalance
    all_train_classes = [s["intensity_class"] for s in train_loader.dataset.samples]
    cls_counts = np.bincount(all_train_classes, minlength=8)
    cls_weights = np.zeros(8, dtype=np.float32)
    for c in range(8):
        cls_weights[c] = (len(all_train_classes) / (8.0 * max(1, cls_counts[c])))
    cls_weights_tensor = torch.tensor(cls_weights, device=device)
    
    # 3. Model setup
    model = VayuDrishtiModel(
        backbone_name=backbone_name,
        temporal_type=temporal_type,
        num_lstm_layers=num_lstm_layers,
        use_gating=use_gating,
        num_classes=8
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {backbone_name} + {temporal_type} | Total Params: {total_params:,} | Trainable: {trainable_params:,}")
    
    # Save model summary
    with open(exp_dir / "model_summary.txt", "w") as f:
        f.write(f"VayuDrishti Architecture Summary\n")
        f.write(f"Backbone: {backbone_name}\n")
        f.write(f"Temporal: {temporal_type} (Layers: {num_lstm_layers})\n")
        f.write(f"Fusion: {'Gated' if use_gating else 'Concat'}\n")
        f.write(f"Total Parameters: {total_params:,}\n")
        f.write(f"Trainable Parameters: {trainable_params:,}\n\n")
        f.write(str(model))
        
    criterion = MultiTaskLoss(
        lambda_det=1.0,
        lambda_cls=1.0,
        lambda_track=1.0,
        lambda_int=1.0,
        class_weights=cls_weights_tensor
    )
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None
    
    # Progressive fine-tuning schedule
    # Phase 1: Freeze backbone for first 3 epochs
    logger.info(">>> PHASE 1: Training temporal, fusion, and head layers (Backbone frozen)...")
    if model.sat_encoder is not None:
        model.sat_encoder.freeze_backbone()
        
    best_loss = float("inf")
    best_f1 = 0.0
    best_dist = float("inf")
    patience_counter = 0
    
    history = {
        "train_loss": [], "val_loss": [],
        "val_acc": [], "val_macro_f1": [],
        "val_h1_dist_km": [], "val_h1_wind_mae": []
    }
    
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        ep_start = time.time()
        
        # Progressive phase switching
        if epoch == 4:
            logger.info(">>> PHASE 2: Unfreezing top backbone stages with reduced learning rate...")
            if model.sat_encoder is not None:
                model.sat_encoder.unfreeze_stages(num_stages=2)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr * 0.5
        elif epoch == 8:
            logger.info(">>> PHASE 3: Unfreezing full backbone for end-to-end fine-tuning...")
            if model.sat_encoder is not None:
                model.sat_encoder.unfreeze_all()
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr * 0.2
                
        train_res = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_res, val_preds, val_targets = evaluate_epoch(model, val_loader, criterion, device)
        scheduler.step(val_res["val_loss"])
        
        ep_duration = time.time() - ep_start
        
        h1_dist = val_res["track_metrics"].get("+1_haversine_mean_km", 0.0)
        h1_wind = val_res["intensity_metrics"].get("+1_wind_mae_kt", 0.0)
        
        history["train_loss"].append(train_res["loss"])
        history["val_loss"].append(val_res["val_loss"])
        history["val_acc"].append(val_res["accuracy"])
        history["val_macro_f1"].append(val_res["macro_f1"])
        history["val_h1_dist_km"].append(h1_dist)
        history["val_h1_wind_mae"].append(h1_wind)
        
        logger.info(
            f"Epoch {epoch:02d}/{epochs:02d} [{ep_duration:.1f}s] | "
            f"Train Loss: {train_res['loss']:.4f} | "
            f"Val Loss: {val_res['val_loss']:.4f} | "
            f"Acc: {val_res['accuracy']*100:.1f}% | "
            f"Macro F1: {val_res['macro_f1']:.3f} | "
            f"+1h Track: {h1_dist:.1f}km | "
            f"+1h Wind MAE: {h1_wind:.1f}kt"
        )
        
        # Checkpointing
        checkpoint_dict = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "val_metrics": val_res,
            "config": {
                "backbone": backbone_name,
                "temporal": temporal_type,
                "gating": use_gating,
                "lr": lr
            },
            "random_seed": 42
        }
        
        # Best Multitask Checkpoint
        if val_res["val_loss"] < best_loss:
            best_loss = val_res["val_loss"]
            patience_counter = 0
            torch.save(checkpoint_dict, models_dir / "best_multitask.pt")
            torch.save(checkpoint_dict, exp_dir / "best_multitask.pt")
            logger.info(f"[*] New best multitask model saved (Val Loss: {best_loss:.4f})")
        else:
            patience_counter += 1
            
        # Best Classification Checkpoint
        if val_res["macro_f1"] > best_f1:
            best_f1 = val_res["macro_f1"]
            torch.save(checkpoint_dict, models_dir / "best_classification.pt")
            
        # Best Forecasting Checkpoint
        if h1_dist < best_dist and h1_dist > 0.0:
            best_dist = h1_dist
            torch.save(checkpoint_dict, models_dir / "best_forecasting.pt")
            
        # Early stopping check
        if patience_counter >= patience:
            logger.info(f"Early stopping triggered at epoch {epoch} (patience={patience})")
            break
            
    total_train_time = time.time() - start_time
    logger.info(f"Training completed in {total_train_time/60:.2f} minutes.")
    
    # Save training curves plot
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.title("Multi-Task Loss")
    plt.xlabel("Epoch")
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(history["val_acc"], label="Accuracy")
    plt.plot(history["val_macro_f1"], label="Macro F1")
    plt.title("Classification Metrics")
    plt.xlabel("Epoch")
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(history["val_h1_dist_km"], label="+1h Track Err (km)", color="coral")
    plt.title("+1h Haversine Error (km)")
    plt.xlabel("Epoch")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(plots_dir / "training_curves.png", dpi=150)
    plt.close()
    
    # Final evaluation on Test Set
    logger.info(">>> Evaluating best multitask model on held-out TEST set...")
    best_ckpt = torch.load(models_dir / "best_multitask.pt", map_location=device)
    model.load_state_dict(best_ckpt["model_state"])
    test_metrics, test_preds, test_targets = evaluate_epoch(model, test_loader, criterion, device)
    
    # Save metrics JSON
    final_results = {
        "experiment_name": exp_tag,
        "backbone": backbone_name,
        "temporal": temporal_type,
        "gating": use_gating,
        "total_params": total_params,
        "train_duration_sec": round(total_train_time, 2),
        "val_metrics": best_ckpt["val_metrics"],
        "test_metrics": test_metrics
    }
    with open(exp_dir / "metrics.json", "w") as f:
        json.dump(final_results, f, indent=2)
        
    logger.info("=== Final Held-Out Test Evaluation ===")
    logger.info(f"Accuracy:         {test_metrics['accuracy']*100:.2f}%")
    logger.info(f"Macro F1:         {test_metrics['macro_f1']:.4f}")
    logger.info(f"Weighted F1:      {test_metrics['weighted_f1']:.4f}")
    for h in ["+1", "+3", "+6"]:
        logger.info(f"Track {h} Error:   {test_metrics['track_metrics'].get(f'{h}_haversine_mean_km', 'N/A')} km")
        logger.info(f"Wind  {h} MAE:     {test_metrics['intensity_metrics'].get(f'{h}_wind_mae_kt', 'N/A')} kt")
        
    return model, final_results

if __name__ == "__main__":
    run_training(epochs=5, batch_size=8, lr=1e-4)
