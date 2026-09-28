"""
Loss Functions and Geographic Metrics Module for VayuDrishti.
Implements:
  1. Multi-Task Weighted Loss (Detection, Intensity Classification, Track, Intensity Forecast)
  2. Great-Circle Haversine Distance Error (km)
  3. Continuous Forecast Metrics (MAE, RMSE, R2)
  4. Classification Metrics (Accuracy, Macro/Weighted F1, AUROC, Confusion Matrix)
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, balanced_accuracy_score, r2_score
)

def haversine_distance_np(lat1, lon1, lat2, lon2):
    """
    Computes great-circle distance between coordinates in kilometers.
    """
    R = 6371.0 # Earth mean radius in km
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat / 2.0)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0)**2
    c = 2.0 * np.arcsin(np.clip(np.sqrt(a), 0.0, 1.0))
    return R * c

class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss for mitigating extreme class imbalance:
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha  # class weights tensor (C,)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss

class MultiTaskLoss(nn.Module):
    def __init__(
        self,
        lambda_det=1.0,
        lambda_cls=1.0,
        lambda_track=1.0,
        lambda_int=1.0,
        class_weights=None,
        cls_loss_type="ce",
        focal_gamma=2.0
    ):
        super().__init__()
        self.lambda_det = lambda_det
        self.lambda_cls = lambda_cls
        self.lambda_track = lambda_track
        self.lambda_int = lambda_int
        self.cls_loss_type = cls_loss_type
        
        self.bce_loss = nn.BCEWithLogitsLoss()
        
        if cls_loss_type == "focal":
            self.cls_loss_fn = FocalLoss(alpha=class_weights, gamma=focal_gamma)
        else:
            if class_weights is not None:
                self.cls_loss_fn = nn.CrossEntropyLoss(weight=class_weights)
            else:
                self.cls_loss_fn = nn.CrossEntropyLoss()
            
        self.huber_track = nn.SmoothL1Loss(reduction="none")
        self.huber_int = nn.SmoothL1Loss(reduction="none")

    def forward(self, preds, batch):
        # 1. Detection loss
        det_targets = batch["detection_label"].to(preds["detection_logit"].device)
        l_det = self.bce_loss(preds["detection_logit"], det_targets)
        
        # 2. Intensity classification loss
        cls_targets = batch["intensity_class"].to(preds["intensity_logits"].device)
        l_cls = self.cls_loss_fn(preds["intensity_logits"], cls_targets)
        
        # 3. Track forecast loss (Huber on normalized lat/lon or raw lat/lon)
        # Target shape: (B, 3, 2), Pred shape: (B, 3, 2)
        # Masks shape: (B, 3)
        track_targets = batch["target_tracks"].to(preds["future_track"].device)
        track_masks = batch["target_masks"].to(preds["future_track"].device).unsqueeze(-1) # (B, 3, 1)
        
        # Scale down coordinates roughly by 10 for balanced gradient
        diff_track = self.huber_track(preds["future_track"] / 10.0, track_targets / 10.0)
        l_track = (diff_track * track_masks).sum() / (track_masks.sum() * 2.0 + 1e-6)
        
        # 4. Intensity forecast loss (Wind/Pressure)
        int_targets = batch["target_intensities"].to(preds["future_intensity"].device)
        # Scale: wind / 50, pressure / 1000
        int_scale = torch.tensor([50.0, 1000.0], device=preds["future_intensity"].device).view(1, 1, 2)
        diff_int = self.huber_int(preds["future_intensity"] / int_scale, int_targets / int_scale)
        l_int = (diff_int * track_masks).sum() / (track_masks.sum() * 2.0 + 1e-6)
        
        # Total weighted loss
        l_total = (
            self.lambda_det * l_det +
            self.lambda_cls * l_cls +
            self.lambda_track * l_track +
            self.lambda_int * l_int
        )
        
        return {
            "total_loss": l_total,
            "loss_det": l_det.item(),
            "loss_cls": l_cls.item(),
            "loss_track": l_track.item(),
            "loss_int": l_int.item()
        }

def compute_metrics(all_preds, all_targets):
    """
    Computes complete suite of scientific metrics:
      - Classification: Accuracy, Macro F1, Weighted F1, Precision, Recall, Confusion Matrix
      - Track: Lat MAE, Lon MAE, Haversine Error in km (+1, +3, +6)
      - Intensity: Wind MAE, Wind RMSE, Pressure MAE, Pressure RMSE, R2
    """
    # 1. Classification metrics
    cls_preds = np.argmax(all_preds["intensity_logits"], axis=-1)
    cls_trues = all_targets["intensity_class"]
    
    acc = accuracy_score(cls_trues, cls_preds)
    bal_acc = balanced_accuracy_score(cls_trues, cls_preds)
    macro_f1 = f1_score(cls_trues, cls_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(cls_trues, cls_preds, average="weighted", zero_division=0)
    prec = precision_score(cls_trues, cls_preds, average="weighted", zero_division=0)
    rec = recall_score(cls_trues, cls_preds, average="weighted", zero_division=0)
    conf_mat = confusion_matrix(cls_trues, cls_preds)
    
    # 2. Track prediction metrics (per horizon +1, +3, +6)
    horizons = ["+1", "+3", "+6"]
    track_preds = all_preds["future_track"]       # (N, 3, 2)
    track_trues = all_targets["target_tracks"]     # (N, 3, 2)
    masks = all_targets["target_masks"]           # (N, 3)
    
    track_metrics = {}
    for h_idx, h_name in enumerate(horizons):
        valid_mask = masks[:, h_idx] > 0.5
        if np.sum(valid_mask) > 0:
            pred_h = track_preds[valid_mask, h_idx]
            true_h = track_trues[valid_mask, h_idx]
            
            lat_mae = float(np.mean(np.abs(pred_h[:, 0] - true_h[:, 0])))
            lon_mae = float(np.mean(np.abs(pred_h[:, 1] - true_h[:, 1])))
            
            # Great-circle distance
            dists = haversine_distance_np(
                true_h[:, 0], true_h[:, 1],
                pred_h[:, 0], pred_h[:, 1]
            )
            dist_mean_km = float(np.mean(dists))
            dist_median_km = float(np.median(dists))
            
            track_metrics[f"{h_name}_lat_mae"] = round(lat_mae, 3)
            track_metrics[f"{h_name}_lon_mae"] = round(lon_mae, 3)
            track_metrics[f"{h_name}_haversine_mean_km"] = round(dist_mean_km, 2)
            track_metrics[f"{h_name}_haversine_median_km"] = round(dist_median_km, 2)
        else:
            track_metrics[f"{h_name}_haversine_mean_km"] = 0.0
            
    # 3. Intensity forecast metrics (Wind, Pressure per horizon)
    int_preds = all_preds["future_intensity"]     # (N, 3, 2)
    int_trues = all_targets["target_intensities"] # (N, 3, 2)
    
    intensity_metrics = {}
    for h_idx, h_name in enumerate(horizons):
        valid_mask = masks[:, h_idx] > 0.5
        if np.sum(valid_mask) > 0:
            w_pred = int_preds[valid_mask, h_idx, 0]
            w_true = int_trues[valid_mask, h_idx, 0]
            p_pred = int_preds[valid_mask, h_idx, 1]
            p_true = int_trues[valid_mask, h_idx, 1]
            
            w_mae = float(np.mean(np.abs(w_pred - w_true)))
            w_rmse = float(np.sqrt(np.mean((w_pred - w_true)**2)))
            w_r2 = float(r2_score(w_true, w_pred)) if len(w_true) > 2 else 0.0
            
            p_mae = float(np.mean(np.abs(p_pred - p_true)))
            p_rmse = float(np.sqrt(np.mean((p_pred - p_true)**2)))
            p_r2 = float(r2_score(p_true, p_pred)) if len(p_true) > 2 else 0.0
            
            intensity_metrics[f"{h_name}_wind_mae_kt"] = round(w_mae, 2)
            intensity_metrics[f"{h_name}_wind_rmse_kt"] = round(w_rmse, 2)
            intensity_metrics[f"{h_name}_wind_r2"] = round(w_r2, 3)
            intensity_metrics[f"{h_name}_pres_mae_hpa"] = round(p_mae, 2)
            intensity_metrics[f"{h_name}_pres_rmse_hpa"] = round(p_rmse, 2)
            intensity_metrics[f"{h_name}_pres_r2"] = round(p_r2, 3)
            
    overall = {
        "accuracy": round(float(acc), 4),
        "balanced_accuracy": round(float(bal_acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "track_metrics": track_metrics,
        "intensity_metrics": intensity_metrics,
        "confusion_matrix": conf_mat.tolist()
    }
    return overall
