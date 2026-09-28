"""
Explainability Module for VayuDrishti.
Implements:
  1. Grad-CAM Spatial Feature Attribution for Satellite Imagery
  2. Multimodal Modality Gating Contribution Breakdown (% Satellite, % ERA5, % Track)
Outputs:
  - reports/plots/gradcam_cyclone.png
  - reports/plots/modality_gating_weights.png
  - reports/explainability_report.json
"""
import os
import sys
import json
import logging
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import load_config, get_path, BASE_DIR
from dataset import get_dataloaders, IDX_TO_INTENSITY
from models import VayuDrishtiModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("explainability")

class LightweightGradCAM:
    """
    Lightweight feature activation map generator for cyclone vision backbones.
    """
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].clone()

        def forward_hook(module, input, output):
            self.activations = output.clone()

        # Hook into to_spatial[1]
        target_layer = self.model.sat_encoder.to_spatial[1]
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(self, sat_seq, era5_seq, track_seq, curr_pos, curr_int, target_class=None):
        self.model.eval()
        self.model.zero_grad()
        
        # Forward pass with gradient tracking on features
        preds = self.model(sat_seq, era5_seq, track_seq, curr_pos, curr_int)
        logits = preds["intensity_logits"]
        
        if target_class is None:
            target_class = torch.argmax(logits, dim=-1)
            
        score = logits[0, target_class]
        score.backward(retain_graph=True)
        
        # Compute CAM weights
        # activations: (B*T, 128*7*7), gradients: (B*T, 128*7*7)
        B, T = sat_seq.shape[:2]
        # Reshape to (B, T, 128, 7, 7)
        act = self.activations.view(B, T, 128, 7, 7)[0, -1] # Last timestep t
        grad = self.gradients.view(B, T, 128, 7, 7)[0, -1]
        
        weights = torch.mean(grad, dim=(1, 2), keepdim=True) # (128, 1, 1)
        cam = torch.sum(weights * act, dim=0).detach().cpu().numpy() # (7, 7)
        cam = np.maximum(cam, 0) # ReLU
        
        # Normalize to [0, 1]
        if np.max(cam) > 0:
            cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
        else:
            cam = np.zeros_like(cam)
            
        # Resize to 224x224
        import scipy.ndimage
        cam_resized = scipy.ndimage.zoom(cam, (224 / 7.0, 224 / 7.0), order=1)
        return cam_resized, preds

def generate_explainability_reports(config=None):
    if config is None:
        config = load_config()
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models_dir = Path(BASE_DIR) / "models"
    ckpt_path = models_dir / "best_multitask.pt"
    
    if not ckpt_path.exists():
        logger.error(f"Checkpoint not found at {ckpt_path}. Run training first.")
        return
        
    plots_dir = Path(BASE_DIR) / "reports" / "plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    # Load model
    logger.info(f"Loading trained model from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=device)
    model = VayuDrishtiModel(
        backbone_name=ckpt["config"]["backbone"],
        temporal_type=ckpt["config"]["temporal"],
        use_gating=ckpt["config"]["gating"]
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    
    # Load test batch
    _, _, test_loader = get_dataloaders(batch_size=8, config=config)
    sample_batch = next(iter(test_loader))
    
    sat_seq = sample_batch["satellite_sequence"][:1].to(device)
    era5_seq = sample_batch["era5_sequence"][:1].to(device)
    track_seq = sample_batch["track_sequence"][:1].to(device)
    curr_pos = sample_batch["current_position"][:1].to(device)
    curr_int = sample_batch["current_intensity"][:1].to(device)
    
    # 1. Grad-CAM Generation
    logger.info("Generating Grad-CAM spatial heatmap for cyclone eye/eyewall...")
    grad_cam = LightweightGradCAM(model)
    cam, preds = grad_cam.generate_heatmap(sat_seq, era5_seq, track_seq, curr_pos, curr_int)
    
    # Image input for visualization
    img_disp = sat_seq[0, -1].detach().cpu().numpy() # (3, 224, 224)
    img_disp = np.transpose(img_disp, (1, 2, 0))
    # Unnormalize
    img_disp = np.clip(img_disp * 0.224 + 0.456, 0.0, 1.0)
    
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(img_disp)
    plt.title("Satellite Reference (t)")
    plt.axis("off")
    
    plt.subplot(1, 3, 2)
    plt.imshow(cam, cmap="jet")
    plt.title("Grad-CAM Activation")
    plt.axis("off")
    
    plt.subplot(1, 3, 3)
    plt.imshow(img_disp)
    plt.imshow(cam, cmap="jet", alpha=0.5)
    plt.title("Overlay (Vortex Focus)")
    plt.axis("off")
    
    gradcam_plot_path = plots_dir / "gradcam_cyclone.png"
    plt.tight_layout()
    plt.savefig(gradcam_plot_path, dpi=150)
    plt.close()
    logger.info(f"Saved Grad-CAM heatmap to: {gradcam_plot_path}")
    
    # 2. Modality Gating Contribution Breakdown
    logger.info("Extracting learned modality weights across test samples...")
    modality_weights_list = []
    with torch.no_grad():
        for b in test_loader:
            s_b = b["satellite_sequence"].to(device)
            e_b = b["era5_sequence"].to(device)
            t_b = b["track_sequence"].to(device)
            c_p = b["current_position"].to(device)
            c_i = b["current_intensity"].to(device)
            
            p = model(s_b, e_b, t_b, c_p, c_i)
            if p["modality_weights"] is not None:
                modality_weights_list.append(p["modality_weights"].cpu().numpy())
                
    all_weights = np.concatenate(modality_weights_list, axis=0) # (N, 3)
    mean_weights = np.mean(all_weights, axis=0) * 100.0 # Percentage
    
    logger.info(f"Modality Contributions: Satellite={mean_weights[0]:.1f}%, ERA5={mean_weights[1]:.1f}%, Track={mean_weights[2]:.1f}%")
    
    # Bar plot of modality contributions
    plt.figure(figsize=(6, 4))
    modalities = ["Satellite (Imagery)", "ERA5 (Atmospheric)", "IBTrACS (Track)"]
    colors = ["#3498db", "#2ecc71", "#e67e22"]
    bars = plt.bar(modalities, mean_weights, color=colors, width=0.5)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1.0, f"{yval:.1f}%", ha='center', va='bottom', fontweight='bold')
    plt.ylabel("Learned Modality Contribution (%)")
    plt.title("VayuDrishti Gated Multimodal Fusion Breakdown")
    plt.ylim(0, 100)
    plt.tight_layout()
    
    gating_plot_path = plots_dir / "modality_gating_weights.png"
    plt.savefig(gating_plot_path, dpi=150)
    plt.close()
    logger.info(f"Saved modality gating weights plot to: {gating_plot_path}")
    
    # Save explainability report JSON
    rep_data = {
        "description": "Modality gating weights represent the relative weighting learned by the gating layer. They indicate multi-source attention but are not claimed as causal explanations.",
        "mean_modality_contributions_pct": {
            "satellite_imagery": round(float(mean_weights[0]), 2),
            "era5_atmospheric": round(float(mean_weights[1]), 2),
            "ibtracs_track_history": round(float(mean_weights[2]), 2)
        },
        "sample_gradcam_peak": round(float(np.max(cam)), 4),
        "visualizations": [
            "reports/plots/gradcam_cyclone.png",
            "reports/plots/modality_gating_weights.png"
        ]
    }
    exp_rep_path = Path(BASE_DIR) / "reports" / "explainability_report.json"
    with open(exp_rep_path, "w", encoding="utf-8") as f:
        json.dump(rep_data, f, indent=2)
    logger.info(f"Saved explainability report to: {exp_rep_path}")
    
    return rep_data

if __name__ == "__main__":
    generate_explainability_reports()
