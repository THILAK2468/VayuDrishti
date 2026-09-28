"""
Model Architecture Module for VayuDrishti.
Implements:
  1. Satellite Encoder: MobileViT-XS (and ConvNeXt-V2-Atto)
  2. Temporal Encoders: Lightweight ConvLSTM (1 & 2 layers), GRU, Temporal Attention
  3. ERA5 Atmospheric Encoder: Small CNN (128-d output)
  4. Track Encoder: Small MLP (64-d output)
  5. Gated Multimodal Fusion (and Concatenation Baseline)
  6. Multi-Task Heads: Detection, Intensity Classification, Track Forecast, Intensity Forecast
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# ---------------------------------------------------------------------
# 1. Temporal ConvLSTM Components
# ---------------------------------------------------------------------
class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size=3):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True
        )

    def forward(self, x, state):
        h, c = state
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.split(gates, self.hidden_channels, dim=1)
        
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, spatial_size, device):
        h, w = spatial_size
        return (
            torch.zeros(batch_size, self.hidden_channels, h, w, device=device),
            torch.zeros(batch_size, self.hidden_channels, h, w, device=device)
        )

class ConvLSTM(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=1, kernel_size=3):
        super().__init__()
        self.num_layers = num_layers
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            c_in = in_channels if i == 0 else hidden_channels
            self.cells.append(ConvLSTMCell(c_in, hidden_channels, kernel_size))

    def forward(self, x_seq):
        # x_seq: (B, T, C, H, W)
        B, T, C, H, W = x_seq.shape
        device = x_seq.device
        
        states = [cell.init_hidden(B, (H, W), device) for cell in self.cells]
        outputs = []
        
        for t in range(T):
            x_t = x_seq[:, t]
            for layer_idx, cell in enumerate(self.cells):
                h, c = cell(x_t, states[layer_idx])
                states[layer_idx] = (h, c)
                x_t = h
            outputs.append(h)
            
        # Last hidden state
        last_h = states[-1][0] # (B, hidden_channels, H, W)
        return last_h

class TemporalAttention(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.Tanh(),
            nn.Linear(in_dim // 2, 1)
        )

    def forward(self, seq):
        # seq: (B, T, D)
        weights = F.softmax(self.attn(seq), dim=1) # (B, T, 1)
        weighted = torch.sum(weights * seq, dim=1)  # (B, D)
        return weighted, weights

# ---------------------------------------------------------------------
# 2. Satellite Encoder (MobileViT-XS / ConvNeXt-V2-Atto)
# ---------------------------------------------------------------------
class SatelliteEncoder(nn.Module):
    def __init__(self, backbone_name="mobilevit_xs", out_dim=256, temporal_type="convlstm", num_lstm_layers=1):
        super().__init__()
        self.backbone_name = backbone_name
        self.temporal_type = temporal_type
        
        # Create backbone
        self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0)
        feat_dim = self.backbone.num_features
        
        # Spatial conv layer to adapt to (C_lstm, 7, 7)
        self.to_spatial = nn.Sequential(
            nn.Linear(feat_dim, 128 * 7 * 7),
            nn.ReLU(inplace=False)
        )
        
        if temporal_type == "convlstm":
            self.temporal = ConvLSTM(in_channels=128, hidden_channels=128, num_layers=num_lstm_layers)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.proj = nn.Linear(128, out_dim)
        elif temporal_type == "gru":
            self.temporal = nn.GRU(input_size=feat_dim, hidden_size=out_dim, batch_first=True)
            self.proj = nn.Identity()
        elif temporal_type == "convlstm_attn":
            self.temporal = ConvLSTM(in_channels=128, hidden_channels=128, num_layers=num_lstm_layers)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.attn = TemporalAttention(128)
            self.proj = nn.Linear(128, out_dim)
        else: # static / single-step
            self.proj = nn.Linear(feat_dim, out_dim)

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_stages(self, num_stages=1):
        # Unfreeze head and final stages
        for p in self.backbone.parameters():
            p.requires_grad = False
        # Unfreeze last parameters
        params = list(self.backbone.parameters())
        unfreeze_count = min(len(params), num_stages * 15)
        for p in params[-unfreeze_count:]:
            p.requires_grad = True

    def unfreeze_all(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

    def forward(self, x_seq):
        # x_seq: (B, T, 3, 224, 224)
        B, T, C, H, W = x_seq.shape
        x_flat = x_seq.view(B * T, C, H, W)
        
        feats = self.backbone(x_flat) # (B*T, feat_dim)
        
        if self.temporal_type in ["convlstm", "convlstm_attn"]:
            spatial = self.to_spatial(feats).view(B, T, 128, 7, 7)
            last_h = self.temporal(spatial) # (B, 128, 7, 7)
            pooled = self.pool(last_h).view(B, 128)
            out = self.proj(pooled) # (B, out_dim)
        elif self.temporal_type == "gru":
            feats_seq = feats.view(B, T, -1)
            _, h_n = self.temporal(feats_seq)
            out = self.proj(h_n.squeeze(0)) # (B, out_dim)
        else:
            # Aggregate mean
            feats_seq = feats.view(B, T, -1).mean(dim=1)
            out = self.proj(feats_seq)
            
        return out

# ---------------------------------------------------------------------
# 3. ERA5 Environmental Atmospheric Encoder
# ---------------------------------------------------------------------
class ERA5Encoder(nn.Module):
    def __init__(self, in_channels=8, out_dim=128):
        super().__init__()
        # Input: (B, T, 8, 40, 40)
        self.conv_net = nn.Sequential(
            nn.Conv2d(in_channels * 5, 32, kernel_size=3, padding=1), # Concat T=5 channels
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # (B, 32, 20, 20)
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # (B, 64, 10, 10)
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)) # (B, 128, 1, 1)
        )
        self.fc = nn.Linear(128, out_dim)

    def forward(self, era5_seq):
        # era5_seq: (B, T, 8, 40, 40)
        B, T, C, H, W = era5_seq.shape
        x = era5_seq.view(B, T * C, H, W)
        feat = self.conv_net(x).view(B, -1)
        out = self.fc(feat)
        return out

# ---------------------------------------------------------------------
# 4. IBTrACS / Track Encoder
# ---------------------------------------------------------------------
class TrackEncoder(nn.Module):
    def __init__(self, in_features=6, seq_len=5, out_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features * seq_len, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, out_dim),
            nn.LayerNorm(out_dim)
        )

    def forward(self, track_seq):
        # track_seq: (B, T, 6)
        B = track_seq.shape[0]
        x = track_seq.view(B, -1)
        return self.mlp(x)

# ---------------------------------------------------------------------
# 5. Gated Multimodal Fusion
# ---------------------------------------------------------------------
class GatedMultimodalFusion(nn.Module):
    def __init__(self, sat_dim=256, era5_dim=128, track_dim=64, out_dim=256, use_gating=True):
        super().__init__()
        self.use_gating = use_gating
        total_dim = sat_dim + era5_dim + track_dim
        
        if use_gating:
            self.gate = nn.Sequential(
                nn.Linear(total_dim, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 3),
                nn.Softmax(dim=-1)
            )
            
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim)
        )

    def forward(self, sat_feat, era5_feat, track_feat):
        # sat_feat: (B, 256), era5_feat: (B, 128), track_feat: (B, 64)
        concat = torch.cat([sat_feat, era5_feat, track_feat], dim=-1) # (B, 448)
        
        if self.use_gating:
            weights = self.gate(concat) # (B, 3) -> [w_sat, w_era5, w_track]
            w_sat = weights[:, 0:1]
            w_era5 = weights[:, 1:2]
            w_track = weights[:, 2:3]
            
            modulated = torch.cat([
                w_sat * sat_feat,
                w_era5 * era5_feat,
                w_track * track_feat
            ], dim=-1)
            shared = self.fusion(modulated)
            return shared, weights
        else:
            shared = self.fusion(concat)
            # Uniform dummy weights for baseline
            dummy_weights = torch.ones(sat_feat.shape[0], 3, device=sat_feat.device) / 3.0
            return shared, dummy_weights

# ---------------------------------------------------------------------
# 6. Multi-Task VayuDrishti Core Network
# ---------------------------------------------------------------------
class VayuDrishtiModel(nn.Module):
    def __init__(
        self,
        backbone_name="mobilevit_xs",
        temporal_type="convlstm",
        num_lstm_layers=1,
        use_gating=True,
        num_classes=8,
        num_horizons=3,
        use_sat=True,
        use_era5=True,
        use_track=True
    ):
        super().__init__()
        self.use_sat = use_sat
        self.use_era5 = use_era5
        self.use_track = use_track
        
        # Encoders
        self.sat_encoder = SatelliteEncoder(
            backbone_name=backbone_name,
            out_dim=256,
            temporal_type=temporal_type,
            num_lstm_layers=num_lstm_layers
        ) if use_sat else None
        
        self.era5_encoder = ERA5Encoder(in_channels=8, out_dim=128) if use_era5 else None
        self.track_encoder = TrackEncoder(in_features=6, seq_len=5, out_dim=64) if use_track else None
        
        # Dimensions
        sat_d = 256 if use_sat else 0
        era5_d = 128 if use_era5 else 0
        track_d = 64 if use_track else 0
        
        # Multimodal fusion
        if use_sat and use_era5 and use_track:
            self.fusion = GatedMultimodalFusion(sat_dim=256, era5_dim=128, track_dim=64, out_dim=256, use_gating=use_gating)
        else:
            # Single or partial modality projection
            in_d = sat_d + era5_d + track_d
            self.fusion = nn.Sequential(
                nn.Linear(in_d, 256),
                nn.LayerNorm(256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 256)
            )

        # Multi-task output heads
        self.detection_head = nn.Linear(256, 1)                                   # Binary classification (BCE)
        self.intensity_head = nn.Linear(256, num_classes)                         # Intensity class logits
        self.track_head = nn.Linear(256, num_horizons * 2)                       # Future lat/lon (+1, +3, +6)
        self.intensity_forecast_head = nn.Linear(256, num_horizons * 2)          # Future wind/pres (+1, +3, +6)

    def forward(self, sat_seq, era5_seq, track_seq, curr_pos=None, curr_int=None):
        B = sat_seq.shape[0] if sat_seq is not None else era5_seq.shape[0]
        device = sat_seq.device if sat_seq is not None else era5_seq.device
        
        # 1. Encode modalities
        feats = []
        if self.use_sat:
            s_f = self.sat_encoder(sat_seq)
            feats.append(s_f)
        if self.use_era5:
            e_f = self.era5_encoder(era5_seq)
            feats.append(e_f)
        if self.use_track:
            t_f = self.track_encoder(track_seq)
            feats.append(t_f)
            
        # 2. Fusion
        modality_weights = None
        if self.use_sat and self.use_era5 and self.use_track:
            shared, modality_weights = self.fusion(s_f, e_f, t_f)
        else:
            cat = torch.cat(feats, dim=-1)
            shared = self.fusion(cat)
            modality_weights = torch.ones(B, 3, device=device) / 3.0
            
        # 3. Heads
        det_logit = self.detection_head(shared).squeeze(-1)                      # (B,)
        intensity_logits = self.intensity_head(shared)                           # (B, num_classes)
        track_disp = self.track_head(shared).view(B, 3, 2)                       # (B, 3, 2)
        int_disp = self.intensity_forecast_head(shared).view(B, 3, 2)            # (B, 3, 2)
        
        # Physical displacement residual from current reference point
        if curr_pos is not None:
            track_pred = curr_pos.unsqueeze(1) + track_disp
        else:
            track_pred = track_disp
            
        if curr_int is not None:
            int_pred = curr_int.unsqueeze(1) + int_disp
        else:
            int_pred = int_disp
        
        return {
            "detection_logit": det_logit,
            "detection_prob": torch.sigmoid(det_logit),
            "intensity_logits": intensity_logits,
            "future_track": track_pred,
            "future_intensity": int_pred,
            "modality_weights": modality_weights,
            "shared_representation": shared
        }

if __name__ == "__main__":
    model = VayuDrishtiModel()
    s_in = torch.randn(2, 5, 3, 224, 224)
    e_in = torch.randn(2, 5, 8, 40, 40)
    t_in = torch.randn(2, 5, 6)
    out = model(s_in, e_in, t_in)
    print("VayuDrishti model forward pass successful:")
    print("Detection Logit:", out["detection_logit"].shape)
    print("Intensity Logits:", out["intensity_logits"].shape)
    print("Future Track:   ", out["future_track"].shape)
    print("Future Intensity:", out["future_intensity"].shape)
    print("Modality Weights:", out["modality_weights"].shape, out["modality_weights"][0].tolist())
