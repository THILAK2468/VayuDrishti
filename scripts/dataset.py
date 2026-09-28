"""
PyTorch Dataset and Sequence Construction Module for VayuDrishti.
Constructs causal temporal sequences (t-4 to t) and multi-horizon forecast targets (+1, +3, +6).
Supports multi-modal input (Satellite, ERA5, IBTrACS track).
"""
import os
import sys
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import netCDF4 as nc
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import load_config, get_path, BASE_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("dataset")

# Standard IMD 8-tier intensity class to integer mapping
INTENSITY_TO_IDX = {
    "SUPER": 7,
    "EXTREME": 6,
    "VERY": 5,
    "SEVERE": 4,
    "NORMAL": 3,
    "DEEP_DEPRESSION": 2,
    "DEPRESSION": 1,
    "LOW": 0,
    "UNKNOWN": 0
}
IDX_TO_INTENSITY = {v: k for k, v in INTENSITY_TO_IDX.items()}

# Core ERA5 variable ordering
ERA5_VARS = ["u10", "v10", "t2m", "msl", "sst", "wind_speed_10m", "wind_direction_10m", "msl_hpa"]

class VayuDrishtiSequenceDataset(Dataset):
    """
    Constructs temporal sequence samples of length T (default 5: t-4 ... t)
    and forecast targets for horizons +1, +3, +6.
    """
    def __init__(self, split_csv_path, norm_params_path=None, seq_len=5, img_size=224, is_train=False, config=None):
        super().__init__()
        self.seq_len = seq_len
        self.img_size = img_size
        self.is_train = is_train
        
        if config is None:
            config = load_config()
        self.config = config
        
        # Load normalization parameters (strictly from train)
        if norm_params_path is None:
            norm_params_path = get_path("normalization_params", config)
            
        with open(norm_params_path, "r", encoding="utf-8") as f:
            self.norm_data = json.load(f)["features"]
            
        # Load split manifest
        self.df = pd.read_csv(split_csv_path)
        self.df["datetime_dt"] = pd.to_datetime(self.df["datetime"], utc=True)
        
        # Build sequences per storm
        self.samples = self._build_sequences()
        logger.info(f"Loaded split {os.path.basename(split_csv_path)}: {len(self.samples)} valid sequence samples.")

    def _build_sequences(self):
        samples = []
        # Group by storm
        for storm_name, group in self.df.groupby("storm_name"):
            group = group.sort_values(by="datetime_dt").reset_index(drop=True)
            n_obs = len(group)
            if n_obs < self.seq_len:
                continue
                
            for i in range(self.seq_len - 1, n_obs):
                # Historical sequence: [i - seq_len + 1, ..., i]
                seq_rows = group.iloc[i - self.seq_len + 1 : i + 1]
                
                # Current reference observation
                curr_row = group.iloc[i]
                
                # Targets for horizons +1, +3, +6 (1 step, 3 steps, 6 steps ahead)
                # In IBTrACS 3-hourly/hourly tracks:
                horizons = [1, 3, 6]
                target_tracks = []
                target_intensities = []
                target_masks = []
                
                for h in horizons:
                    target_idx = i + h
                    if target_idx < n_obs:
                        tgt_row = group.iloc[target_idx]
                        tgt_lat = float(tgt_row["latitude"])
                        tgt_lon = float(tgt_row["longitude"])
                        tgt_w = float(tgt_row["wind_speed"]) if pd.notna(tgt_row["wind_speed"]) else float(curr_row["wind_speed"])
                        tgt_p = float(tgt_row["pressure"]) if pd.notna(tgt_row["pressure"]) else float(curr_row["pressure"])
                        target_tracks.append([tgt_lat, tgt_lon])
                        target_intensities.append([tgt_w, tgt_p])
                        target_masks.append(1.0)
                    else:
                        # Extrapolate flat or mask out
                        target_tracks.append([float(curr_row["latitude"]), float(curr_row["longitude"])])
                        target_intensities.append([float(curr_row["wind_speed"]), float(curr_row["pressure"])])
                        target_masks.append(0.0) # Masked in loss
                        
                # Intensity class integer
                cls_str = curr_row["intensity_class"]
                cls_idx = INTENSITY_TO_IDX.get(cls_str, 0)
                
                # Detection label: 1 = cyclone
                det_label = 1.0
                
                samples.append({
                    "sample_id": curr_row["sample_id"],
                    "storm_name": storm_name,
                    "storm_id": curr_row["storm_id"],
                    "datetime": curr_row["datetime"],
                    "seq_rows": seq_rows,
                    "curr_row": curr_row,
                    "intensity_class": cls_idx,
                    "detection_label": det_label,
                    "target_tracks": np.array(target_tracks, dtype=np.float32),
                    "target_intensities": np.array(target_intensities, dtype=np.float32),
                    "target_masks": np.array(target_masks, dtype=np.float32)
                })
                
        return samples

    def __len__(self):
        return len(self.samples)

    def _load_era5_patch(self, era5_rel_path):
        """Loads and normalizes an ERA5 patch netcdf file."""
        if not era5_rel_path or pd.isna(era5_rel_path):
            return np.zeros((len(ERA5_VARS), 40, 40), dtype=np.float32)
            
        full_p = os.path.join(BASE_DIR, era5_rel_path)
        if not os.path.exists(full_p):
            return np.zeros((len(ERA5_VARS), 40, 40), dtype=np.float32)
            
        try:
            with nc.Dataset(full_p, "r") as ds:
                channels = []
                for v in ERA5_VARS:
                    if v in ds.variables:
                        arr = ds.variables[v][:]
                        if np.ma.is_masked(arr):
                            arr = arr.filled(np.nan)
                        # Replace NaNs (e.g. SST over land) with mean
                        mean_v = self.norm_data[v]["mean"]
                        std_v = self.norm_data[v]["std"]
                        arr = np.nan_to_num(arr, nan=mean_v)
                        # Normalize
                        normed = (arr - mean_v) / std_v
                        # Ensure 40x40 shape
                        if normed.shape != (40, 40):
                            # Interpolate or pad/crop to 40x40
                            pad_h = max(0, 40 - normed.shape[0])
                            pad_w = max(0, 40 - normed.shape[1])
                            normed = np.pad(normed, ((0, pad_h), (0, pad_w)), mode="edge")[:40, :40]
                        channels.append(normed)
                    else:
                        channels.append(np.zeros((40, 40), dtype=np.float32))
                return np.stack(channels, axis=0).astype(np.float32)
        except Exception:
            return np.zeros((len(ERA5_VARS), 40, 40), dtype=np.float32)

    def _load_satellite_img(self, sat_rel_path, era5_normed_patch=None):
        """
        Loads and normalizes satellite image (3, 224, 224).
        If satellite file is absent, constructs a scientifically valid multi-channel proxy
        from the centered environmental wind/thermal fields to maintain multimodal compatibility.
        """
        if sat_rel_path and pd.notna(sat_rel_path) and sat_rel_path != "":
            full_p = os.path.join(BASE_DIR, sat_rel_path)
            if os.path.exists(full_p):
                try:
                    with Image.open(full_p) as img:
                        img = img.convert("RGB").resize((self.img_size, self.img_size))
                        arr = np.array(img, dtype=np.float32) / 255.0
                        # Standard ImageNet / IR normalization
                        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                        normed = (arr - mean) / std
                        return np.transpose(normed, (2, 0, 1)).astype(np.float32), 1.0 # 1.0 = valid image
                except Exception:
                    pass
                    
        # Scientific fallback proxy: synthesize 3-channel environmental vorticity/thermal representation
        # Channels: (0: wind speed, 1: t2m / infrared proxy, 2: pressure gradient / msl)
        if era5_normed_patch is not None:
            # era5_normed_patch is (8, 40, 40)
            # Resize from 40x40 to 224x224 using PIL
            ch0 = Image.fromarray(era5_normed_patch[5]).resize((self.img_size, self.img_size), Image.Resampling.BILINEAR) # wind_speed
            ch1 = Image.fromarray(era5_normed_patch[2]).resize((self.img_size, self.img_size), Image.Resampling.BILINEAR) # t2m
            ch2 = Image.fromarray(era5_normed_patch[7]).resize((self.img_size, self.img_size), Image.Resampling.BILINEAR) # msl_hpa
            synth = np.stack([np.array(ch0), np.array(ch1), np.array(ch2)], axis=0).astype(np.float32)
            return synth, 0.0 # 0.0 = synthetic proxy
            
        return np.zeros((3, self.img_size, self.img_size), dtype=np.float32), 0.0

    def __getitem__(self, idx):
        item = self.samples[idx]
        seq_rows = item["seq_rows"]
        
        era5_seq = []
        sat_seq = []
        sat_masks = []
        track_seq = []
        
        prev_lat, prev_lon = None, None
        prev_wind = None
        
        for _, row in seq_rows.iterrows():
            lat = float(row["latitude"])
            lon = float(row["longitude"])
            wind = float(row["wind_speed"]) if pd.notna(row["wind_speed"]) else 30.0
            pres = float(row["pressure"]) if pd.notna(row["pressure"]) else 1000.0
            
            # Displacements & intensity changes
            disp = np.sqrt((lat - prev_lat)**2 + (lon - prev_lon)**2) if prev_lat is not None else 0.0
            d_wind = (wind - prev_wind) if prev_wind is not None else 0.0
            
            prev_lat, prev_lon = lat, lon
            prev_wind = wind
            
            # Normalize track features
            norm_lat = (lat - 15.0) / 10.0
            norm_lon = (lon - 75.0) / 15.0
            norm_w = (wind - 50.0) / 30.0
            norm_p = (pres - 1000.0) / 20.0
            norm_disp = disp / 2.0
            norm_dwind = d_wind / 10.0
            
            track_feat = [norm_lat, norm_lon, norm_w, norm_p, norm_disp, norm_dwind]
            track_seq.append(track_feat)
            
            # Load ERA5 patch
            era5_p = self._load_era5_patch(row.get("era5_path", ""))
            era5_seq.append(era5_p)
            
            # Load satellite image
            sat_img, sat_m = self._load_satellite_img(row.get("satellite_path", ""), era5_p)
            sat_seq.append(sat_img)
            sat_masks.append(sat_m)
            
        era5_tensor = torch.tensor(np.stack(era5_seq, axis=0), dtype=torch.float32)     # (T, 8, 40, 40)
        sat_tensor = torch.tensor(np.stack(sat_seq, axis=0), dtype=torch.float32)       # (T, 3, 224, 224)
        track_tensor = torch.tensor(np.array(track_seq), dtype=torch.float32)           # (T, 6)
        
        curr_row = item["curr_row"]
        curr_pos = torch.tensor([float(curr_row["latitude"]), float(curr_row["longitude"])], dtype=torch.float32)
        curr_int = torch.tensor([float(curr_row["wind_speed"]), float(curr_row["pressure"])], dtype=torch.float32)
        
        return {
            "sample_id": item["sample_id"],
            "storm_name": item["storm_name"],
            "datetime": item["datetime"],
            "satellite_sequence": sat_tensor,
            "era5_sequence": era5_tensor,
            "track_sequence": track_tensor,
            "current_position": curr_pos,
            "current_intensity": curr_int,
            "intensity_class": torch.tensor(item["intensity_class"], dtype=torch.long),
            "detection_label": torch.tensor(item["detection_label"], dtype=torch.float32),
            "target_tracks": torch.tensor(item["target_tracks"], dtype=torch.float32),               # (3, 2)
            "target_intensities": torch.tensor(item["target_intensities"], dtype=torch.float32),     # (3, 2)
            "target_masks": torch.tensor(item["target_masks"], dtype=torch.float32)                  # (3,)
        }

def get_dataloaders(splits_dir=None, batch_size=8, config=None):
    if config is None:
        config = load_config()
    if splits_dir is None:
        splits_dir = get_path("splits_dir", config)
        
    train_csv = os.path.join(splits_dir, "train.csv")
    val_csv = os.path.join(splits_dir, "validation.csv")
    test_csv = os.path.join(splits_dir, "test.csv")
    
    train_ds = VayuDrishtiSequenceDataset(train_csv, is_train=True, config=config)
    val_ds = VayuDrishtiSequenceDataset(val_csv, is_train=False, config=config)
    test_ds = VayuDrishtiSequenceDataset(test_csv, is_train=False, config=config)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    train_l, val_l, test_l = get_dataloaders(batch_size=4)
    batch = next(iter(train_l))
    print("Batch loaded successfully:")
    print("Satellite Seq:", batch["satellite_sequence"].shape)
    print("ERA5 Seq:     ", batch["era5_sequence"].shape)
    print("Track Seq:    ", batch["track_sequence"].shape)
    print("Targets Track:", batch["target_tracks"].shape)
    print("Targets Int:  ", batch["target_intensities"].shape)
    print("Classes:      ", batch["intensity_class"])
