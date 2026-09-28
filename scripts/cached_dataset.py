"""
High-Speed Cached PyTorch Dataset for CPU-First Training.
Loads pre-processed .npy tensors directly from disk:
  - processed/era5_features/<sample_id>.npy
  - processed/satellite_features/<sample_id>_<img_size>.npy
Eliminates NetCDF decoding and image resizing overhead on CPU.
"""
import os
import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import BASE_DIR, load_config, get_path
from dataset import INTENSITY_TO_IDX

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("cached_dataset")

class CachedCycloneSequenceDataset(Dataset):
    def __init__(self, split_csv_path, img_size=160, seq_len=5, is_train=False, config=None):
        super().__init__()
        self.split_csv_path = Path(split_csv_path)
        self.img_size = img_size
        self.seq_len = seq_len
        self.is_train = is_train
        
        self.era5_dir = Path(BASE_DIR) / "processed" / "era5_features"
        self.sat_dir = Path(BASE_DIR) / "processed" / "satellite_features"
        
        df_split = pd.read_csv(self.split_csv_path)
        df_split["datetime"] = pd.to_datetime(df_split["datetime"], utc=True)
        df_split = df_split.sort_values(by=["storm_name", "datetime"]).reset_index(drop=True)
        
        # Build causal sequence windows (t-(seq_len-1) ... t)
        self.samples = []
        grouped = df_split.groupby("storm_name")
        
        for sname, group in grouped:
            group = group.sort_values("datetime").reset_index(drop=True)
            n_obs = len(group)
            if n_obs < self.seq_len:
                continue
                
            for i in range(self.seq_len - 1, n_obs):
                seq_rows = group.iloc[i - (self.seq_len - 1) : i + 1]
                curr_row = group.iloc[i]
                
                # Multi-horizon forecast targets (+1, +3, +6 steps)
                target_offsets = [1, 3, 6]
                target_tracks = []
                target_intensities = []
                target_masks = []
                
                for off in target_offsets:
                    fut_idx = i + off
                    if fut_idx < n_obs:
                        f_row = group.iloc[fut_idx]
                        f_lat = float(f_row["latitude"])
                        f_lon = float(f_row["longitude"])
                        f_w = float(f_row["wind_speed"]) if pd.notna(f_row["wind_speed"]) else float(curr_row["wind_speed"])
                        f_p = float(f_row["pressure"]) if pd.notna(f_row["pressure"]) else float(curr_row["pressure"])
                        target_tracks.append([f_lat, f_lon])
                        target_intensities.append([f_w, f_p])
                        target_masks.append(1.0)
                    else:
                        target_tracks.append([0.0, 0.0])
                        target_intensities.append([0.0, 0.0])
                        target_masks.append(0.0)
                        
                cls_str = curr_row.get("intensity_class", "UNKNOWN")
                cls_idx = INTENSITY_TO_IDX.get(cls_str, 0)
                
                self.samples.append({
                    "sample_id": curr_row["sample_id"],
                    "storm_name": sname,
                    "datetime": curr_row["datetime"].isoformat(),
                    "seq_rows": seq_rows,
                    "curr_row": curr_row,
                    "intensity_class": cls_idx,
                    "detection_label": 1.0,
                    "target_tracks": np.array(target_tracks, dtype=np.float32),
                    "target_intensities": np.array(target_intensities, dtype=np.float32),
                    "target_masks": np.array(target_masks, dtype=np.float32)
                })
                
        logger.info(f"Loaded {self.split_csv_path.name}: {len(self.samples)} valid sequence samples (T={self.seq_len}, img_size={self.img_size}).")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        seq_rows = item["seq_rows"]
        
        era5_seq = []
        sat_seq = []
        track_seq = []
        
        prev_lat, prev_lon = None, None
        prev_wind = None
        
        for _, row in seq_rows.iterrows():
            sid = row["sample_id"]
            lat = float(row["latitude"])
            lon = float(row["longitude"])
            wind = float(row["wind_speed"]) if pd.notna(row["wind_speed"]) else 30.0
            pres = float(row["pressure"]) if pd.notna(row["pressure"]) else 1000.0
            
            # Displacements & intensity changes
            disp = np.sqrt((lat - prev_lat)**2 + (lon - prev_lon)**2) if prev_lat is not None else 0.0
            d_wind = (wind - prev_wind) if prev_wind is not None else 0.0
            
            prev_lat, prev_lon = lat, lon
            prev_wind = wind
            
            # Normalize track kinematics
            norm_lat = (lat - 15.0) / 10.0
            norm_lon = (lon - 75.0) / 15.0
            norm_w = (wind - 50.0) / 30.0
            norm_p = (pres - 1000.0) / 20.0
            norm_disp = disp / 2.0
            norm_dwind = d_wind / 10.0
            track_seq.append([norm_lat, norm_lon, norm_w, norm_p, norm_disp, norm_dwind])
            
            # Fast disk load for ERA5 patch
            era5_file = self.era5_dir / f"{sid}.npy"
            if era5_file.exists():
                era5_p = np.load(era5_file)
            else:
                era5_p = np.zeros((8, 40, 40), dtype=np.float32)
            era5_seq.append(era5_p)
            
            # Fast disk load for Satellite tensor
            sat_file = self.sat_dir / f"{sid}_{self.img_size}.npy"
            if sat_file.exists():
                sat_t = np.load(sat_file)
            else:
                sat_t = np.zeros((3, self.img_size, self.img_size), dtype=np.float32)
            sat_seq.append(sat_t)
            
        era5_tensor = torch.tensor(np.stack(era5_seq, axis=0), dtype=torch.float32)     # (T, 8, 40, 40)
        sat_tensor = torch.tensor(np.stack(sat_seq, axis=0), dtype=torch.float32)       # (T, 3, img_size, img_size)
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

def get_cached_dataloaders(splits_dir=None, batch_size=8, img_size=160, seq_len=5, num_workers=0):
    if splits_dir is None:
        config = load_config()
        splits_dir = get_path("splits_dir", config)
        
    train_csv = os.path.join(splits_dir, "train.csv")
    val_csv = os.path.join(splits_dir, "validation.csv")
    test_csv = os.path.join(splits_dir, "test.csv")
    
    train_ds = CachedCycloneSequenceDataset(train_csv, img_size=img_size, seq_len=seq_len, is_train=True)
    val_ds = CachedCycloneSequenceDataset(val_csv, img_size=img_size, seq_len=seq_len, is_train=False)
    test_ds = CachedCycloneSequenceDataset(test_csv, img_size=img_size, seq_len=seq_len, is_train=False)
    
    # CPU-friendly: pin_memory=False, small num_workers
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=num_workers, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=False)
    
    return train_loader, val_loader, test_loader
