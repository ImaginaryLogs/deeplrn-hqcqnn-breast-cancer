from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from src.utils.yaml_config_reader import CONFIG

from src.qcqcnn.preprocessing import ResizeComp, load_dicom_pixels

@dataclass
class SplitIndices:
    train: list
    val: list
    test: list

def stratified_patient_split(metadata: pd.DataFrame,
                              train_frac: float = CONFIG.training.split.train_split,
                              val_frac: float = CONFIG.training.split.validation_split,
                              seed: int = CONFIG.seed) -> SplitIndices:
    patient_col: str = "patient_id"
    label_col: str = "pathology"

    # stratify by patients
    patient_labels = (
        metadata.groupby(patient_col)[label_col]
        .agg(lambda x: x.value_counts().idxmax())
        .reset_index()
    )
 
    # get train split
    train_p, temp_p = train_test_split(
        patient_labels[patient_col],
        train_size=train_frac,
        stratify=patient_labels[label_col],
        random_state=seed,
    )
    remaining = patient_labels[patient_labels[patient_col].isin(temp_p)]
    
    # get valiation and testing split
    validation_size = val_frac / (1 - train_frac)
    val_p, test_p = train_test_split(
        remaining[patient_col],
        train_size=validation_size,
        stratify=remaining[label_col],
        random_state=seed,
    )
 
    get_idx_for = lambda patients: metadata.index[metadata[patient_col].isin(patients)].tolist()
 
    return SplitIndices(
        train=get_idx_for(train_p),
        val=get_idx_for(val_p),
        test=get_idx_for(test_p),
    )

class CBISDDSMDataset(Dataset):
    LABEL_MAP = {"BENIGN": 0, "BENIGN_WITHOUT_CALLBACK": 0, "MALIGNANT": 1}

    def __init__(self, 
                 metadata: pd.DataFrame, 
                 indices: list,
                 condition: str = "naive", 
                 target_size: int = CONFIG.target_size,
                 cache_dir: str | None = None) -> None:
        super().__init__()
        self.metadata = metadata.loc[indices].reset_index(drop=True)
        self.target_size = target_size
        self.condition = condition
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.resize = ResizeComp()

        if self.cache_dir: self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self):
        """
        How big is the dataset
        """
        return len(self.metadata)
        
    def _cache_path(self, row_idx: int) -> Path | None:
        if not self.cache_dir:
            return None
        return self.cache_dir / f"{self.condition}_{row_idx}.npy"
    
    def __getitem__(self, idx: int):
        row = self.metadata.iloc[idx]
        cache_path = self._cache_path(idx)
 
        if cache_path and cache_path.exists():
            arr = np.load(cache_path)
        else:
            raw = load_dicom_pixels(row["dicom_path"])
            arr = self.resize.preprocess(raw, self.condition)
            if cache_path:
                np.save(cache_path, arr)
 
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]  # add channel dim -> (1, H, W)
 
        tensor = torch.from_numpy(arr).float()
        label = self.LABEL_MAP.get(str(row["pathology"]).upper(), None)
        if label is None:
            raise ValueError(f"Unrecognized pathology label: {row['pathology']}")
        return tensor, torch.tensor(label, dtype=torch.long)
 
