from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from src.utils.yaml_reader import CONFIG

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
                 metadata: pd.DataFrame) -> None:
        super().__init__()
        
