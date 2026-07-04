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
    ...

class CBISDDSMDataset(Dataset):
    LABEL_MAP = {"BENIGN": 0, "BENIGN_WITHOUT_CALLBACK": 0, "MALIGNANT": 1}

    def __init__(self, 
                 metadata: pd.DataFrame) -> None:
        super().__init__()
        
