import pytorch_lightning as pl
import torch
import pandas as pd
from typing import Optional
from torch.utils.data import DataLoader, WeightedRandomSampler
from src.utils.yaml_config_reader import CONFIG
from src.qcqcnn.dataset import CBISDDSMDataset, stratified_patient_split


class CBISDDSMDataModule(pl.LightningDataModule):
    def __init__(
        self,
        metadata_path: str,
        preprocessing_condition: str = "naive",
        batch_size: int = CONFIG.training.batch_size,
        num_workers: int = 4,
        cache_dir: str | None = None,
    ):
        super().__init__()
        self.metadata_path = metadata_path
        self.preprocessing_condition = preprocessing_condition
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cache_dir = cache_dir

        # Placeholders to hold metadata and splits
        self.metadata_df: Optional[pd.DataFrame] = None
        self.train_dataset: Optional[CBISDDSMDataset] = None
        self.val_dataset: Optional[CBISDDSMDataset] = None
        self.test_dataset: Optional[CBISDDSMDataset] = None
        self.pos_weight = 1.0

    def prepare_data(self):
        """Heavy I/O: Read metadata once from disk before setup."""
        self.metadata_df = pd.read_csv(self.metadata_path)

    def setup(self, stage: str | None = None):
        """Assembles dataset splits using stratified patient partition indices."""
        # Get train/val/test index mappings using the script's logic        
        if self.metadata_df is None or self.train_dataset is None or self.val_dataset is None or self.test_dataset is None:
            raise ValueError(f"Data not instantiated - Metadata:{self.metadata_df is None}, Train:{self.train_dataset is None}, Val:{self.val_dataset is None}, Test:{self.test_dataset is None}")
            

        splits = stratified_patient_split(self.metadata_df)

        if stage == "fit" or stage is None:
            self.train_dataset = CBISDDSMDataset(
                metadata=self.metadata_df,
                indices=splits.train,
                condition=self.preprocessing_condition,
                cache_dir=self.cache_dir,
            )
            self.val_dataset = CBISDDSMDataset(
                metadata=self.metadata_df,
                indices=splits.val,
                condition=self.preprocessing_condition,
                cache_dir=self.cache_dir,
            )

            # Compute inverse class frequencies dynamically on the train split only
            train_labels = []
            for idx in splits.train:
                row = self.metadata_df.iloc[idx]
                label = CBISDDSMDataset.LABEL_MAP.get(str(row["pathology"]).upper())
                train_labels.append(label)
            
            labels_tensor = torch.tensor(train_labels, dtype=torch.long)
            neg_count = (labels_tensor == 0).sum().item()
            pos_count = (labels_tensor == 1).sum().item()
            
            # Guard against division by zero if dataset is missing a class
            self.pos_weight = float(neg_count) / max(pos_count, 1)

        if stage == "test" or stage is None:
            self.test_dataset = CBISDDSMDataset(
                metadata=self.metadata_df,
                indices=splits.test,
                condition=self.preprocessing_condition,
                cache_dir=self.cache_dir,
            )

    def train_dataloader(self) -> DataLoader:
        """Returns the training data loader optimized via a WeightedRandomSampler."""
        if self.train_dataset is None:
            raise ValueError("Train dataset is empty")
        
        
        targets = torch.tensor([
            CBISDDSMDataset.LABEL_MAP.get(str(row["pathology"]).upper())
            for _, row in self.train_dataset.metadata.iterrows()
        ], dtype=torch.long)
        
        class_sample_count = torch.tensor([(targets == 0).sum(), (targets == 1).sum()])
        weight = 1.0 / class_sample_count.float()
        samples_weight = torch.tensor([weight[t] for t in targets]).tolist()

        sampler = WeightedRandomSampler(
            weights=samples_weight, 
            num_samples=len(samples_weight), 
            replacement=True
        )

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise ValueError("Validation dataset is empty")
        
        
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise ValueError("Test dataset is empty")
        
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )