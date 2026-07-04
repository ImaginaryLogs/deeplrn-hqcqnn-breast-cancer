import argparse
import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from src.qcqcnn.datamodule import CBISDDSMDataModule
from src.qcqcnn.lightning_module import QCQCNNLightningModule

def run_cell(preprocessing, middle_layer, hparams):
    datamodule = CBISDDSMDataModule(
        metadata_path="data/metadata/mass_case_description.csv", 
        preprocessing_condition=preprocessing,
        batch_size=hparams['training']['batch_size'],
        cache_dir="data/cache/"
    )
    
    # Setup to calculate pos_weight
    datamodule.setup(stage="fit")
    
    # 2. LightningModule
    model = QCQCNNLightningModule(
        middle_layer_name=middle_layer,
        lr=hparams['training']['learning_rate'],
        weight_decay=hparams['training']['weight_decay'],
        ansatz_reps=hparams['training']['ansatz_reps'],
        pos_weight=datamodule.pos_weight
    )
    
    # 3. Trainer
    trainer = pl.Trainer(
        max_epochs=50,
        callbacks=[
            EarlyStopping(monitor="val_auc", mode="max", patience=5),
            ModelCheckpoint(dirpath="outputs/checkpoints/", filename=f"{preprocessing}_{middle_layer}")
        ],
        accelerator="auto",
        devices=1
    )
    
    trainer.fit(model, datamodule=datamodule)
    return trainer.callback_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocessing", type=str, required=True)
    parser.add_argument("--middle_layer", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        hparams = yaml.safe_load(f)
        
    run_cell(args.preprocessing, args.middle_layer, hparams)
