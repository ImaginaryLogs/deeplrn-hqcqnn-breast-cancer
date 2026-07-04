import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.classification import BinaryAUROC, BinaryF1Score

from src.qcqcnn.quantum_filter import QuantumConvFilter
from src.qcqcnn.middle_layers import build_middle_layer
from src.qcqcnn.vqc_head import VQCHead


class QCQCNNLightningModule(pl.LightningModule):
    def __init__(
        self,
        middle_layer_name: str,
        in_channels: int = 1,      # Dictated by quantum filter channel output
        out_channels: int = 4,     # Dictated by VQC expected features
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        ansatz_reps: int = 2,
        pos_weight: float = 1.0,
    ):
        """Factorial experiment variant wrapper."""
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.weight_decay = weight_decay

        # Quantum Conv Filter (Frozen structure)
        self.quantum_filter = QuantumConvFilter()
        # Classical Parameter-Matched Middle Layer (e.g., baseline, residual, dense, inception)
        self.middle_layer = build_middle_layer(
            name=middle_layer_name, 
            in_channels=in_channels, 
            out_channels=out_channels
        )
        # VQC Head
        self.vqc_head = VQCHead(ansatz_reps=ansatz_reps)

        # Loss function with imbalance mitigation
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))


        self.train_auc = BinaryAUROC()
        self.val_auc = BinaryAUROC()
        self.test_auc = BinaryAUROC()

        self.train_f1 = BinaryF1Score()
        self.val_f1 = BinaryF1Score()
        self.test_f1 = BinaryF1Score()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Sequential routing matching architectural layout."""
        # x = self.quantum_filter(x)
        x = self.middle_layer(x)
        # x = self.vqc_head(x)
        return x.squeeze(-1)

    def _shared_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y.float())
        preds = torch.sigmoid(logits)
        return loss, preds, y

    def training_step(self, batch, batch_idx):
        loss, preds, y = self._shared_step(batch, batch_idx)
        self.train_auc(preds, y)
        self.train_f1(preds, y)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_auc", self.train_auc, on_step=False, on_epoch=True)
        self.log("train_f1", self.train_f1, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, y = self._shared_step(batch, batch_idx)
        self.val_auc(preds, y)
        self.val_f1(preds, y)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_auc", self.val_auc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_f1", self.val_f1, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        loss, preds, y = self._shared_step(batch, batch_idx)
        self.test_auc(preds, y)
        self.test_f1(preds, y)

        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_auc", self.test_auc, on_step=False, on_epoch=True)
        self.log("test_f1", self.test_f1, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay
        )