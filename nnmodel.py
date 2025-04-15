import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn as nn
import pytorch_lightning as pyl
from sklearn.preprocessing import FunctionTransformer, StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error as rmse
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, RegressorMixin
import deepchem as dc
from deepchem.models.torch_models.dmpnn import DMPNNModel, DMPNN
from deepchem.models.gbdt_models.gbdt_model import GBDTModel
from deepchem.data.data_loader import CSVLoader


BATCH_SIZE = 32
EPOCHS = 40
HIDDEN_SIZE = 256


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, y_pred, y_true):
        return torch.sqrt(self.mse(y_pred, y_true) + self.eps)


class HybridDMPNN(pyl.LightningModule):
    def __init__(self, fingerprint_size, numeric_features_size, ffn_hidden, dmpnn_model: DMPNN, hidden_size=HIDDEN_SIZE):
        super(HybridDMPNN, self).__init__()
        self.test_predictions = []
        self.targets = []

        self.fingerprint_fc = nn.Sequential(
            nn.Linear(fingerprint_size, hidden_size),
            # nn.LayerNorm(hidden_size),
            nn.LeakyReLU(),
            # nn.Dropout(0.2),
            # nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            # nn.LayerNorm(hidden_size // 2),
            nn.LeakyReLU()
        )

        self.numeric_fc = nn.Sequential(
            nn.Linear(numeric_features_size, hidden_size),
            # nn.LayerNorm(hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            # nn.LayerNorm(hidden_size),
            nn.LeakyReLU()
        )

        # self.dmpnn_enc = nn.Sequential(
        #     dmpnn_model.encoder(),
        #     nn.Linear(ffn_hidden, ffn_hidden),
        #     nn.LeakyReLU(),
        #     nn.Linear(ffn_hidden, ffn_hidden // 2),
        #     nn.LeakyReLU()
        # )

        self.dmpnn = dmpnn_model

        # self.numeric_cont_fc = nn.Sequential(
        #     nn.Linear(numeric_cont_features_size, hidden_size * 8),
        #     # nn.LayerNorm(hidden_size * 4),
        #     nn.LeakyReLU()
        # )

        # self.numeric_fc = nn.Sequential(
        #     ResidualBlock(numeric_features_size, hidden_size)
        # )

        self.combined_fc = nn.Sequential(
            nn.Linear(hidden_size * 3 // 2 + ffn_hidden, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, 1)
        )

        self.apply(self.initialize_weights)
        self.loss_fn = RMSELoss()

    def initialize_weights(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_normal_(layer.weight, a=0.1)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, fingerprint, numeric_features):
        fingerprint_out = self.fingerprint_fc(fingerprint)
        numeric_out = self.numeric_fc(numeric_features)
        dmpnn_out = self.dmpnn()
        # numeric_cont_out = self.numeric_cont_fc(numeric_cont_features)

        combined = torch.cat([fingerprint_out, numeric_out, dmpnn_out], dim=1)

        return self.combined_fc(combined)

    def training_step(self, batch, batch_idx):
        fingerprint, numeric_features, y = batch
        y_pred = self(fingerprint, numeric_features)

        loss = self.loss_fn(y_pred, y)
        self.log('Train RMSE', loss, on_step=False, on_epoch=True, prog_bar=True)

        r2 = r2_score(y.detach().cpu().numpy().reshape(-1), y_pred.detach().cpu().numpy().reshape(-1))
        self.log('Train R²', r2, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        fingerprint, numeric_features, y = batch
        y_pred = self(fingerprint, numeric_features)

        loss = self.loss_fn(y_pred, y)
        self.log('Validation RMSE', loss, on_step=False, on_epoch=True, prog_bar=True)

        r2 = r2_score(y.detach().cpu().numpy().reshape(-1), y_pred.detach().cpu().numpy().reshape(-1))
        self.log('Validation R²', r2, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        fingerprint, numeric_features, y = batch
        y_pred = self(fingerprint, numeric_features)

        loss = self.loss_fn(y_pred, y)
        self.test_predictions.extend(y_pred.detach().cpu().numpy())

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
