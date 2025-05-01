import torch
import torch.nn as nn
import pytorch_lightning as pyl
from sklearn.metrics import r2_score

HIDDEN_SIZE = 256


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, y_pred, y_true):
        return torch.sqrt(self.mse(y_pred, y_true) + self.eps)


class HybridDMPNN(pyl.LightningModule):
    def __init__(self, fingerprint_size, numeric_features_size, dmpnn_model, num_hidden_size, comb_hidden_size,
                 hidden_size=HIDDEN_SIZE, use_numeric=True, use_fingerprint=True, use_dmpnn=True):
        super(HybridDMPNN, self).__init__()
        self.save_hyperparameters()
        self.test_predictions = []
        self.targets = []
        self.use_numeric = use_numeric
        self.use_fingerprint = use_fingerprint
        self.use_dmpnn = use_dmpnn
        self.hidden_size = hidden_size
        self.num_hidden_size = num_hidden_size

        self.fingerprint_fc = nn.Sequential(
            nn.Linear(fingerprint_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU()
        )

        self.numeric_fc = nn.Sequential(
            nn.Linear(numeric_features_size, num_hidden_size),
            nn.LeakyReLU(),
            nn.Linear(num_hidden_size, num_hidden_size // 2),
            nn.LeakyReLU()
        )

        self.dmpnn = dmpnn_model

        ffn_hidden = dmpnn_model.n_tasks
        self.ffn_hidden = ffn_hidden

        combined_in = num_hidden_size // 2 + hidden_size // 2 + ffn_hidden
        self.combined_fc = nn.Sequential(
            nn.Linear(combined_in, comb_hidden_size),
            nn.LayerNorm(comb_hidden_size),
            nn.LeakyReLU(),
            nn.Linear(comb_hidden_size, 1)
        )

        self.apply(self.initialize_weights)
        self.loss_fn = RMSELoss()

    def initialize_weights(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_normal_(layer.weight, a=0.1)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, fingerprint, numeric_features, graphs):
        parts = []

        if self.use_fingerprint:
            fp_out = self.fingerprint_fc(fingerprint)
        else:
            fp_out = torch.zeros(fingerprint.shape[0], self.hidden_size // 2, device=fingerprint.device)
        parts.append(fp_out)

        if self.use_numeric:
            num_out = self.numeric_fc(numeric_features)
        else:
            num_out = torch.zeros(numeric_features.shape[0], self.num_hidden_size // 2, device=numeric_features.device)
        parts.append(num_out)

        if self.use_dmpnn:
            dmpnn_out = self.dmpnn(graphs)
        else:
            dmpnn_out = torch.zeros(graphs.num_graphs, self.ffn_hidden)
        parts.append(dmpnn_out)

        combined = torch.cat(parts, dim=1)

        return self.combined_fc(combined)

    def training_step(self, batch, batch_idx):
        graph_feature, fingerprint, numeric_features, y = batch
        y_pred = self(fingerprint, numeric_features, graph_feature)

        batch_size = y.size(0)
        loss = self.loss_fn(y_pred, y)
        self.log('Train RMSE', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)

        r2 = r2_score(y.detach().cpu().numpy().reshape(-1), y_pred.detach().cpu().numpy().reshape(-1))
        self.log('Train R²', r2, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        graph_feature, fingerprint, numeric_features, y = batch
        y_pred = self(fingerprint, numeric_features, graph_feature)

        batch_size = y.size(0)
        loss = self.loss_fn(y_pred, y)
        self.log('Validation RMSE', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)

        r2 = r2_score(y.detach().cpu().numpy().reshape(-1), y_pred.detach().cpu().numpy().reshape(-1))
        self.log('Validation R²', r2, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)

        return loss

    def test_step(self, batch, batch_idx):
        graph_feature, fingerprint, numeric_features, y = batch
        y_pred = self(fingerprint, numeric_features, graph_feature)

        loss = self.loss_fn(y_pred, y)
        self.test_predictions.extend(y_pred.detach().cpu().numpy())

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


