import torch
from pytorch_lightning.loggers import TensorBoardLogger
from nnmodel import HybridDMPNN
from deepchem.models.torch_models.dmpnn import DMPNN
import pytorch_lightning as pyl
from preprocessing_data import get_data
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

BATCH_SIZE = 16
EPOCHS = 50


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df_file = 'df_mean_logP.csv'
    data = pd.read_csv(df_file)
    comb_df = get_data(data, is_test)
    fingerprint_size = dfs['fp'][0].shape[1]
    numeric_features_size = dfs['desc'][0].shape[1]

    train_loader, valid_loader, test_loader = get_loaders(dfs)
    model = HybridDMPNN(fingerprint_size=fingerprint_size,
                        numeric_features_size=numeric_features_size,
                        dmpnn_model=DMPNN(n_tasks=300, ffn_layers=2),
                        hidden_size=4096,
                        num_hidden_size=512,
                        comb_hidden_size=256).to(device)

    logger = TensorBoardLogger("lightning_logs", name="model_test_2")
    trainer = pyl.Trainer(max_epochs=EPOCHS, logger=logger)
    trainer.fit(model, train_loader, valid_loader)

    trainer.save_checkpoint("best_model.ckpt")


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
