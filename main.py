from torch_geometric.data import Batch
import torch
from torch.utils.data import TensorDataset, Dataset
from torch_geometric.loader import DataLoader
from preprocessing_data import split_data
import pandas as pd
from pytorch_lightning.loggers import TensorBoardLogger
from nnmodel import HybridDMPNN
from deepchem.models.torch_models.dmpnn import DMPNN
import pytorch_lightning as pyl

import warnings
warnings.filterwarnings("ignore")

BATCH_SIZE = 16
EPOCHS = 50


class CombinedDataset(Dataset):
    def __init__(self, graph_dataset, numeric_datasets: TensorDataset):
        assert len(graph_dataset) == len(numeric_datasets), "The lengths of the datasets must match"
        self.graph_dataset = graph_dataset
        self.numeric_datasets = numeric_datasets.tensors

    def __len__(self):
        return len(self.graph_dataset)

    def __getitem__(self, idx):
        graph_data = self.graph_dataset[idx]
        desc_data = self.numeric_datasets[0][idx]
        fp_data = self.numeric_datasets[1][idx]
        y_data = self.numeric_datasets[2][idx]

        return graph_data, fp_data, desc_data, y_data


def load_data():
    df_filtered = pd.read_csv('filtered_df2.csv')
    df_test = pd.read_csv('test_data80_cleaned.csv')
    dfs = split_data(df_filtered, df_test)

    return dfs


def hybrid_collate(batch):
    graphs, fp_batch, desc_batch, y_batch = zip(*batch)
    pyg_patch = Batch()
    graph_batch = pyg_patch.from_data_list(graphs)

    return graph_batch, torch.stack(fp_batch), torch.stack(desc_batch), torch.stack(y_batch)


def get_loaders(combined_df):
    datasets = dict()
    for i, name in enumerate(['train', 'val', 'test']):
        tensor_dataset = TensorDataset(combined_df['desc'][i], combined_df['fp'][i], combined_df['y'][i])
        datasets[name] = CombinedDataset(combined_df['graph'][i], tensor_dataset)

    train_loader = DataLoader(datasets['train'],
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              collate_fn=hybrid_collate,
                              num_workers=4,
                              persistent_workers=True)
    val_loader = DataLoader(datasets['val'],
                            batch_size=BATCH_SIZE,
                            shuffle=False,
                            collate_fn=hybrid_collate,
                            num_workers=4,
                            persistent_workers=True)
    test_loader = DataLoader(datasets['test'],
                             collate_fn=hybrid_collate)

    return train_loader, val_loader, test_loader


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dfs = load_data()
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
