from torch.utils.data import TensorDataset, Dataset
from torch_geometric.data import Batch
import torch


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


def hybrid_collate(batch):
    graphs, fp_batch, desc_batch, y_batch = zip(*batch)
    pyg_patch = Batch()
    graph_batch = pyg_patch.from_data_list(graphs)

    return graph_batch, torch.stack(fp_batch), torch.stack(desc_batch), torch.stack(y_batch)
