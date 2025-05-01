from torch_geometric.loader import DataLoader
from torch.utils.data import TensorDataset
from datasets.combined_dataset import CombinedDataset, hybrid_collate

BATCH_SIZE = 128


def get_loaders(comb_df, is_test=False, batch_size=BATCH_SIZE):
    datasets = dict()

    for name in ['train', 'val', 'test']:
        if comb_df[name] is not None:
            tensor_dataset = TensorDataset(comb_df[name]['desc'], comb_df[name]['fp'], comb_df[name]['y'])
            datasets[name] = CombinedDataset(comb_df[name]['graph'], tensor_dataset)

    if not is_test:
        train_loader = DataLoader(datasets['train'],
                                  batch_size=batch_size,
                                  shuffle=True,
                                  collate_fn=hybrid_collate,
                                  num_workers=4,
                                  persistent_workers=True)
        val_loader = DataLoader(datasets['val'],
                                batch_size=batch_size,
                                shuffle=False,
                                collate_fn=hybrid_collate,
                                num_workers=4,
                                persistent_workers=True)
        return train_loader, val_loader
    return DataLoader(datasets['test'], collate_fn=hybrid_collate)
