import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, MolFromSmiles, rdFingerprintGenerator as fp
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import deepchem as dc
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset


RANDOM_STATE = 616
BATCH_SIZE = 16


class CombinedDataset(Dataset):
    def __init__(self, graph_dataset, numeric_dataset):
        assert len(graph_dataset) == len(numeric_dataset), "Длины датасетов должны совпадать"
        self.graph_dataset = graph_dataset
        self.numeric_dataset = numeric_dataset

    def __len__(self):
        return len(self.graph_dataset)

    def __getitem__(self, idx):
        graph_data = self.graph_dataset[idx]
        numeric_data = self.numeric_dataset[idx]

        return graph_data, numeric_data


def smiles_to_descriptors_v4(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return {"BalabanJ": Descriptors.BalabanJ(mol),
            "qed": Descriptors.qed(mol),
            "SPS": Descriptors.SPS(mol),
            "fr_Al_OH": Descriptors.fr_Al_OH(mol),
            "fr_Ar_OH": Descriptors.fr_Ar_OH(mol),
            "fr_ether": Descriptors.fr_ether(mol),
            "fr_aldehyde": Descriptors.fr_aldehyde(mol),
            "fr_ketone": Descriptors.fr_ketone(mol),
            "fr_ester": Descriptors.fr_ester(mol),
            "fr_lactone": Descriptors.fr_lactone(mol),
            "fr_epoxide": Descriptors.fr_epoxide(mol),
            "fr_amide": Descriptors.fr_amide(mol),
            "fr_aniline": Descriptors.fr_aniline(mol),
            "fr_Imine": Descriptors.fr_Imine(mol),
            "fr_nitrile": Descriptors.fr_nitrile(mol),
            "fr_azide": Descriptors.fr_azide(mol),
            "fr_quatN": Descriptors.fr_quatN(mol),
            "fr_SH": Descriptors.fr_SH(mol),
            "fr_sulfide": Descriptors.fr_sulfide(mol),
            "fr_sulfonamd": Descriptors.fr_sulfonamd(mol),
            "fr_sulfone": Descriptors.fr_sulfone(mol),
            "fr_thiocyan": Descriptors.fr_thiocyan(mol),
            "fr_halogen": Descriptors.fr_halogen(mol),
            "fr_benzene": Descriptors.fr_benzene(mol),
            "fr_Ar_N": Descriptors.fr_Ar_N(mol),
            "fr_Ar_NH": Descriptors.fr_Ar_NH(mol),
            "fr_pyridine": Descriptors.fr_pyridine(mol),
            "fr_furan": Descriptors.fr_furan(mol),
            "fr_COO": Descriptors.fr_COO(mol),
            "fr_urea": Descriptors.fr_urea(mol),
            "fr_guanido": Descriptors.fr_guanido(mol),
            "fr_allylic_oxid": Descriptors.fr_allylic_oxid(mol),
            "fr_term_acetylene": Descriptors.fr_term_acetylene(mol),
            "fr_tetrazole": Descriptors.fr_tetrazole(mol),
            "NHOHCount": Descriptors.NHOHCount(mol),
            "NumRotatableBonds": Descriptors.NumRotatableBonds(mol),
            "NumAliphaticHeterocycles": Descriptors.NumAliphaticHeterocycles(mol),
            "RingCount": Descriptors.RingCount(mol),
            "MW": Descriptors.MolWt(mol),
            "TPSA": Descriptors.TPSA(mol)}


def calc_fingerprints(sm):
    """Генерация молекулярных отпечатков по методу Моргана"""

    morgan_fpgenerator = fp.GetMorganGenerator(radius=3, fpSize=2048)
    return morgan_fpgenerator.GetFingerprintAsNumPy(MolFromSmiles(sm))


def smiles2graph(sm):
    featurizer = dc.feat.DMPNNFeaturizer()
    graph = featurizer.featurize(sm)
    return graph


def get_features(df):
    df_desc = df['Smiles_cleaned'].apply(smiles_to_descriptors_v4).apply(pd.Series)
    df_fp = df['Smiles_cleaned'].apply(calc_fingerprints).apply(pd.Series)

    df_featurized = pd.concat([df[['Smiles_cleaned']], df_desc, df_fp], axis=1)
    df_graph = df['Smiles_cleaned'].apply(smiles2graph).apply(pd.Series)

    return df_desc, df_fp, df_graph


def to_scale(*data):
    '''Data must be (train, valid, test)'''

    cols2log = ['MW', 'SPS', 'NumRotatableBonds', 'NHOHCount', 'TPSA']

    for n, df in enumerate(data):
        for col in cols2log:
            if col in df.columns:
                df[col] = df[col].apply(np.log1p)

    scaler = MinMaxScaler()
    scaler.fit(data[0])
    data = list(map(scaler.transform, data))

    return data


def split_data(train_data, test_data, target='LogP'):
    combined_df = {'train': None, 'val': None, 'test': None}
    y = train_data[target]
    X = train_data.drop(target, axis=1)
    X_test = test_data
    y_test = torch.zeros(len(X_test))

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    X_train_desc, X_train_fp, X_train_graph = get_features(X_train)
    X_val_desc, X_val_fp, X_val_graph = get_features(X_val)
    X_test_desc, X_test_fp, X_test_graph = get_features(X_test)
    X_train_desc, X_val_desc, X_test_desc = to_scale(X_train_desc, X_val_desc, X_test_desc)
    



def get_loaders():


    X_train_desc, X_val_desc, X_test_desc = map(torch.Tensor, [X_train_desc, X_val_desc, X_test_desc])
    X_train_fp, X_val_fp, X_test_fp = map(torch.Tensor, map(np.array, [X_train_fp, X_val_fp, X_test_fp]))
    X_train_graph, X_val_graph, X_test_graph = map(dc.data.NumpyDataset, [X_train_graph, X_val_graph, X_test_graph])
    y_train, y_val, y_test = map(lambda x: torch.Tensor(x).unsqueeze(-1), map(np.array, [y_train, y_val, y_test]))

    train_tensor_dataset = TensorDataset(X_train_desc, X_train_fp, y_train)
    val_tensor_dataset = TensorDataset(X_val_desc, X_val_fp, y_val)
    test_tensor_dataset = TensorDataset(X_test_desc, X_test_fp, y_test)

    train_dataset = CombinedDataset(X_train_graph, train_tensor_dataset)
    val_dataset = CombinedDataset(X_val_graph, val_tensor_dataset)
    test_dataset = CombinedDataset(X_test_graph, test_tensor_dataset)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset)

    return train_loader, val_loader, test_loader


