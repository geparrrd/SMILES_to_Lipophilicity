import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, MolFromSmiles, MACCSkeys, LayeredFingerprint
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator, GetAtomPairGenerator, GetTopologicalTorsionGenerator
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from deepchem.feat import DMPNNFeaturizer
import torch
from torch_geometric.data import Data
from deepchem.models.torch_models.dmpnn import _MapperDMPNN
import warnings
warnings.filterwarnings("ignore")

RANDOM_STATE = 616


def smiles_to_descriptors(smiles):
    '''To get descriptors'''

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
    """Get fingerprints"""

    mol = MolFromSmiles(sm)

    fp_size = 2048
    morgan_fpgenerator = GetMorganGenerator(radius=3, fpSize=2 * fp_size)
    morgan_np = morgan_fpgenerator.GetCountFingerprintAsNumPy(mol)

    maccs = MACCSkeys.GenMACCSKeys(mol)
    maccs_np = np.array(maccs)

    # Atom Pair
    ap_gen = GetAtomPairGenerator(fpSize=fp_size)
    atom_pair_np = ap_gen.GetCountFingerprintAsNumPy(mol)

    # Layered
    # layered = LayeredFingerprint(mol, fpSize=2 * fp_size)
    # layered_np = np.array(layered)

    # Topological Torsion
    # topo_gen = GetTopologicalTorsionGenerator(fpSize=2 * fp_size)
    # torsion_np = topo_gen.GetCountFingerprintAsNumPy(mol)

    combined_fp = np.concatenate([
        morgan_np, maccs_np, atom_pair_np
    ])

    return combined_fp


def smiles2graph(sm):
    '''Get graph features'''

    featurizer = DMPNNFeaturizer()
    graph = featurizer.featurize(sm)
    return graph


def get_features(df):
    '''Get all features'''

    feature = 'Smiles_cleaned'
    df_desc = df[feature].apply(smiles_to_descriptors).apply(pd.Series)
    df_fp = df[feature].apply(calc_fingerprints).apply(pd.Series)
    df_graph = df[feature].apply(smiles2graph).apply(pd.Series)

    return df_desc, df_fp, df_graph


def scale_desc(*data):
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


def scale_fp(*data):
    '''Data must be (train, valid, test)'''

    scaler = MinMaxScaler()
    scaler.fit(data[0])
    data = list(map(scaler.transform, data))

    return data


def fix_dim_data_features(data_feature, max_size=6):
    '''bring the dimensions to a single format'''

    if data_feature.shape[1] < max_size:
        return np.pad(data_feature, ((0, 0), (0, max_size - data_feature.shape[1])), constant_values=-1)
    elif data_feature.shape[1] > max_size:
        raise ValueError(f'Размер слишком большой: {data_feature.shape[1]} > {max_size}')
    return data_feature


def mapper_graph(graph):
    '''Custom mapper for DMPNN model'''

    mapper = _MapperDMPNN(graph)
    atom_features, f_ini_atoms_bonds, atom_to_incoming_bonds, mapping, global_features = mapper.values
    atom_features = torch.from_numpy(atom_features).float()
    f_ini_atoms_bonds = torch.from_numpy(f_ini_atoms_bonds).float()
    try:
        atom_to_incoming_bonds = torch.from_numpy(fix_dim_data_features(atom_to_incoming_bonds))
        mapping = torch.from_numpy(fix_dim_data_features(mapping))
    except Exception as e:
        raise e

    global_features = torch.from_numpy(global_features).float()
    data = Data(atom_features=atom_features,
                f_ini_atoms_bonds=f_ini_atoms_bonds,
                atom_to_incoming_bonds=atom_to_incoming_bonds,
                mapping=mapping, global_features=global_features)
    return data


def split_data(train_data, test_data, target='LogP'):
    '''Split data to train-valid'''

    combined_df = {'desc': None, 'fp': None, 'graph': None, 'y': None}
    y = train_data[target]
    X = train_data.drop(target, axis=1)
    X_test = test_data
    y_test = torch.zeros(len(X_test))

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    X_train_desc, X_train_fp, X_train_graph = get_features(X_train)
    X_val_desc, X_val_fp, X_val_graph = get_features(X_val)
    X_test_desc, X_test_fp, X_test_graph = get_features(X_test)
    X_train_desc, X_val_desc, X_test_desc = scale_desc(X_train_desc, X_val_desc, X_test_desc)
    X_train_fp, X_val_fp, X_test_fp = scale_fp(X_train_fp, X_val_fp, X_test_fp)

    combined_df['desc'] = [X_train_desc, X_val_desc, X_test_desc]
    combined_df['fp'] = [X_train_fp, X_val_fp, X_test_fp]
    combined_df['graph'] = [X_train_graph, X_val_graph, X_test_graph]
    combined_df['y'] = [y_train, y_val, y_test]

    combined_df['desc'] = list(map(torch.Tensor, map(np.array, combined_df['desc'])))
    combined_df['fp'] = list(map(torch.Tensor, map(np.array, combined_df['fp'])))
    combined_df['graph'] = list(map(lambda x: list(map(lambda y: mapper_graph(y[0]), x.values)), combined_df['graph']))
    combined_df['y'] = list(map(lambda x: torch.Tensor(x).unsqueeze(-1), map(np.array, combined_df['y'])))

    return combined_df








