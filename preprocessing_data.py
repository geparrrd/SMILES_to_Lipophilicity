import pandas as pd
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, MolFromSmiles, MACCSkeys, LayeredFingerprint
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator, GetAtomPairGenerator, GetTopologicalTorsionGenerator
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from deepchem.feat import DMPNNFeaturizer
import torch
from torch_geometric.data import Data
from deepchem.models.torch_models.dmpnn import _MapperDMPNN
import joblib
import warnings
warnings.filterwarnings("ignore")
RDLogger.DisableLog('rdApp.*')

RANDOM_STATE = 616


def smiles_to_descriptors(smiles):
    '''Get descriptors'''

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
    morgan_np = morgan_fpgenerator.GetFingerprintAsNumPy(mol)

    maccs = MACCSkeys.GenMACCSKeys(mol)
    maccs_np = np.array(maccs)

    # Atom Pair
    # ap_gen = GetAtomPairGenerator(fpSize=fp_size)
    # atom_pair_np = ap_gen.GetCountFingerprintAsNumPy(mol)

    # Layered
    # layered = LayeredFingerprint(mol, fpSize=2 * fp_size)
    # layered_np = np.array(layered)

    # Topological Torsion
    # topo_gen = GetTopologicalTorsionGenerator(fpSize=2 * fp_size)
    # torsion_np = topo_gen.GetCountFingerprintAsNumPy(mol)

    combined_fp = np.concatenate([
        morgan_np, maccs_np
    ])

    return combined_fp


def smiles2graph(sm):
    '''Get graph features'''

    featurizer = DMPNNFeaturizer()
    graph = featurizer.featurize(sm)
    return graph


def scale_desc(df, is_test=False):
    '''Custom scaler for descriptors'''

    cols2log = ['MW', 'SPS', 'NumRotatableBonds', 'NHOHCount', 'TPSA']

    for col in cols2log:
        if col in df.columns:
            df[col] = df[col].apply(np.log1p)

    if not is_test:
        scaler = MinMaxScaler()
        scaler.fit(df)
        joblib.dump(scaler, 'scaler_desc.pkl')
    else:
        scaler = joblib.load('scaler_desc.pkl')
    df = scaler.transform(df)

    return df


def scale_fp(df, is_test=False):
    '''Scaler for fingerprints'''

    if not is_test:
        scaler = MinMaxScaler()
        scaler.fit(df)
        joblib.dump(scaler, 'scaler_fp.pkl')
    else:
        scaler = joblib.load('scaler_fp.pkl')
    df = scaler.transform(df)

    return df


def get_features(df):
    '''Get all features'''

    df_desc = df.apply(smiles_to_descriptors).apply(pd.Series)
    df_fp = df.apply(calc_fingerprints).apply(pd.Series)
    df_graph = df.apply(smiles2graph).apply(pd.Series)

    return {'desc': df_desc, 'fp': df_fp, 'graph': df_graph}


def fix_dim_data_features(data_feature, max_size=6):
    '''Bring the dimensions to a single format'''

    if data_feature.shape[1] < max_size:
        return np.pad(data_feature, ((0, 0), (0, max_size - data_feature.shape[1])), constant_values=-1)
    elif data_feature.shape[1] > max_size:
        raise ValueError(f'Too big size: {data_feature.shape[1]} > {max_size}')
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


def split_dict_arrays(data_dict, test_size=0.2, random_state=RANDOM_STATE):
    '''Split train dataset to train/valid'''

    n_samples = list(data_dict.values())[0].shape[0]
    indices = np.arange(n_samples)

    train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=random_state)

    train_dict = {key: val.iloc[train_idx, :] for key, val in data_dict.items()}
    test_dict  = {key: val.iloc[test_idx, :]  for key, val in data_dict.items()}

    return train_dict, test_dict


def get_data(df, target='LogP', is_test=False):
    '''Prepare data for training or inference of the model'''

    combined_df = {'train': None, 'val': None, 'test': None}
    y = df[[target]] if not is_test else torch.zeros(len(df))

    featured_df = get_features(df['Smiles_cleaned'])
    featured_df['y'] = y
    if not is_test:
        train_df, valid_df = split_dict_arrays(featured_df)
        train_df['desc'] = scale_desc(train_df['desc'])
        train_df['fp'] = scale_fp(train_df['fp'])
        valid_df['desc'] = scale_desc(valid_df['desc'], is_test=True)
        valid_df['fp'] = scale_fp(valid_df['fp'], is_test=True)
        combined_df['train'], combined_df['val'] = train_df, valid_df
    else:
        test_df = featured_df
        test_df['desc'] = scale_desc(test_df['desc'], is_test=True)
        test_df['fp'] = scale_fp(test_df['fp'], is_test=True)
        combined_df['test'] = test_df

    for name, data in combined_df.items():
        if data is not None:
            data['desc'] = torch.Tensor(data['desc'])
            data['fp'] = torch.Tensor(data['fp'])
            data['graph'] = list(map(lambda x: mapper_graph(x[0]), data['graph'].to_numpy()))
            data['y'] = torch.Tensor(data['y'].to_numpy())

    return combined_df
