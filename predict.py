import os.path

import pandas as pd
from nnmodel import HybridDMPNN
from nnmodel import pyl
from preprocessing_data import get_data
from loaders.make_loaders import get_loaders


def predict():
    '''Function to predict'''

    df_file = 'data/test_data.csv'
    data = pd.read_csv(df_file)
    comb_df = get_data(data, is_test=True)
    test_loader = get_loaders(comb_df, is_test=True)

    model_path = input('Path to model (*.ckpt): ')
    if os.path.exists(model_path):
        trained_model = HybridDMPNN.load_from_checkpoint(model_path)
        trainer = pyl.Trainer()
        trainer.test(trained_model, test_loader)

        y_pred_test = trained_model.test_predictions

        submission = pd.read_csv('data/final_sample_submission80.csv')
        submission['LogP'] = list(map(lambda x: x[0], y_pred_test))

        filename = 'submission.csv'
        i = 0
        while os.path.exists(filename):
            i += 1
            name, ext = filename.split('.')
            filename = f'{name}_{i}.{ext}'
        submission.to_csv(f'submissions/{filename}', index=False)
    else:
        print('Model is not found')


predict()
