from nnmodel import HybridDMPNN
from preprocessing_data import get_loaders
import pandas as pd

df_filtered = pd.read_csv('filtered_df.csv')
df_test = pd.read_csv('test_data80_cleaned.csv')

train, valid, test = get_loaders(df_filtered, df_test)

model = HybridDMPNN(fingerprint_size=train.shape[1],
                  numeric_features_size=Xd_train.shape[1]).to(device)

# metrics_callback = MetricsCallback(val_loader, train_loader, device)

trainer = pyl.Trainer(max_epochs=EPOCHS, accelerator="auto")
trainer.fit(model, train_loader, val_loader)