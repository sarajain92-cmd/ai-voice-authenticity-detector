import torch
import pandas as pd
import joblib
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class CSVDataset(Dataset):
    def __init__(self, csv_path, scaler_path="models/scaler.pkl", fit_scaler=False):
        df = pd.read_csv(csv_path)

        self.X = df.drop("LABEL", axis=1).values.astype("float32")
        self.y = df["LABEL"].map({"REAL": 0, "FAKE": 1}).values
        assert not pd.isnull(self.y).any(), "Label mapping failed!"
        if fit_scaler:
            self.scaler = StandardScaler()
            self.X = self.scaler.fit_transform(self.X)
            joblib.dump(self.scaler, scaler_path)
        else:
            self.scaler = joblib.load(scaler_path)
            self.X = self.scaler.transform(self.X)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx]).float()
        y = torch.tensor(self.y[idx]).long()
        return x, y