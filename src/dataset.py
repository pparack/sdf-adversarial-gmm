import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from typing import List, Tuple

def make_tensors_and_loaders(train_df, valid_df, test_df, feature_names: List[str],
                             device,
                             batch_size_train=1024, batch_size_eval=8192):
    X_train = train_df[feature_names].values.astype(np.float32)
    y_train = train_df['y_excess'].values.astype(np.float32)
    X_valid = valid_df[feature_names].values.astype(np.float32)
    y_valid = valid_df['y_excess'].values.astype(np.float32)
    X_test  = test_df[feature_names].values.astype(np.float32)
    y_test  = test_df['y_excess'].values.astype(np.float32)

    train_X_t = torch.from_numpy(X_train).to(device)
    train_y_t = torch.from_numpy(y_train).to(device)
    valid_X_t = torch.from_numpy(X_valid).to(device)
    valid_y_t = torch.from_numpy(y_valid).to(device)
    test_X_t  = torch.from_numpy(X_test).to(device)
    test_y_t  = torch.from_numpy(y_test).to(device)

    train_ds = TensorDataset(train_X_t, train_y_t)
    valid_ds = TensorDataset(valid_X_t, valid_y_t)
    test_ds  = TensorDataset(test_X_t,  test_y_t)

    train_loader = DataLoader(train_ds, batch_size=batch_size_train, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size_eval, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size_eval, shuffle=False)

    return (train_X_t, train_y_t, valid_X_t, valid_y_t, test_X_t, test_y_t,
            train_loader, valid_loader, test_loader)
