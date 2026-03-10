import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from utils._npy_manipulation import *
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

class Conv1D_BatchNorm_Relu_Dropout(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, drop_probability=0.5):
        super().__init__()
        
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(p=drop_probability)
        )

    def forward(self, X):
        return self.block(X)


class FC_Classifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims=256, drop_probability=0.5):
        super().__init__()
        
        self.block = nn.Sequential(
            nn.Linear(input_dim, hidden_dims),
            nn.BatchNorm1d(hidden_dims),
            nn.ReLU(),
            nn.Dropout(p=drop_probability),
            nn.Linear(hidden_dims, 1)
        )

    def forward(self, X):
        return self.block(X)


class TempCNN(torch.nn.Module): 
    def __init__(self, kernel_size=5, hidden_dims=64, dropout=0.3, input_channels=10, input_timesteps=45):
        super(TempCNN, self).__init__()
        self.conv_bn_relu1 = Conv1D_BatchNorm_Relu_Dropout(input_channels, hidden_dims, kernel_size=kernel_size, drop_probability=dropout)
        self.conv_bn_relu2 = Conv1D_BatchNorm_Relu_Dropout(hidden_dims, hidden_dims, kernel_size=kernel_size, drop_probability=dropout)
        self.conv_bn_relu3 = Conv1D_BatchNorm_Relu_Dropout(hidden_dims, hidden_dims, kernel_size=kernel_size, drop_probability=dropout)

        self.flatten = nn.Flatten()

        # Dummy pass to determine the flattened size
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, input_timesteps)
            dummy_out = self.conv_bn_relu3(self.conv_bn_relu2(self.conv_bn_relu1(dummy_input)))
            flattened_dim = dummy_out.numel()

        self.classifier = FC_Classifier(input_dim=flattened_dim, hidden_dims=256)

    def forward(self, x):
        # require NxTxD
        conv1 = self.conv_bn_relu1(x)
        conv2 = self.conv_bn_relu2(conv1)
        conv3 = self.conv_bn_relu3(conv2)
        emb = self.flatten(conv3)
        return self.classifier(emb) # return self.classifier(emb), emb
    # Mean Accuracy: 0.855 ± 0.024
    # Mean F1 Score: 0.915 ± 0.015
    

def evaluate_TempCNN_binary(data, metadata, train_split_target=0.6, validation_split_target=0.2, n_splits=5, n_epochs=1000, lr=1e-3, early_stopping_rounds=20, batch_size=None):
    accs = []
    f1s = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for seed in range(n_splits):
        print(f'\n      Split {seed+1}/{n_splits}')
        torch.manual_seed(seed)

        X_train, X_val, X_test, y_train, y_val, y_test = train_validation_test_split_by_plotid(
            data, metadata, train_split_target, validation_split_target, seed=seed
        )

        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_val   = torch.tensor(X_val,   dtype=torch.float32)
        X_test  = torch.tensor(X_test,  dtype=torch.float32)

        y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        y_val   = torch.tensor(y_val,   dtype=torch.float32).view(-1, 1)
        y_test  = torch.tensor(y_test,  dtype=torch.float32).view(-1, 1)

        X_train, X_val, X_test = X_train.to(device), X_val.to(device), X_test.to(device)
        y_train, y_val, y_test = y_train.to(device), y_val.to(device), y_test.to(device)

        if batch_size:
            train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
            val_loader   = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)
            test_loader  = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size)

        model = TempCNN(input_channels=data.shape[1], input_timesteps=data.shape[2])
        model.to(device)

        loss_fn = nn.BCEWithLogitsLoss() # no Sigmoid in model
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        best_val_loss = float('inf')
        best_model = None
        patience_counter = 0

        for epoch in range(n_epochs):
            model.train()

            if batch_size:
                total_loss = 0
                for batch_X, batch_y in train_loader:
                    pred = model(batch_X)
                    loss = loss_fn(pred, batch_y)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
            else:
                y_pred = model(X_train)
                loss = loss_fn(y_pred, y_train)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_pred = model(X_val)
                val_loss = loss_fn(val_pred, y_val)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_rounds:
                    print(f'Early stopping at epoch {epoch}')
                    break

            if epoch % 20 == 0:
                print(f'Epoch {epoch} | Train Loss: {loss.item():.3f} | Val Loss: {val_loss.item():.3f}')

        # evaluate best model on test
        model.load_state_dict(best_model) # loads saved weights and biases from best_model
        model.eval()

        with torch.no_grad():
            logits = model(X_test)  # shape: (N, 1)
            print('Raw logits:', logits[:10].squeeze())

            probas = torch.sigmoid(logits)  # convert logits to [0, 1]
            y_test_pred = (probas > 0.5).cpu().int().numpy() # .numpy() converts pytorch tensor into a np array for metric computation

            y_test_true = y_test.int().cpu().numpy() # .numpy() converts pytorch tensor into a np array for metric computation

        acc = accuracy_score(y_test_true, y_test_pred)
        f1 = f1_score(y_test_true, y_test_pred)
        accs.append(acc)
        f1s.append(f1)
        print(f'accuracy: {acc:.3f} | f1: {f1:.3f}')

    print(f'\nTempCNN Binary Classification {batch_size} {lr}')
    print(f'Mean Accuracy: {np.mean(accs):.3f} ± {np.std(accs):.3f}')
    print(f'Mean F1 Score: {np.mean(f1s):.3f} ± {np.std(f1s):.3f}')
