import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score
from utils._npy_manipulation import *
import matplotlib.pyplot as plt
import numpy as np


class MLP_BCE(nn.Module):
    def __init__(self, input_dim, hidden1=512, hidden2=256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))
    # Mean Accuracy: 0.821 ± 0.021
    # Mean F1 Score: 0.891 ± 0.013

class MLP_BCE2(nn.Module):
    def __init__(self, input_dim, hidden1=512, hidden2=256, dropout_rate=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.bn1 = nn.BatchNorm1d(hidden1)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(hidden1, hidden2)
        self.bn2 = nn.BatchNorm1d(hidden2)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.fc3 = nn.Linear(hidden2, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)

        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)

        return torch.sigmoid(self.fc3(x))
    # Mean Accuracy: 0.830 ± 0.033
    # Mean F1 Score: 0.897 ± 0.021


class MLP_CE(nn.Module):
    def __init__(self, input_dim, hidden1=512, hidden2=256, dropout_rate=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.bn1 = nn.BatchNorm1d(hidden1)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(hidden1, hidden2)
        self.bn2 = nn.BatchNorm1d(hidden2)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.fc3 = nn.Linear(hidden2, 2)  # output=2 for binary classification

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        return self.fc3(x)  # return raw logits


def evaluate_MLP_BCE(data, metadata, train_split_target=0.6, validation_split_target=0.2, n_splits=5, n_epochs=1000, lr=1e-3, early_stopping_rounds=20, batch_size=64):
    accs = []
    f1s = []

    for seed in range(n_splits):
        # print(f'\n      Split {seed+1}/{n_splits}')
        torch.manual_seed(seed)

        X_train, X_val, X_test, y_train, y_val, y_test = train_validation_test_split_by_plotid(
            data, metadata, train_split_target, validation_split_target, seed=seed
        )

        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1) # .view(-1, 1) ensures y_train is shape (batch_size, 1) to match output
        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        model = MLP_BCE2(input_dim=data.shape[1])
        loss_fn = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        best_val_loss = float('inf') # inf guarantees first validation loss will always be lower than initialized value
        best_model = None
        patience_counter = 0
        
        # for epoch in range(n_epochs):
        #     model.train()
        #     y_pred = model(X_train)
        #     loss = loss_fn(y_pred, y_train)
            
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()

        for epoch in range(n_epochs):
            model.train()
            epoch_loss = 0.0

            for batch_X, batch_y in train_loader:
                y_pred = model(batch_X)
                loss = loss_fn(y_pred, batch_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            # Validation
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
                    # print(f"Early stopping at epoch {epoch}")
                    break

            # if epoch % 20 == 0:
            #     print(f"Epoch {epoch} | Train Loss: {loss.item():.3f} | Val Loss: {val_loss.item():.3f}")

        # evaluate best model on test
        model.load_state_dict(best_model) # loads saved weights and biases from best_model
        model.eval()
        with torch.no_grad():
            y_test_pred = model(X_test).round().numpy() # .numpy() converts pytorch tensor into a np array for metric computation
            y_test_true = y_test.numpy() # .numpy() converts pytorch tensor into a np array for metric computation

        acc = accuracy_score(y_test_true, y_test_pred)
        f1 = f1_score(y_test_true, y_test_pred)
        accs.append(acc)
        f1s.append(f1)
        # print(f'accuracy: {acc:.3f} | f1: {f1:.3f}')

    # print("\nMulti_Layer Perceptron")
    print(f"Mean Accuracy: {np.mean(accs):.3f} ± {np.std(accs):.3f}")
    print(f"Mean F1 Score: {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")


def evaluate_MLP_CE(data, metadata, train_split_target=0.6, validation_split_target=0.2, n_splits=5, n_epochs=1000, lr=1e-3, early_stopping_rounds=20, batch_size=64):
    accs = []
    f1s = []

    for seed in range(n_splits):
        # print(f'\n      Split {seed+1}/{n_splits}')
        torch.manual_seed(seed)

        X_train, X_val, X_test, y_train, y_val, y_test = train_validation_test_split_by_plotid(
            data, metadata, train_split_target, validation_split_target, seed=seed
        )

        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long).squeeze()
        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.long).squeeze()
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.long).squeeze() 

        train_dataset = TensorDataset(X_train, y_train.long())
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        model = MLP_CE(input_dim=data.shape[1])
        loss_fn = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        best_val_loss = float('inf') # inf guarantees first validation loss will always be lower than initialized value
        best_model = None
        patience_counter = 0
        
        for epoch in range(n_epochs):
            model.train()
            epoch_loss = 0.0

            for batch_X, batch_y in train_loader:
                y_pred = model(batch_X)
                loss = loss_fn(y_pred, batch_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

        # WITHOUT BATCH
        # for epoch in range(n_epochs):
        #     model.train()
        #     y_pred = model(X_train)
        #     loss = loss_fn(y_pred, y_train)
            
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()

            # Validation
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
                    # print(f"Early stopping at epoch {epoch}")
                    break

            # if epoch % 20 == 0:
            #     print(f"Epoch {epoch} | Train Loss: {loss.item():.3f} | Val Loss: {val_loss.item():.3f}")

        # evaluate best model on test
        model.load_state_dict(best_model) # loads saved weights and biases from best_model
        model.eval()

        # BCE
        # with torch.no_grad():
        #     y_test_pred = model(X_test).round().numpy() # .numpy() converts pytorch tensor into a np array for metric computation
        #     y_test_true = y_test.numpy() # .numpy() converts pytorch tensor into a np array for metric computation

        with torch.no_grad():
            test_logits = model(X_test)
            y_test_pred = torch.argmax(test_logits, dim=1).numpy()
            y_test_true = y_test.numpy()

        acc = accuracy_score(y_test_true, y_test_pred)
        f1 = f1_score(y_test_true, y_test_pred)
        accs.append(acc)
        f1s.append(f1)
        # print(f'accuracy: {acc:.3f} | f1: {f1:.3f}')

    print("\nMulti_Layer Perceptron")
    print(f"Mean Accuracy: {np.mean(accs):.3f} ± {np.std(accs):.3f}")
    print(f"Mean F1 Score: {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")

