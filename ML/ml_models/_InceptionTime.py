import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score
from utils._npy_manipulation import *
import matplotlib.pyplot as plt
import numpy as np


class InceptionLayer(nn.Module):
    # PyTorch translation of the Keras code in https://github.com/hfawaz/dl-4-tsc
    def __init__(self, nb_filters=32, use_bottleneck=True,
                 bottleneck_size=32, kernel_size=40):
        super(InceptionLayer, self).__init__()

        # self.in_channels = in_channels
        kernel_size_s = [(kernel_size) // (2 ** i) for i in range(3)] # = [40, 20, 10]
        kernel_size_s = [x+1 for x in kernel_size_s] # Avoids warning about even kernel_size with padding="same"
        self.bottleneck_size = bottleneck_size
        self.use_bottleneck = use_bottleneck


        # Bottleneck layer
        self.bottleneck = nn.LazyConv1d(self.bottleneck_size, kernel_size=1,
                                    stride=1, padding="same", bias=False)
        self.max_pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.bottleneck_conv = nn.LazyConv1d(nb_filters, kernel_size=1,
                                         stride=1, padding="same", bias=False)

        # Convolutional layer (several filter lenghts)
        self.conv_list = nn.ModuleList([])
        for i in range(len(kernel_size_s)):
            # Input size could be self.in_channels or self.bottleneck_size (if bottleneck was applied)
            self.conv_list.append(nn.LazyConv1d(nb_filters, kernel_size=kernel_size_s[i],
                                            stride=1, padding='same', bias=False))

        self.bn = nn.BatchNorm1d(4*self.bottleneck_size)
        self.relu = nn.ReLU()

    def forward(self, input):
        in_channels = input.shape[-2]
        if self.use_bottleneck and int(in_channels) > self.bottleneck_size:
            input_inception = self.bottleneck(input)
        else:
            input_inception = input

        max_pool = self.max_pool(input)
        output = self.bottleneck_conv(max_pool)
        for conv in self.conv_list:
            output = torch.cat((output,conv(input_inception)),dim=1)

        output = self.bn(output)
        output = self.relu(output)

        return output


class Inception(nn.Module):
    # PyTorch translation of the Keras code in https://github.com/hfawaz/dl-4-tsc
    def __init__(self, nb_classes, nb_filters=32, use_residual=True,
                 use_bottleneck=True, bottleneck_size=32, depth=6, kernel_size=41):
        super(Inception, self).__init__()

        self.use_residual = use_residual

        # Inception layers
        self.inception_list = nn.ModuleList(
            [InceptionLayer(nb_filters,use_bottleneck, bottleneck_size, kernel_size) for _ in range(depth)])
        # Explicit input sizes (i.e. without using Lazy layers). Requires n_var passed as a constructor input
        # self.inception_list = nn.ModuleList([InceptionLayer(n_var, nb_filters,use_bottleneck, bottleneck_size, kernel_size) for _ in range(depth)])
        # for _ in range(1,depth):
        #     inception = InceptionLayer(4*nb_filters,nb_filters,use_bottleneck, bottleneck_size, kernel_size)
        #     self.inception_list.append(inception)

        # Fully-connected layer
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(nb_classes),
            # nn.Softmax(dim=1) # already performed inside CrossEntropyLoss
        )

        # Shortcut layers
        # First residual layer has n_var channels as inputs while the remaining have 4*nb_filters
        self.conv = nn.ModuleList([
            nn.LazyConv1d(4*nb_filters, kernel_size=1,
                            stride=1, padding="same", bias=False)
            for _ in range(int(depth/3))
        ])
        self.bn = nn.ModuleList([nn.BatchNorm1d(4*nb_filters) for _ in range(int(depth/3))])
        self.relu = nn.ModuleList([nn.ReLU() for _ in range(int(depth/3))])

    def _shortcut_layer(self, input_tensor, out_tensor, id):
        shortcut_y = self.conv[id](input_tensor)
        shortcut_y = self.bn[id](shortcut_y)
        x = torch.add(shortcut_y, out_tensor)
        x = self.relu[id](x)
        return x

    def forward(self, x):
        input_res = x

        for d, inception in enumerate(self.inception_list):
            x = inception(x)

            # Residual layer
            if self.use_residual and d % 3 == 2:
                x = self._shortcut_layer(input_res,x, int(d/3))
                input_res = x

        gap_layer = self.gap(x)
        return self.classifier(gap_layer), gap_layer, x
    

def evaluate_Inception_classification(data, metadata, nb_classes, train_split_target=0.6, validation_split_target=0.2, n_splits=5, n_epochs=1000, lr=1e-3, early_stopping_rounds=20, batch_size=64):
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

        y_train = torch.tensor(y_train, dtype=torch.long)
        y_val   = torch.tensor(y_val,   dtype=torch.long)
        y_test  = torch.tensor(y_test,  dtype=torch.long)

        X_train, X_val, X_test = X_train.to(device), X_val.to(device), X_test.to(device)
        y_train, y_val, y_test = y_train.to(device), y_val.to(device), y_test.to(device)

        # Create DataLoader for batching
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        model = Inception(nb_classes=nb_classes)
        model.to(device)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        best_val_loss = float('inf')
        best_model = None
        patience_counter = 0

        for epoch in range(n_epochs):
            model.train()
            epoch_loss = 0.0

            for batch_X, batch_y in train_loader:
                logits, _, _ = model(batch_X)
                loss = loss_fn(logits, batch_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            # Validation
            model.eval()
            with torch.no_grad():
                val_logits, _, _ = model(X_val)
                val_loss = loss_fn(val_logits, y_val)

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
                print(f'Epoch {epoch} | Train Loss: {epoch_loss:.3f} | Val Loss: {val_loss.item():.3f}')

        # Evaluate best model on test set
        model.load_state_dict(best_model)
        model.eval()

        with torch.no_grad():
            logits, _, _ = model(X_test)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            y_true = y_test.cpu().numpy()

        # print("Unique labels in y_test_pred:", np.unique(preds))
        # print("Unique labels in y_test_true:", np.unique(y_true))

        acc = accuracy_score(y_true, preds)
        f1 = f1_score(y_true, preds, average='binary' if nb_classes == 2 else 'weighted')
        accs.append(acc)
        f1s.append(f1)
        print(f'accuracy: {acc:.3f} | f1: {f1:.3f}')

    print('\nInception Classification')
    print(f'Mean Accuracy: {np.mean(accs):.3f} ± {np.std(accs):.3f}')
    print(f'Mean F1 Score: {np.mean(f1s):.3f} ± {np.std(f1s):.3f}')
