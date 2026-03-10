import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from utils._npy_manipulation import *
from utils.augmentations import *


class Conv1D_BatchNorm_Relu_Dropout(nn.Module):
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


class FC_Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dims=256, drop_probability=0.5):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, hidden_dims),
            nn.BatchNorm1d(hidden_dims),
            nn.ReLU(),
            nn.Dropout(p=drop_probability),
            nn.Linear(hidden_dims, 1)  # logits
        )

    def forward(self, X):
        return self.block(X)


# TempCNN encoder for SSL
class TempCNN_Encoder(nn.Module):
    """
    Same conv stack as your TempCNN, but without the classifier.
    Adds a projection head for SimCLR during pretraining.
    """
    def __init__(self, kernel_size=5, hidden_dims=64, dropout=0.3,
                 input_channels=10, input_timesteps=45, proj_dim=128):
        super().__init__()
        self.conv_bn_relu1 = Conv1D_BatchNorm_Relu_Dropout(input_channels, hidden_dims, kernel_size, dropout)
        self.conv_bn_relu2 = Conv1D_BatchNorm_Relu_Dropout(hidden_dims, hidden_dims, kernel_size, dropout)
        self.conv_bn_relu3 = Conv1D_BatchNorm_Relu_Dropout(hidden_dims, hidden_dims, kernel_size, dropout)
        self.flatten = nn.Flatten()

        # determine flattened dim
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, input_timesteps)
            x = self.conv_bn_relu3(self.conv_bn_relu2(self.conv_bn_relu1(dummy)))
            self.flattened_dim = x.numel()

        # Projection head (only used in SSL)
        self.projection = nn.Sequential(
            nn.Linear(self.flattened_dim, 256),
            nn.ReLU(),
            nn.Linear(256, proj_dim)
        )

    def forward_features(self, x):
        x = self.conv_bn_relu1(x)
        x = self.conv_bn_relu2(x)
        x = self.conv_bn_relu3(x)
        emb = self.flatten(x)  # [B, flattened_dim]
        return emb

    def forward(self, x, project=True, l2norm=True):
        emb = self.forward_features(x)
        if project:
            z = self.projection(emb)
        else:
            z = emb
        if l2norm:
            z = F.normalize(z, dim=-1)
        return z  # representation (projected or not)


# SimCLR loss (NT-Xent)
def simclr_loss(z1, z2, temperature=0.5):
    """
    z1, z2: [B, D] L2-normalized embeddings from two augmented views.
    Implements NT-Xent across the 2B samples.
    """
    B, D = z1.shape
    z = torch.cat([z1, z2], dim=0)                    # [2B, D]
    sim = torch.matmul(z, z.T)                        # cosine sim since normalized
    sim = sim / temperature

    # mask self-similarity
    mask = torch.eye(2 * B, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, -1e9)

    # For each i in [0..2B-1], the positive index is i^B (swap halves):
    # 0<->B, 1<->B+1, ..., B-1<->2B-1
    pos_index = torch.arange(2 * B, device=z.device)
    pos_index = (pos_index + B) % (2 * B)

    # Cross-entropy over rows; target is the index of the positive sample in that row
    loss = F.cross_entropy(sim, pos_index)
    return loss
    

# Pretraining (SSL) routine
def pretrain_simclr_encoder(
    data,                       # numpy array [N, C, T]
    n_epochs=100,
    batch_size=256,
    lr=1e-3,
    temperature=0.5,
    kernel_size=5,
    hidden_dims=64,
    dropout=0.3,
    proj_dim=128,
    aug_cfg=None,
    verbose_every=10
):
    """
    Returns a pretrained TempCNN_Encoder (with projection head intact).
    Uses ALL provided 'data' as unlabeled for SSL.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = torch.tensor(data, dtype=torch.float32)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    encoder = TempCNN_Encoder(
        kernel_size=kernel_size,
        hidden_dims=hidden_dims,
        dropout=dropout,
        input_channels=data.shape[1],
        input_timesteps=data.shape[2],
        proj_dim=proj_dim
    ).to(device)

    optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)

    if aug_cfg is None:
        # Default: jitter + crop + mask
        aug_cfg = {
            'augmentations': ['jitter', 'random_crop', 'temporal_mask'],
            'jitter': {'std': 0.01},
            'random_crop': {'min_crop': 0.8, 'max_crop': 1.0},
            'temporal_mask': {'prob': 0.2, 'span_ratio': 0.1}
        }
    aug = Augmentations(**aug_cfg)

    for epoch in range(1, n_epochs + 1):
        encoder.train()
        epoch_loss = 0.0
        for batch in loader:
            batch = batch.to(device)  # [B, C, T]
            x1, x2 = aug(batch)

            z1 = encoder(x1, project=True, l2norm=True)
            z2 = encoder(x2, project=True, l2norm=True)
            loss = simclr_loss(z1, z2, temperature=temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        if epoch % verbose_every == 0 or epoch == 1 or epoch == n_epochs:
            print(f"Epoch {epoch}/{n_epochs} | Loss: {epoch_loss/len(loader):.4f}")

    return encoder


# Fine-tune classifier on labels
# class TempCNN_Finetune(nn.Module):
#     """
#     Wraps a pretrained encoder and adds a classifier head.
#     Projection head is disabled; we use encoder features.
#     """
#     def __init__(self, pretrained_encoder: TempCNN_Encoder, hidden_dims_cls=256, drop_probability=0.5):
#         super().__init__()
#         self.encoder = pretrained_encoder
#         # disable projection in forward for classification; we will call forward_features directly
#         self.classifier = FC_Classifier(input_dim=self.encoder.flattened_dim,
#                                         hidden_dims=hidden_dims_cls,
#                                         drop_probability=drop_probability)

#     def forward(self, x):
#         emb = self.encoder.forward_features(x)  # [B, flattened_dim]
#         logits = self.classifier(emb)           # [B, 1]
#         return logits

class TempCNN_HeadClassifier(nn.Module):
    def __init__(self, pretrained_encoder: TempCNN_Encoder,
            mode="finetune", hidden_dims_cls=256, drop_probability=0.5):
        super().__init__()
        self.encoder = pretrained_encoder
        self.mode = mode

        if mode == "probe":
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.encoder.eval()
            # self.head = nn.Linear(self.encoder.flattened_dim, 1)
            self.head = FC_Classifier(input_dim=self.encoder.flattened_dim,
                            hidden_dims=hidden_dims_cls,
                            drop_probability=drop_probability)
        else:
            self.head = FC_Classifier(input_dim=self.encoder.flattened_dim,
                                    hidden_dims=hidden_dims_cls,
                                    drop_probability=drop_probability)

    def forward(self, x):
        if self.mode == "probe":
            with torch.no_grad():
                emb = self.encoder.forward_features(x)
        else:
            emb = self.encoder.forward_features(x)
        return self.head(emb)


def evaluate_TempCNN_binary_with_optional_simclr(
    data, metadata,
    train_split_target=0.6, validation_split_target=0.2,
    n_splits=5,
    augmentation=None,
    use_simclr_pretrain=True,
    linear_probe=False,
    n_epochs_supervised=50,
    lr_supervised=1e-3,
    early_stopping_rounds=20,
    batch_size=256,
    # SSL switches
    simclr_epochs=300,
    simclr_batch_size=256,
    simclr_lr=1e-3,
    simclr_temperature=0.5,
    kernel_size=5, hidden_dims=64, dropout=0.3, proj_dim=128
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if use_simclr_pretrain:
        print('[SimCLR pretraining on whole data]')
        encoder = pretrain_simclr_encoder(
            data=data,
            n_epochs=simclr_epochs,
            batch_size=simclr_batch_size,
            lr=simclr_lr,
            temperature=simclr_temperature,
            kernel_size=kernel_size,
            hidden_dims=hidden_dims,
            dropout=dropout,
            proj_dim=proj_dim,
            aug_cfg=augmentation,
            verbose_every=max(1, simclr_epochs // 10)
        )
        
    else:
        encoder = TempCNN_Encoder(
            kernel_size=kernel_size, hidden_dims=hidden_dims, dropout=dropout,
            input_channels=data.shape[1], input_timesteps=data.shape[2], proj_dim=proj_dim
        )

    encoder.to(device)

    initial_state = {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()}

    results = {subset: {"acc": [], "f1": []} for subset in [5, 10, 20, 40]}

    for len_subset in [5, 10, 20, 40]:
        for seed in range(n_splits):
            print(f'\n===== Split {seed+1}/{n_splits} | subset={len_subset} =====')
            torch.manual_seed(seed)
            np.random.seed(seed)

            X_train, y_train, X_test, y_test = finetune_split(data, metadata, len_subset, seed=seed)
            encoder.load_state_dict({k: v.clone() for k, v in initial_state.items()})
            encoder.to(device)

            model = TempCNN_HeadClassifier(encoder, mode="probe" if linear_probe else "finetune").to(device)

            if linear_probe:
                optimizer = torch.optim.Adam(model.head.parameters(), lr=lr_supervised)
            else:
                optimizer = torch.optim.Adam(model.parameters(), lr=lr_supervised)

            X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
            y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)

            X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
            y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)

            loss_fn = nn.BCEWithLogitsLoss()

            if batch_size and batch_size > 0:
                train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
            else:
                train_loader = [(X_train_t, y_train_t)]

            best_loss = float('inf')
            best_state, patience = None, 0

            for epoch in range(1, n_epochs_supervised + 1):
                model.train()
                running = 0.0
                for xb, yb in train_loader:
                    logits = model(xb)
                    loss = loss_fn(logits, yb)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    running += loss.item()

                mean_loss = running / len(train_loader)

                if mean_loss < best_loss:
                    best_loss = mean_loss
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    patience = 0
                else:
                    patience += 1

                if epoch % 20 == 0 or epoch == 1 or epoch == n_epochs_supervised:
                    print(f'[FT] Epoch {epoch}/{n_epochs_supervised} | TrainLoss {mean_loss:.4f}')

                if patience >= early_stopping_rounds:
                    print(f'[FT] Early stopping at epoch {epoch}')
                    break

            # restore best
            if best_state is not None:
                model.load_state_dict(best_state)

            # test
            model.eval()
            with torch.no_grad():
                logits = model(X_test_t)
                probas = torch.sigmoid(logits).cpu().numpy()
                y_pred = (probas > 0.5).astype(np.int32)
                y_true = y_test_t.cpu().numpy().astype(np.int32)

            acc = accuracy_score(y_true, y_pred)
            f1  = f1_score(y_true, y_pred)
            results[len_subset]["acc"].append(acc)
            results[len_subset]["f1"].append(f1)

            print(f'[Test] accuracy: {acc:.3f} | f1: {f1:.3f}')

    print('\n=== TempCNN SSL Results ===')
    for subset, metrics in results.items():
        print(f"Subset {subset:>2}: "
              f"Accuracy {np.mean(metrics['acc']):.3f} ± {np.std(metrics['acc']):.3f} | "
              f"F1 {np.mean(metrics['f1']):.3f} ± {np.std(metrics['f1']):.3f}")
