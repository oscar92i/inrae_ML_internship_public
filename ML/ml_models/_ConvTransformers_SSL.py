import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import math
from einops import rearrange
  
from utils._npy_manipulation import *
from utils.augmentations import *


class ConvTranEncoder(nn.Module):
    def __init__(self, config, proj_dim=None):
        super().__init__()
        channel_size, seq_len = config['Data_shape']
        emb_size = config['emb_size']
        num_heads = config['num_heads']
        dim_ff = config['dim_ff']
        self.Fix_pos_encode = config['Fix_pos_encode']
        self.Rel_pos_encode = config['Rel_pos_encode']

        # Embedding Layers
        self.embed_layer = nn.Sequential(
            nn.Conv2d(1, emb_size*4, kernel_size=[1, 8], padding='same'),
            nn.BatchNorm2d(emb_size*4),
            nn.GELU()
        )

        self.embed_layer2 = nn.Sequential(
            nn.Conv2d(emb_size*4, emb_size, kernel_size=[channel_size, 1], padding='valid'),
            nn.BatchNorm2d(emb_size),
            nn.GELU()
        )

        # Position encoding
        if self.Fix_pos_encode == 'tAPE':
            self.Fix_Position = tAPE(emb_size, dropout=config['dropout'], max_len=seq_len)
        elif self.Fix_pos_encode == 'Sin':
            self.Fix_Position = AbsolutePositionalEncoding(emb_size, dropout=config['dropout'], max_len=seq_len)
        elif self.Fix_pos_encode == 'Learn':
            self.Fix_Position = LearnablePositionalEncoding(emb_size, dropout=config['dropout'], max_len=seq_len)
        else:
            self.Fix_Position = None

        # Attention
        self.attention_layer = Attention(emb_size, num_heads, config['dropout'])

        self.LayerNorm = nn.LayerNorm(emb_size, eps=1e-5)
        self.LayerNorm2 = nn.LayerNorm(emb_size, eps=1e-5)

        self.FeedForward = nn.Sequential(
            nn.Linear(emb_size, dim_ff),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(dim_ff, emb_size),
            nn.Dropout(config['dropout'])
        )

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.emb_size = emb_size
        self.flattened_dim = emb_size

        # Optional projection head for SimCLR
        if proj_dim is not None:
            self.projection = nn.Sequential(
                nn.Linear(self.flattened_dim, emb_size),
                nn.ReLU(),
                nn.Linear(emb_size, proj_dim)
            )
        else:
            self.projection = None

    def forward(self, x, project=False, l2norm=False):
        features = self.forward_features(x)
        if self.projection is not None and project:
            features = self.projection(features)
        if l2norm:
            features = F.normalize(features, dim=-1)
        return features

    def forward_features(self, x):
        """Return embeddings without projection (used for classification)."""
        x = x.unsqueeze(1)  # (B, 1, C, T)
        x_src = self.embed_layer(x)
        x_src = self.embed_layer2(x_src).squeeze(2)
        x_src = x_src.permute(0, 2, 1)

        if self.Fix_Position is not None:
            x_src = self.Fix_Position(x_src)

        att = x_src + self.attention_layer(x_src)
        att = self.LayerNorm(att)
        out = att + self.FeedForward(att)
        out = self.LayerNorm2(out)
        out = out.permute(0, 2, 1)
        out = self.gap(out)
        out = self.flatten(out)
        return out


# ======================================================
# Classifier head (separate)
# ======================================================

class ConvTranClassifier(nn.Module):
    def __init__(self, encoder, num_classes=1):
        super().__init__()
        self.encoder = encoder
        self.fc = nn.Linear(encoder.emb_size, num_classes)

    def forward(self, x):
        features = self.encoder(x)
        return self.fc(features)
    

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


# ======================================================
# SSL Wrapper
# ======================================================

class SSLConvTran(nn.Module):
    def __init__(self, encoder, projection_dim=128):
        super().__init__()
        self.encoder = encoder
        emb_size = encoder.emb_size
        self.projection_head = nn.Sequential(
            nn.Linear(emb_size, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, projection_dim)
        )

    def forward(self, x):
        features = self.encoder(x)
        z = self.projection_head(features)
        return F.normalize(z, dim=-1)


# ======================================================
# Attention
# ======================================================

class Attention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.scale = emb_size ** -0.5
        # self.to_qkv = nn.Linear(inp, inner_dim * 3, bias=False)
        self.key = nn.Linear(emb_size, emb_size, bias=False)
        self.value = nn.Linear(emb_size, emb_size, bias=False)
        self.query = nn.Linear(emb_size, emb_size, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.LayerNorm(emb_size)

    def forward(self, x):

        batch_size, seq_len, _ = x.shape
        k = self.key(x).reshape(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 3, 1)
        v = self.value(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        q = self.query(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        # k,v,q shape = (batch_size, num_heads, seq_len, d_head)

        attn = torch.matmul(q, k) * self.scale
        # attn shape (seq_len, seq_len)
        attn = nn.functional.softmax(attn, dim=-1)

        # import matplotlib.pyplot as plt
        # plt.plot(x[0, :, 0].detach().cpu().numpy())
        # plt.show()

        out = torch.matmul(attn, v)
        # out.shape = (batch_size, num_heads, seq_len, d_head)
        out = out.transpose(1, 2)
        # out.shape == (batch_size, seq_len, num_heads, d_head)
        out = out.reshape(batch_size, seq_len, -1)
        # out.shape == (batch_size, seq_len, d_model)
        out = self.to_out(out)
        return out


class Attention_Rel_Scl(nn.Module):
    def __init__(self, emb_size, num_heads, seq_len, dropout):
        super().__init__()
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.scale = emb_size ** -0.5
        # self.to_qkv = nn.Linear(inp, inner_dim * 3, bias=False)
        self.key = nn.Linear(emb_size, emb_size, bias=False)
        self.value = nn.Linear(emb_size, emb_size, bias=False)
        self.query = nn.Linear(emb_size, emb_size, bias=False)

        self.relative_bias_table = nn.Parameter(torch.zeros((2 * self.seq_len - 1), num_heads))
        coords = torch.meshgrid((torch.arange(1), torch.arange(self.seq_len)))
        coords = torch.flatten(torch.stack(coords), 1)
        relative_coords = coords[:, :, None] - coords[:, None, :]
        relative_coords[1] += self.seq_len - 1
        relative_coords = rearrange(relative_coords, 'c h w -> h w c')
        relative_index = relative_coords.sum(-1).flatten().unsqueeze(1)
        self.register_buffer("relative_index", relative_index)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.LayerNorm(emb_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        k = self.key(x).reshape(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 3, 1)
        v = self.value(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        q = self.query(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        # k,v,q shape = (batch_size, num_heads, seq_len, d_head)

        attn = torch.matmul(q, k) * self.scale
        # attn shape (seq_len, seq_len)
        attn = nn.functional.softmax(attn, dim=-1)

        # Use "gather" for more efficiency on GPUs
        relative_bias = self.relative_bias_table.gather(0, self.relative_index.repeat(1, 8))
        relative_bias = rearrange(relative_bias, '(h w) c -> 1 c h w', h=1 * self.seq_len, w=1 * self.seq_len)
        attn = attn + relative_bias

        # distance_pd = pd.DataFrame(relative_bias[0,0,:,:].cpu().detach().numpy())
        # distance_pd.to_csv('scalar_position_distance.csv')

        out = torch.matmul(attn, v)
        # out.shape = (batch_size, num_heads, seq_len, d_head)
        out = out.transpose(1, 2)
        # out.shape == (batch_size, seq_len, num_heads, d_head)
        out = out.reshape(batch_size, seq_len, -1)
        # out.shape == (batch_size, seq_len, d_model)
        out = self.to_out(out)
        return out


class Attention_Rel_Vec(nn.Module):
    def __init__(self, emb_size, num_heads, seq_len, dropout):
        super().__init__()
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.scale = emb_size ** -0.5
        # self.to_qkv = nn.Linear(inp, inner_dim * 3, bias=False)
        self.key = nn.Linear(emb_size, emb_size, bias=False)
        self.value = nn.Linear(emb_size, emb_size, bias=False)
        self.query = nn.Linear(emb_size, emb_size, bias=False)

        self.Er = nn.Parameter(torch.randn(self.seq_len, int(emb_size/num_heads)))

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(self.seq_len, self.seq_len))
            .unsqueeze(0).unsqueeze(0)
        )

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.LayerNorm(emb_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        k = self.key(x).reshape(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 3, 1)
        v = self.value(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        q = self.query(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        # k,v,q shape = (batch_size, num_heads, seq_len, d_head)

        QEr = torch.matmul(q, self.Er.transpose(0, 1))
        Srel = self.skew(QEr)
        # Srel.shape = (batch_size, self.num_heads, seq_len, seq_len)

        attn = torch.matmul(q, k)
        # attn shape (seq_len, seq_len)
        attn = (attn + Srel) * self.scale

        attn = nn.functional.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        # out.shape = (batch_size, num_heads, seq_len, d_head)
        out = out.transpose(1, 2)
        # out.shape == (batch_size, seq_len, num_heads, d_head)
        out = out.reshape(batch_size, seq_len, -1)
        # out.shape == (batch_size, seq_len, d_model)
        out = self.to_out(out)
        return out

    def skew(self, QEr):
        # QEr.shape = (batch_size, num_heads, seq_len, seq_len)
        padded = nn.functional.pad(QEr, (1, 0))
        # padded.shape = (batch_size, num_heads, seq_len, 1 + seq_len)
        batch_size, num_heads, num_rows, num_cols = padded.shape
        reshaped = padded.reshape(batch_size, num_heads, num_cols, num_rows)
        # reshaped.size = (batch_size, num_heads, 1 + seq_len, seq_len)
        Srel = reshaped[:, :, 1:, :]
        # Srel.shape = (batch_size, num_heads, seq_len, seq_len)
        return Srel


class tAPE(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=1024).
    """

    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0):
        super(tAPE, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)  # positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin((position * div_term)*(d_model/max_len))
        pe[:, 1::2] = torch.cos((position * div_term)*(d_model/max_len))
        pe = scale_factor * pe.unsqueeze(0)
        self.register_buffer('pe', pe)  # this stores the variable in the state_dict (used for non-trainable variables)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """
        x = x + self.pe
        return self.dropout(x)


class AbsolutePositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=1024).
    """

    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0):
        super(AbsolutePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)  # positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = scale_factor * pe.unsqueeze(0)
        self.register_buffer('pe', pe)  # this stores the variable in the state_dict (used for non-trainable variables)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """
        x = x + self.pe
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Each position gets its own embedding
        # Since indices are always 0 ... max_len, we don't have to do a look-up
        self.pe = nn.Parameter(torch.empty(max_len, d_model))  # requires_grad automatically set to True
        nn.init.uniform_(self.pe, -0.02, 0.02)

        # distance = torch.matmul(self.pe, self.pe[10])
        # import matplotlib.pyplot as plt

        # plt.plot(distance.detach().numpy())
        # plt.show()

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe
        # distance = torch.matmul(self.pe, self.pe.transpose(1,0))
        # distance_pd = pd.DataFrame(distance.cpu().detach().numpy())
        # distance_pd.to_csv('learn_position_distance.csv')
        return self.dropout(x)
    

def nt_xent_loss(z1, z2, temperature=0.5):
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


class ConvTran_HeadClassifier(nn.Module):
    def __init__(self, pretrained_encoder: ConvTranEncoder,
                 mode="finetune", hidden_dims_cls=256, drop_probability=0.5):
        super().__init__()
        self.encoder = pretrained_encoder
        self.mode = mode

        if mode == "probe":
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.encoder.eval()
            self.head = FC_Classifier(
                input_dim=self.encoder.flattened_dim,
                hidden_dims=hidden_dims_cls,
                drop_probability=drop_probability
            )
        else:
            self.head = FC_Classifier(
                input_dim=self.encoder.flattened_dim,
                hidden_dims=hidden_dims_cls,
                drop_probability=drop_probability
            )

    def forward(self, x):
        if self.mode == "probe":
            with torch.no_grad():
                emb = self.encoder.forward_features(x)
        else:
            emb = self.encoder.forward_features(x)
        return self.head(emb)


def pretrain_simclr_encoder_convtran(
    data,                       # numpy array [N, C, T]
    config,
    n_epochs=100,
    batch_size=256,
    lr=1e-3,
    temperature=0.5,
    proj_dim=128,
    aug_cfg=None,
    verbose_every=10
):
    """
    Returns a pretrained ConvTranEncoder (with projection head intact).
    Uses ALL provided 'data' as unlabeled for SSL.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = torch.tensor(data, dtype=torch.float32)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    encoder = ConvTranEncoder(config, proj_dim=proj_dim).to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)

    if aug_cfg is None:
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
            loss = nt_xent_loss(z1, z2, temperature=temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        if epoch % verbose_every == 0 or epoch == 1 or epoch == n_epochs:
            print(f"[SSL] Epoch {epoch}/{n_epochs} | Loss: {epoch_loss/len(loader):.4f}")

    return encoder


def evaluate_ConvTran_binary_with_optional_simclr(
    data, metadata,
    config,
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
    proj_dim=128
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if use_simclr_pretrain:
        print('[SimCLR pretraining on whole data]')
        encoder = pretrain_simclr_encoder_convtran(
            data=data,
            config=config,
            n_epochs=simclr_epochs,
            batch_size=simclr_batch_size,
            lr=simclr_lr,
            temperature=simclr_temperature,
            proj_dim=proj_dim,
            aug_cfg=augmentation,
            verbose_every=max(1, simclr_epochs // 10)
        )
    else:
        encoder = ConvTranEncoder(config, proj_dim=proj_dim)

    encoder.to(device)
    initial_state = {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()}
    results = {subset: {"acc": [], "f1": []} for subset in [5, 10, 20, 40]}

    for len_subset in [5, 10, 20, 40]:
        for seed in range(n_splits):
            print(f"\n===== Split {seed+1}/{n_splits} | subset={len_subset} =====")
            torch.manual_seed(seed)
            np.random.seed(seed)

            X_train, y_train, X_test, y_test = finetune_split(data, metadata, len_subset, seed=seed)
            encoder.load_state_dict({k: v.clone() for k, v in initial_state.items()})
            encoder.to(device)

            model = ConvTran_HeadClassifier(encoder, mode="probe" if linear_probe else "finetune").to(device)
            optimizer = torch.optim.Adam(
                model.head.parameters() if linear_probe else model.parameters(),
                lr=lr_supervised
            )

            X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
            y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
            X_test_t  = torch.tensor(X_test,  dtype=torch.float32).to(device)
            y_test_t  = torch.tensor(y_test,  dtype=torch.float32).view(-1, 1).to(device)

            loss_fn = nn.BCEWithLogitsLoss()
            if batch_size and batch_size > 0:
                train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
            else:
                train_loader = [(X_train_t, y_train_t)]

            best_loss, best_state, patience = float('inf'), None, 0
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
                    best_loss, best_state, patience = mean_loss, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
                else:
                    patience += 1

                if epoch % 20 == 0 or epoch == 1 or epoch == n_epochs_supervised:
                    print(f"[FT] Epoch {epoch}/{n_epochs_supervised} | TrainLoss {mean_loss:.4f}")

                if patience >= early_stopping_rounds:
                    print(f"[FT] Early stopping at epoch {epoch}")
                    break

            if best_state is not None:
                model.load_state_dict(best_state)

            # --- Test ---
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
            print(f"[Test] accuracy: {acc:.3f} | f1: {f1:.3f}")

    print("\n=== ConvTran SSL Results ===")
    for subset, metrics in results.items():
        print(f"Subset {subset:>2}: "
              f"Accuracy {np.mean(metrics['acc']):.3f} ± {np.std(metrics['acc']):.3f} | "
              f"F1 {np.mean(metrics['f1']):.3f} ± {np.std(metrics['f1']):.3f}")
