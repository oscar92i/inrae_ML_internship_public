import numpy as np
from utils._npy_manipulation import *
import torch

print(np.__version__)

data_path = path_join('npy_file', 'data.npy')
data_npy = load(data_path)
print(f'data shape : {data_npy.shape}')

data_normalised = normalise_per_bands(data_npy)
print(f'data_normalised shape : {data_normalised.shape}')

infos_path = path_join('npy_file', 'infos.npy')
infos_npy = load(infos_path)
print(f'infos_npy shape : {infos_npy.shape}')

print("Unique GIDs:", np.unique(infos_npy[:,1]))
print("Num unique GIDs:", len(np.unique(infos_npy[:,1])))
print(infos_npy[:20, 1])

n_splits=5

for seed in range(n_splits):
    print(f'\n      Split {seed+1}/{n_splits}')
    torch.manual_seed(seed)

    train_split_target=0.6
    validation_split_target=0.2

    try:
        test_split_target = 1 - train_split_target - validation_split_target
    except (TypeError, ValueError):
        print('train_split_target and validation_split_target must be float')

    assert np.isclose(train_split_target + validation_split_target + test_split_target, 1.0), 'total split must sum to 1.'

    np.random.seed(seed)

    gids, inverse_indices = np.unique(infos_npy[:, 1], return_inverse=True)
    gid_to_pixel_indices = defaultdict(list)

    for idx, inv in enumerate(inverse_indices): # idx is gid of pixel, inv is the index into the gids array
        gid_to_pixel_indices[gids[inv]].append(idx) # grouping all pixel indices under their gid
    gid_to_pixel_indices = {k: np.array(v) for k, v in gid_to_pixel_indices.items()} # convert each list of pixel indices to np array for perf

    total_pixels = infos_npy.shape[0]
    train_target = train_split_target * total_pixels
    val_target = validation_split_target * total_pixels
    test_target = test_split_target * total_pixels

    shuffled_gids = gids.copy()
    np.random.shuffle(shuffled_gids) # deterministic due to np.random.seed(seed)
    print("First 10 shuffled GIDs:", shuffled_gids[:10])
    
    train_gids, val_gids, test_gids = [], [], []
    pixel_counts = {'train': 0, 'val': 0, 'test': 0}

    for gid in shuffled_gids:
        n_pixels = len(gid_to_pixel_indices[gid])
        gaps = {
            'train': train_target - pixel_counts['train'],
            'val': val_target - pixel_counts['val'],
            'test': test_target - pixel_counts['test']
        }
        best_split = max(gaps, key=gaps.get)

        if best_split == 'train':
            train_gids.append(gid)
            pixel_counts['train'] += n_pixels
        elif best_split == 'val':
            val_gids.append(gid)
            pixel_counts['val'] += n_pixels
        else:
            test_gids.append(gid)
            pixel_counts['test'] += n_pixels

    print(f'train gids len : {len(train_gids)}')
    print(f'val gids len : {len(val_gids)}')
    print(f'test gids len : {len(test_gids)}')

    train_idx = np.concatenate([gid_to_pixel_indices[gid] for gid in train_gids]) # list of idx
    val_idx = np.concatenate([gid_to_pixel_indices[gid] for gid in val_gids]) # list of idx
    test_idx = np.concatenate([gid_to_pixel_indices[gid] for gid in test_gids]) # list of idx

    X_train = data_normalised[train_idx]
    X_val = data_normalised[val_idx]
    X_test = data_normalised[test_idx]

    y_train = infos_npy[train_idx, 2].astype(np.int8) 
    y_val = infos_npy[val_idx, 2].astype(np.int8)
    y_test = infos_npy[test_idx, 2].astype(np.int8)

    len_total = len(train_idx) + len(val_idx) + len(test_idx)
    print(f'Train split: {len(train_idx)/len_total:.3f}')
    print(f'Val split: {len(val_idx)/len_total:.3f}')
    print(f'Test split: {len(test_idx)/len_total:.3f}')