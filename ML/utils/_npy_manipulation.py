import sys
import os

import numpy as np
from pprint import pprint
from collections import defaultdict


def path_join(path1, path2):
    return os.path.join(path1, path2)


def _sync(fh):
    fh.flush() # internal buffer to os
    os.fsync(fh.fileno()) # os forced to write to disk


def save(array, pth):
    with open(pth, 'wb+') as fh: # wb because binary
        np.save(fh, array, allow_pickle=False) # allow_pickle=False because we are not saving object like gdf so not needed also safer (can run scripts randomly)
        _sync(fh)


def load(pth):
    with open(pth, 'rb') as fh:  # rb because binary
        return np.load(fh, allow_pickle=False)
    

def normalise_per_bands(data_array, band_names=None, percentiles=(2, 98)):
    normalised_array = data_array.copy()

    if band_names==None:
        for i in range(0, data_array.shape[1]):
            data_band = normalised_array[:, i, :]

            min_band, max_band = np.percentile(data_band, percentiles)

            clipped = np.clip(data_band, min_band, max_band)
            normalised = (clipped - min_band) / (max_band - min_band) # Subtracting min_band in both the numerator and denominator shifts the lowest value at 0

            normalised_array[:, i, :] = normalised        

    else:
        for i, band in enumerate(band_names):
            data_band = normalised_array[:, i, :]

            min_band, max_band = np.percentile(data_band, percentiles)

            clipped = np.clip(data_band, min_band, max_band)
            normalised = (clipped - min_band) / (max_band - min_band) # Subtracting min_band in both the numerator and denominator shifts the lowest value at 0

            normalised_array[:, i, :] = normalised

    return normalised_array


def flatten_data(cube):
    return cube.reshape(cube.shape[0], -1)


def train_validation_test_split_by_gid(data, metadata, train_split_target: float, validation_split_target: float, seed=42, print_len_splits=False):
    """
    Split data and metadata by pixel count, grouping all pixels from the same gid together. Split targets are not exactly fulfilled it is an approximation
    """
    try:
        test_split_target = 1 - train_split_target - validation_split_target
    except (TypeError, ValueError):
        print('train_split_target and validation_split_target must be float')

    assert np.isclose(train_split_target + validation_split_target + test_split_target, 1.0), 'total split must sum to 1.'

    np.random.seed(seed)

    gids, inverse_indices = np.unique(metadata[:, 1], return_inverse=True)
    gid_to_pixel_indices = defaultdict(list)

    for idx, inv in enumerate(inverse_indices): # idx is gid of pixel, inv is the index into the gids array
        gid_to_pixel_indices[gids[inv]].append(idx) # grouping all pixel indices under their gid
    gid_to_pixel_indices = {k: np.array(v) for k, v in gid_to_pixel_indices.items()} # convert each list of pixel indices to np array for perf

    total_pixels = metadata.shape[0]
    train_target = train_split_target * total_pixels
    val_target = validation_split_target * total_pixels
    test_target = test_split_target * total_pixels

    shuffled_gids = gids.copy()
    np.random.shuffle(shuffled_gids) # deterministic due to np.random.seed(seed)

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

    train_idx = np.concatenate([gid_to_pixel_indices[gid] for gid in train_gids]) # list of idx
    val_idx = np.concatenate([gid_to_pixel_indices[gid] for gid in val_gids]) # list of idx
    test_idx = np.concatenate([gid_to_pixel_indices[gid] for gid in test_gids]) # list of idx

    X_train = data[train_idx]
    X_val = data[val_idx]
    X_test = data[test_idx]

    y_train = metadata[train_idx, 2].astype(np.int8) 
    y_val = metadata[val_idx, 2].astype(np.int8)
    y_test = metadata[test_idx, 2].astype(np.int8)

    if print_len_splits:
        len_total = len(train_idx) + len(val_idx) + len(test_idx)
        print(f'Train split: {len(train_idx)/len_total:.3f}')
        print(f'Val split: {len(val_idx)/len_total:.3f}')
        print(f'Test split: {len(test_idx)/len_total:.3f}')

    return X_train, X_val, X_test, y_train, y_val, y_test


def train_validation_test_split_by_plotid(data, metadata, train_split_target: float, validation_split_target: float, seed=42, print_len_splits=False):
    """
    Split data and metadata by pixel count, grouping all pixels from the same plotid together. Split targets are not exactly fulfilled it is an approximation
    """
    try:
        test_split_target = 1 - train_split_target - validation_split_target
    except (TypeError, ValueError):
        print('train_split_target and validation_split_target must be float')

    assert np.isclose(train_split_target + validation_split_target + test_split_target, 1.0), 'total split must sum to 1.'

    np.random.seed(seed)

    plotids, inverse_indices = np.unique(metadata[:, 0], return_inverse=True)
    plotid_to_pixel_indices = defaultdict(list)

    for idx, inv in enumerate(inverse_indices): # idx is plotid of pixel, inv is the index into the plotids array
        plotid_to_pixel_indices[plotids[inv]].append(idx) # grouping all pixel indices under their plotid
    plotid_to_pixel_indices = {k: np.array(v) for k, v in plotid_to_pixel_indices.items()} # convert each list of pixel indices to np array for perf

    total_pixels = metadata.shape[0]
    train_target = train_split_target * total_pixels
    val_target = validation_split_target * total_pixels
    test_target = test_split_target * total_pixels

    shuffled_plotids = plotids.copy()
    np.random.shuffle(shuffled_plotids) # deterministic due to np.random.seed(seed)

    train_plotids, val_plotids, test_plotids = [], [], []
    pixel_counts = {'train': 0, 'val': 0, 'test': 0}

    for plotid in shuffled_plotids:
        n_pixels = len(plotid_to_pixel_indices[plotid])
        gaps = {
            'train': train_target - pixel_counts['train'],
            'val': val_target - pixel_counts['val'],
            'test': test_target - pixel_counts['test']
        }
        best_split = max(gaps, key=gaps.get)

        if best_split == 'train':
            train_plotids.append(plotid)
            pixel_counts['train'] += n_pixels
        elif best_split == 'val':
            val_plotids.append(plotid)
            pixel_counts['val'] += n_pixels
        else:
            test_plotids.append(plotid)
            pixel_counts['test'] += n_pixels

    train_idx = np.concatenate([plotid_to_pixel_indices[plotid] for plotid in train_plotids]) # list of idx
    val_idx = np.concatenate([plotid_to_pixel_indices[plotid] for plotid in val_plotids]) # list of idx
    test_idx = np.concatenate([plotid_to_pixel_indices[plotid] for plotid in test_plotids]) # list of idx

    X_train = data[train_idx]
    X_val = data[val_idx]
    X_test = data[test_idx]

    y_train = metadata[train_idx, 2].astype(np.int8) 
    y_val = metadata[val_idx, 2].astype(np.int8)
    y_test = metadata[test_idx, 2].astype(np.int8)

    if print_len_splits:
        len_total = len(train_idx) + len(val_idx) + len(test_idx)
        print(f'Train split: {len(train_idx)/len_total:.3f}')
        print(f'Val split: {len(val_idx)/len_total:.3f}')
        print(f'Test split: {len(test_idx)/len_total:.3f}')

    return X_train, X_val, X_test, y_train, y_val, y_test


def train_test_split_by_gid(data, metadata, train_split_target=0.7, seed=42, print_len_splits=False):
    """
    Split data and metadata by pixel count, grouping all pixels from the same gid together. Split targets are not exactly fulfilled it is an approximation
    """
    try:
        test_split_target = 1 - train_split_target
    except (TypeError, ValueError):
        print('train_split_target must be float')

    assert np.isclose(train_split_target + test_split_target, 1.0), 'total split must sum to 1.'
    
    np.random.seed(seed)

    gids, inverse_indices = np.unique(metadata[:, 1], return_inverse=True)
    gid_to_pixel_indices = defaultdict(list)

    for idx, inv in enumerate(inverse_indices): # idx is gid of pixel, inv is the index into the gids array
        gid_to_pixel_indices[gids[inv]].append(idx) # grouping all pixel indices under their gid
    gid_to_pixel_indices = {k: np.array(v) for k, v in gid_to_pixel_indices.items()} # convert each list of pixel indices to np array for perf

    total_pixels = metadata.shape[0]
    train_target = train_split_target * total_pixels
    test_target = (1-train_split_target) * total_pixels

    shuffled_gids = gids.copy()
    np.random.shuffle(shuffled_gids) # deterministic due to np.random.seed(seed)

    train_gids, test_gids = [], []
    pixel_counts = {'train': 0, 'test': 0}

    for gid in shuffled_gids:
        n_pixels = len(gid_to_pixel_indices[gid])
        gaps = {
            'train': train_target - pixel_counts['train'],
            'test': test_target - pixel_counts['test']
        }
        best_split = max(gaps, key=gaps.get)

        if best_split == 'train':
            train_gids.append(gid)
            pixel_counts['train'] += n_pixels
        else:
            test_gids.append(gid)
            pixel_counts['test'] += n_pixels

    train_idx = np.concatenate([gid_to_pixel_indices[gid] for gid in train_gids]) # list of idx
    test_idx = np.concatenate([gid_to_pixel_indices[gid] for gid in test_gids]) # list of idx

    X_train = data[train_idx]
    X_test = data[test_idx]

    y_train = metadata[train_idx, 2].astype(np.int8) 
    y_test = metadata[test_idx, 2].astype(np.int8)

    if print_len_splits:
        len_total = len(total_pixels)
        print(f'Train split: {len(train_idx)/len_total:.3f}')
        print(f'Test split: {len(test_idx)/len_total:.3f}')

    return X_train, X_test, y_train, y_test


def train_test_split_by_plotid(data, metadata, train_split_target=0.7, seed=42, print_len_splits=False):
    """
    Split data and metadata by pixel count, grouping all pixels from the same gid together. Split targets are not exactly fulfilled it is an approximation
    """
    try:
        test_split_target = 1 - train_split_target
    except (TypeError, ValueError):
        print('train_split_target must be float')

    assert np.isclose(train_split_target + test_split_target, 1.0), 'total split must sum to 1.'
    
    np.random.seed(seed)

    plotids, inverse_indices = np.unique(metadata[:, 0], return_inverse=True)
    plotid_to_pixel_indices = defaultdict(list)

    for idx, inv in enumerate(inverse_indices): # idx is gid of pixel, inv is the index into the gids array
        plotid_to_pixel_indices[plotids[inv]].append(idx) # grouping all pixel indices under their gid
    plotid_to_pixel_indices = {k: np.array(v) for k, v in plotid_to_pixel_indices.items()} # convert each list of pixel indices to np array for perf

    total_pixels = metadata.shape[0]
    train_target = train_split_target * total_pixels
    test_target = (1-train_split_target) * total_pixels

    shuffled_plotids = plotids.copy()
    np.random.shuffle(shuffled_plotids) # deterministic due to np.random.seed(seed)

    train_plotids, test_plotids = [], []
    pixel_counts = {'train': 0, 'test': 0}

    for plotid in shuffled_plotids:
        n_pixels = len(plotid_to_pixel_indices[plotid])
        gaps = {
            'train': train_target - pixel_counts['train'],
            'test': test_target - pixel_counts['test']
        }
        best_split = max(gaps, key=gaps.get)

        if best_split == 'train':
            train_plotids.append(plotid)
            pixel_counts['train'] += n_pixels
        else:
            test_plotids.append(plotid)
            pixel_counts['test'] += n_pixels

    train_idx = np.concatenate([plotid_to_pixel_indices[plotid] for plotid in train_plotids]) # list of idx
    test_idx = np.concatenate([plotid_to_pixel_indices[plotid] for plotid in test_plotids]) # list of idx

    X_train = data[train_idx]
    X_test = data[test_idx]

    y_train = metadata[train_idx, 2].astype(np.int8) 
    y_test = metadata[test_idx, 2].astype(np.int8)

    if print_len_splits:
        len_total = len(total_pixels)
        print(f'Train split: {len(train_idx)/len_total:.3f}')
        print(f'Test split: {len(test_idx)/len_total:.3f}')

    return X_train, X_test, y_train, y_test


def finetune_split(data, metadata, len_subset: int, seed=42, print_class=False):
    rng = np.random.default_rng(seed)

    mask_0 = metadata[:, 2] == 0
    mask_1 = metadata[:, 2] == 1

    class_0_data, class_0_meta = data[mask_0], metadata[mask_0]
    class_1_data, class_1_meta = data[mask_1], metadata[mask_1]

    if len_subset > len(class_0_data) or len_subset > len(class_1_data):
        raise ValueError(f'len subset {len_subset} too big')

    idx_0 = rng.choice(len(class_0_data), size=len_subset, replace=False)
    idx_1 = rng.choice(len(class_1_data), size=len_subset, replace=False)

    X_train_0, y_train_0 = class_0_data[idx_0], np.zeros(len_subset, dtype=int)
    X_train_1, y_train_1 = class_1_data[idx_1], np.ones(len_subset, dtype=int)

    X_train = np.vstack([X_train_0, X_train_1])
    y_train = np.concatenate([y_train_0, y_train_1])

    mask_0_rest = np.ones(len(class_0_data), dtype=bool); mask_0_rest[idx_0] = False
    mask_1_rest = np.ones(len(class_1_data), dtype=bool); mask_1_rest[idx_1] = False

    X_test = np.vstack([class_0_data[mask_0_rest], class_1_data[mask_1_rest]])
    y_test = np.concatenate([
        np.zeros(mask_0_rest.sum(), dtype=int),
        np.ones(mask_1_rest.sum(), dtype=int)
    ])

    if print_class:
        print(f"[Train] class0={len(y_train_0)} | class1={len(y_train_1)}")
        print(f"[Test ] class0={mask_0_rest.sum()} | class1={mask_1_rest.sum()}")

    # if print_class:
    #         print(f'subset meta unique values : {np.unique(subset_0_meta)} | {np.unique(subset_1_meta)}')
    #         assert all(len(x) == len_subset for x in [subset_0_data, subset_0_meta, subset_1_data, subset_1_meta]), f'Subset len error, expected {len_subset}, got {[len(x) for x in [subset_0_data, subset_0_meta, subset_1_data, subset_1_meta]]}'

    return X_train, y_train, X_test, y_test
