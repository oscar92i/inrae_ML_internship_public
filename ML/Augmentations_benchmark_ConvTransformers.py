from utils._npy_manipulation import *
from ml_models._ConvTransformers_SSL import *
from sklearn.model_selection import ParameterGrid

data_path = path_join('npy_file', 'data.npy')
data_npy = load(data_path)
print(f'data shape : {data_npy.shape}')

infos_path = path_join('npy_file', 'infos.npy')
infos_npy = load(infos_path)
print(f'infos_npy shape : {infos_npy.shape}')

augmentation_configs = {
    "jitter": {
        "augmentations": ["jitter"],
        "jitter": {"std": 0.05},
    },
    "random_crop": {
        "augmentations": ["random_crop"],
        "random_crop": {"min_crop": 0.7, "max_crop": 1.0},
    },
    "temporal_mask": {
        "augmentations": ["temporal_mask"],
        "temporal_mask": {"prob": 0.3, "span_ratio": 0.1},
    },
    "resampling": {
        "augmentations": ["resampling"],
        "resampling": {"upsampling_factor": 2.0, "subsequence_length_ratio": 0.5},
    },
    "all": {
        "augmentations": ["jitter", "random_crop", "temporal_mask", "resampling"],
        "jitter": {"std": 0.05}, "random_crop": {"min_crop": 0.7, "max_crop": 1.0}, "temporal_mask": {"prob": 0.3, "span_ratio": 0.1}, "resampling": {"upsampling_factor": 2.0, "subsequence_length_ratio": 0.5},
    },
}

config = {
'Data_shape': (10, 45),        # channel_size=10, seq_len=45
'emb_size': 64,
'num_heads': 8,
'dim_ff': 256,
'Fix_pos_encode': 'tAPE',   
'Rel_pos_encode': 'eRPE', 
'dropout': 0.1
}

#supervised
print('\n_-_-_-_-_-_-_-_-_-_-_-_')
print(('SUPERVISED'))
print('-_-_-_-_-_-_-_-_-_-_-_-')
aug_name = 'fully supervised'
print(f'\nTesting {aug_name}')
try:
    evaluate_ConvTran_binary_with_optional_simclr(
        data_npy, infos_npy,
        augmentation=None,
        use_simclr_pretrain=False,
        linear_probe=False,
        simclr_epochs=50,
        simclr_batch_size=128,
        simclr_lr=0.0001,
        n_epochs_supervised=100,
        batch_size=256,
        lr_supervised=0.001,
        config=config
    )
    print(f'Config {aug_name} finished')
except RuntimeError as e:
    if 'CUDA out of memory' in str(e):
        print(f'Config {aug_name} failed with OOM')
    else:
        print(f'Config {aug_name} failed with RuntimeError: {e}')
    torch.cuda.empty_cache()
except Exception as e:
    print(f'Config {aug_name} crashed with {type(e).__name__}: {e}')

#finetuning
print('\n_-_-_-_-_-_-_-_-_-_-_-_')
print(('FINETUNING'))
print('-_-_-_-_-_-_-_-_-_-_-_-')
for aug_name, aug_cfg in augmentation_configs.items():
    print(f'\nTesting params: {aug_name}')
    try:
        evaluate_ConvTran_binary_with_optional_simclr(
            data_npy, infos_npy,
            augmentation=aug_cfg,
            use_simclr_pretrain=True,
            linear_probe=False,
            simclr_epochs=400,
            simclr_batch_size=128,
            simclr_lr=0.0001,
            n_epochs_supervised=100,
            batch_size=256,
            lr_supervised=0.001,
            config=config
        )
        print(f'Config {aug_name} finished')
    except RuntimeError as e:
        if 'CUDA out of memory' in str(e):
            print(f'Config {aug_name} failed with OOM')
        else:
            print(f'Config {aug_name} failed with RuntimeError: {e}')
        torch.cuda.empty_cache()
    except Exception as e:
        print(f'Config {aug_name} crashed with {type(e).__name__}: {e}')

#linearprobing
print('\n_-_-_-_-_-_-_-_-_-_-_-_')
print(('LINEAR PROBING'))
print('-_-_-_-_-_-_-_-_-_-_-_-')
for aug_name, aug_cfg in augmentation_configs.items():
    print(f'\nTesting params: {aug_name}')
    try:
        evaluate_ConvTran_binary_with_optional_simclr(
            data_npy, infos_npy,
            augmentation=aug_cfg,
            use_simclr_pretrain=True,
            linear_probe=True,
            simclr_epochs=400,
            simclr_batch_size=128,
            simclr_lr=0.0001,
            n_epochs_supervised=100,
            batch_size=256,
            lr_supervised=0.001,
            config=config
        )
        print(f'Config {aug_name} finished')
    except RuntimeError as e:
        if 'CUDA out of memory' in str(e):
            print(f'Config {aug_name} failed with OOM')
        else:
            print(f'Config {aug_name} failed with RuntimeError: {e}')
        torch.cuda.empty_cache()
    except Exception as e:
        print(f'Config {aug_name} crashed with {type(e).__name__}: {e}')