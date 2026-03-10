from utils._npy_manipulation import *
from ml_models._dummy_RF import *
from sklearn.model_selection import ParameterGrid

data_path = path_join('npy_file', 'data.npy')
data_npy = load(data_path)
print(f'data shape : {data_npy.shape}')

infos_path = path_join('npy_file', 'infos.npy')
infos_npy = load(infos_path)
print(f'infos_npy shape : {infos_npy.shape}')

data_normalised = normalise_per_bands(data_npy)

data_normalised_flatten = flatten_data(data_normalised)

aug_name = 'RF baseline'
print(f'\nTesting {aug_name}')
try:
    evaluate_RF_baseline(
        data_normalised_flatten, 
        infos_npy
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