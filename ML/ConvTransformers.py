from utils._npy_manipulation import *
from ml_models._ConvTransformerGPU import *
from sklearn.model_selection import ParameterGrid

data_path = path_join('npy_file', 'data.npy')
data_npy = load(data_path)
print(f'data shape : {data_npy.shape}')

data_normalised = normalise_per_bands(data_npy)
print(f'data_normalised shape : {data_normalised.shape}')

infos_path = path_join('npy_file', 'infos.npy')
infos_npy = load(infos_path)
print(f'infos_npy shape : {infos_npy.shape}')

# infos_npy[0] # plotid, gid, bc

nb_classes = len(np.unique(infos_npy[:, 2]))

#evaluate_ConvTran_binary(data_normalised, infos_npy, nb_classes)
#evaluate_ConvTran_binary_batch(data_normalised, infos_npy, train_split_size=0.6, validation_split_size=0.2)

param_grid = {
    'batch_size': [16, 32, 64, 128],
    'lr': [1e-2, 1e-3, 1e-4]
}

for params in ParameterGrid(param_grid):
    print(f'Testing params: {params}')
    evaluate_ConvTran_binary_batch(
        data_normalised, 
        infos_npy, 
        train_split_size=0.6, 
        validation_split_size=0.2,
        batch_size=params['batch_size'],
        lr=params['lr'])
    
'''
BEST PARAMETER
Testing params: {'batch_size': 32, 'lr': 0.0001}
ConvTran Binary Classification Batch 32
Mean Accuracy: 0.820 ± 0.072
Mean F1 Score: 0.887 ± 0.049
'''

'''
python ConvTransformers.py

data shape : (19811, 10, 45)
data_normalised shape : (19811, 10, 45)
infos_npy shape : (19811, 3)

Testing params: {'batch_size': 16, 'lr': 0.01}
ConvTran Binary Classification Batch 16
Mean Accuracy: 0.810 ± 0.047
Mean F1 Score: 0.881 ± 0.033
 
Testing params: {'batch_size': 16, 'lr': 0.001}
ConvTran Binary Classification Batch 16
Mean Accuracy: 0.802 ± 0.051
Mean F1 Score: 0.876 ± 0.034

Testing params: {'batch_size': 16, 'lr': 0.0001}
ConvTran Binary Classification Batch 16
Mean Accuracy: 0.791 ± 0.079
Mean F1 Score: 0.867 ± 0.056

Testing params: {'batch_size': 32, 'lr': 0.01}
ConvTran Binary Classification Batch 32
Mean Accuracy: 0.807 ± 0.048
Mean F1 Score: 0.880 ± 0.035

Testing params: {'batch_size': 32, 'lr': 0.001}
ConvTran Binary Classification Batch 32
Mean Accuracy: 0.806 ± 0.055
Mean F1 Score: 0.879 ± 0.036

Testing params: {'batch_size': 32, 'lr': 0.0001}
ConvTran Binary Classification Batch 32
Mean Accuracy: 0.820 ± 0.072
Mean F1 Score: 0.887 ± 0.049

Testing params: {'batch_size': 64, 'lr': 0.01}
ConvTran Binary Classification Batch 64
Mean Accuracy: 0.809 ± 0.052
Mean F1 Score: 0.881 ± 0.032

Testing params: {'batch_size': 64, 'lr': 0.001}
ConvTran Binary Classification Batch 64
Mean Accuracy: 0.816 ± 0.047
Mean F1 Score: 0.886 ± 0.031

Testing params: {'batch_size': 64, 'lr': 0.0001}
ConvTran Binary Classification Batch 64
Mean Accuracy: 0.800 ± 0.042
Mean F1 Score: 0.875 ± 0.029

Testing params: {'batch_size': 128, 'lr': 0.01}
ConvTran Binary Classification Batch 128
Mean Accuracy: 0.808 ± 0.063
Mean F1 Score: 0.880 ± 0.041

Testing params: {'batch_size': 128, 'lr': 0.001}
ConvTran Binary Classification Batch 128
Mean Accuracy: 0.797 ± 0.098
Mean F1 Score: 0.870 ± 0.071

Testing params: {'batch_size': 128, 'lr': 0.0001}
ConvTran Binary Classification Batch 128
Mean Accuracy: 0.804 ± 0.051
Mean F1 Score: 0.877 ± 0.035
'''