from utils._npy_manipulation import *
from ml_models._InceptionTime import *
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

param_grid = {
    'batch_size': [16, 32, 64, 128],
    'lr': [1e-2, 1e-3, 1e-4]
}

for params in ParameterGrid(param_grid):
    print('\n-----------------------------')
    print(f'Testing params: {params}')
    evaluate_Inception_classification(
        data_normalised, 
        infos_npy, 
        nb_classes,
        train_split_target=0.6, 
        validation_split_target=0.2,
        batch_size=params['batch_size'],
        lr=params['lr'])
    
'''
BEST MODEL
Testing params: {'batch_size': 32, 'lr': 0.0001}
Inception Classification
Mean Accuracy: 0.831 ± 0.038
Mean F1 Score: 0.897 ± 0.024
'''

'''
python InceptionTime.py
data shape : (19811, 10, 45)
data_normalised shape : (19811, 10, 45)
infos_npy shape : (19811, 3)

-----------------------------
Testing params: {'batch_size': 16, 'lr': 0.01}
Inception Classification
Mean Accuracy: 0.807 ± 0.013
Mean F1 Score: 0.881 ± 0.008

-----------------------------
Testing params: {'batch_size': 16, 'lr': 0.001}
Inception Classification
Mean Accuracy: 0.813 ± 0.046
Mean F1 Score: 0.885 ± 0.031

-----------------------------
Testing params: {'batch_size': 16, 'lr': 0.0001}
Inception Classification
Mean Accuracy: 0.800 ± 0.054
Mean F1 Score: 0.876 ± 0.036

-----------------------------
Testing params: {'batch_size': 32, 'lr': 0.01}
Inception Classification
Mean Accuracy: 0.829 ± 0.051
Mean F1 Score: 0.895 ± 0.033

-----------------------------
Testing params: {'batch_size': 32, 'lr': 0.001}
Inception Classification
Mean Accuracy: 0.817 ± 0.056
Mean F1 Score: 0.885 ± 0.037

-----------------------------
Testing params: {'batch_size': 32, 'lr': 0.0001}
Inception Classification
Mean Accuracy: 0.831 ± 0.038
Mean F1 Score: 0.897 ± 0.024

-----------------------------
Testing params: {'batch_size': 64, 'lr': 0.01}
Inception Classification
Mean Accuracy: 0.831 ± 0.034
Mean F1 Score: 0.896 ± 0.023

-----------------------------
Testing params: {'batch_size': 64, 'lr': 0.001}
Inception Classification
Mean Accuracy: 0.829 ± 0.038
Mean F1 Score: 0.897 ± 0.025

-----------------------------
Testing params: {'batch_size': 64, 'lr': 0.0001}
Inception Classification
Mean Accuracy: 0.831 ± 0.049
Mean F1 Score: 0.897 ± 0.032

-----------------------------
Testing params: {'batch_size': 128, 'lr': 0.01}
Inception Classification
Mean Accuracy: 0.811 ± 0.037
Mean F1 Score: 0.883 ± 0.027

-----------------------------
Testing params: {'batch_size': 128, 'lr': 0.001}
Inception Classification
Mean Accuracy: 0.809 ± 0.061
Mean F1 Score: 0.882 ± 0.041

-----------------------------
Testing params: {'batch_size': 128, 'lr': 0.0001}
Inception Classification
Mean Accuracy: 0.826 ± 0.027
Mean F1 Score: 0.894 ± 0.017
'''