from utils._npy_manipulation import *
from ml_models._tempCNN import *
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

param_grid = {
    'batch_size': [16, 32, 64, 128],
    'lr': [1e-2, 1e-3, 1e-4]
}

for params in ParameterGrid(param_grid):
    print('\n-----------------------------')
    print(f'Testing params: {params}')
    evaluate_TempCNN_binary(
        data_normalised, 
        infos_npy, 
        train_split_target=0.6, 
        validation_split_target=0.2,
        batch_size=params['batch_size'],
        lr=params['lr'])
    
'''
BEST MODEL 
Testing params: {'batch_size': 64, 'lr': 0.0001}
TempCNN Binary Classification 64 0.0001
Mean Accuracy: 0.837 ± 0.035
Mean F1 Score: 0.901 ± 0.021
'''

'''
python TempCNN.py
data shape : (19811, 10, 45)
data_normalised shape : (19811, 10, 45)
infos_npy shape : (19811, 3)

-----------------------------
Testing params: {'batch_size': 16, 'lr': 0.01}
TempCNN Binary Classification 16 0.01
Mean Accuracy: 0.822 ± 0.054
Mean F1 Score: 0.889 ± 0.036

-----------------------------
Testing params: {'batch_size': 16, 'lr': 0.001}
TempCNN Binary Classification 16 0.001
Mean Accuracy: 0.811 ± 0.053
Mean F1 Score: 0.883 ± 0.035

-----------------------------
Testing params: {'batch_size': 16, 'lr': 0.0001}
TempCNN Binary Classification 16 0.0001
Mean Accuracy: 0.820 ± 0.042
Mean F1 Score: 0.890 ± 0.027

-----------------------------
Testing params: {'batch_size': 32, 'lr': 0.01}
TempCNN Binary Classification 32 0.01
Mean Accuracy: 0.830 ± 0.045
Mean F1 Score: 0.896 ± 0.029

-----------------------------
Testing params: {'batch_size': 32, 'lr': 0.001}
TempCNN Binary Classification 32 0.001
Mean Accuracy: 0.826 ± 0.050
Mean F1 Score: 0.893 ± 0.033

-----------------------------
Testing params: {'batch_size': 32, 'lr': 0.0001}
TempCNN Binary Classification 32 0.0001
Mean Accuracy: 0.835 ± 0.029
Mean F1 Score: 0.900 ± 0.019

-----------------------------
Testing params: {'batch_size': 64, 'lr': 0.01}
TempCNN Binary Classification 64 0.01
Mean Accuracy: 0.828 ± 0.056
Mean F1 Score: 0.895 ± 0.036

-----------------------------
Testing params: {'batch_size': 64, 'lr': 0.001}
TempCNN Binary Classification 64 0.001
Mean Accuracy: 0.823 ± 0.039
Mean F1 Score: 0.892 ± 0.025

-----------------------------
Testing params: {'batch_size': 64, 'lr': 0.0001}
TempCNN Binary Classification 64 0.0001
Mean Accuracy: 0.837 ± 0.035
Mean F1 Score: 0.901 ± 0.021

-----------------------------
Testing params: {'batch_size': 128, 'lr': 0.01}
TempCNN Binary Classification 128 0.01
Mean Accuracy: 0.810 ± 0.062
Mean F1 Score: 0.884 ± 0.040

-----------------------------
Testing params: {'batch_size': 128, 'lr': 0.001}
TempCNN Binary Classification 128 0.001
Mean Accuracy: 0.828 ± 0.040
Mean F1 Score: 0.895 ± 0.026

-----------------------------
Testing params: {'batch_size': 128, 'lr': 0.0001}
TempCNN Binary Classification 128 0.0001
Mean Accuracy: 0.834 ± 0.033
Mean F1 Score: 0.899 ± 0.020'''