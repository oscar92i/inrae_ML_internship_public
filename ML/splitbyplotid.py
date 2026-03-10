from utils._npy_manipulation import *
from ml_models._ConvTransformerGPU import *
from ml_models._InceptionTime import *
from ml_models._MLP import *
from ml_models._tempCNN import *
from ml_models._RF_XGBoost import *

data_path = path_join('npy_file', 'data.npy')
data_npy = load(data_path)
print(f'data shape : {data_npy.shape}')

data_normalised = normalise_per_bands(data_npy)
print(f'data_normalised shape : {data_normalised.shape}')

infos_path = path_join('npy_file', 'infos.npy')
infos_npy = load(infos_path)
print(f'infos_npy shape : {infos_npy.shape}')

data_normalised_flatten = flatten_data(data_normalised)
print(f'data_normalised_flatten shape : {data_normalised_flatten.shape}')

# infos_npy[0] # plotid, gid, bc

nb_classes = len(np.unique(infos_npy[:, 2]))


evaluate_ConvTran_binary_batch(
    data_normalised, 
    infos_npy, 
    train_split_size=0.6, 
    validation_split_size=0.2,
    batch_size=32,
    lr=0.0001)

evaluate_TempCNN_binary(
    data_normalised, 
    infos_npy, 
    train_split_target=0.6, 
    validation_split_target=0.2,
    batch_size=64,
    lr=0.0001)

evaluate_Inception_classification(
    data_normalised, 
    infos_npy, 
    nb_classes,
    train_split_target=0.6, 
    validation_split_target=0.2,
    batch_size=32,
    lr=0.0001)

evaluate_MLP_CE(
    data_normalised_flatten,
    infos_npy,
    batch_size=64,
    lr=0.001
)

evaluate_random_forest(data_normalised_flatten, infos_npy, train_split_target=0.6, validation_split_target=0.2)

evaluate_xgboost(data_normalised_flatten, infos_npy, train_split_target=0.6, validation_split_target=0.2)

'''
python splitbyplotid.py
data shape : (19811, 10, 45)
data_normalised shape : (19811, 10, 45)
infos_npy shape : (19811, 3)
data_normalised_flatten shape : (19811, 450)

ConvTran Binary Classification Batch 32 0.0001
Mean Accuracy: 0.700 ± 0.105
Mean F1 Score: 0.805 ± 0.080

TempCNN Binary Classification Batch 64 0.0001
Mean Accuracy: 0.763 ± 0.121
Mean F1 Score: 0.844 ± 0.094

Inception Classification Batch 32 
Mean Accuracy: 0.745 ± 0.125
Mean F1 Score: 0.833 ± 0.099

Multi_Layer Perceptron
Mean Accuracy: 0.809 ± 0.039
Mean F1 Score: 0.882 ± 0.024

RandomForest
mean accuracy: 0.721 ± 0.151
mean f1 score: 0.489 ± 0.124

XGBoost
mean accuracy: 0.729 ± 0.156
mean f1 score: 0.512 ± 0.134
'''