from utils._npy_manipulation import *
from ml_models._tempCNN_SSL import *
from sklearn.model_selection import ParameterGrid

data_path = path_join('npy_file', 'data.npy')
data_npy = load(data_path)
print(f'data shape : {data_npy.shape}')

infos_path = path_join('npy_file', 'infos.npy')
infos_npy = load(infos_path)
print(f'infos_npy shape : {infos_npy.shape}')

# infos_npy[0] # plotid, gid, bc

# evaluate_TempCNN_binary_with_optional_simclr(
#     data_npy, infos_npy,
#     use_simclr_pretrain=True,           # set False for pure supervised
#     simclr_epochs=100,
#     simclr_batch_size=256,
#     n_epochs_supervised=200,
#     batch_size=256
# )

'''
TempCNN SSL gid
Mean Accuracy: 0.825 ± 0.060
Mean F1 Score: 0.892 ± 0.041
'''
'''
TempCNN SSL plotid
Mean Accuracy: 0.731 ± 0.132
Mean F1 Score: 0.825 ± 0.101
'''

param_grid = {
    # Pretraining SimCLR
    'simclr_batch_size': [128], 
    'simclr_lr': [1e-4, 1e-3],  

    # Fine-tuning supervised
    'finetune_batch_size': [128, 256, 512, 1024], 
    'finetune_lr': [1e-4, 1e-3],  
}

i=0
for params in ParameterGrid(param_grid):
    i+=1
    print(f'Testing params: {params}')
    try:
        evaluate_TempCNN_binary_with_optional_simclr(
            data_npy, infos_npy,
            use_simclr_pretrain=True,
            simclr_epochs=50,
            simclr_batch_size=params['simclr_batch_size'],
            simclr_lr=params['simclr_lr'],
            n_epochs_supervised=200,
            batch_size=params['finetune_batch_size'],
            lr_supervised=params['finetune_lr'],
        )
        print(f'Config {i} finished')
    except RuntimeError as e:
        if 'CUDA out of memory' in str(e):
            print(f'Config {i} failed with OOM')
        else:
            print(f'Config {i} failed with RuntimeError: {e}')
        torch.cuda.empty_cache()
    except Exception as e:
        print(f'Config {i} crashed with {type(e).__name__}: {e}')

'''
Testing params: {'finetune_batch_size': 128, 'finetune_lr': 0.0001, 'simclr_batch_size': 128, 'simclr_lr': 0.0001}
Mean Accuracy: 0.821 ± 0.041
Mean F1 Score: 0.892 ± 0.027

Testing params: {'finetune_batch_size': 128, 'finetune_lr': 0.0001, 'simclr_batch_size': 128, 'simclr_lr': 0.001}
Mean Accuracy: 0.813 ± 0.046
Mean F1 Score: 0.886 ± 0.030

Testing params: {'finetune_batch_size': 128, 'finetune_lr': 0.001, 'simclr_batch_size': 128, 'simclr_lr': 0.0001}
Mean Accuracy: 0.804 ± 0.051
Mean F1 Score: 0.878 ± 0.036

Testing params: {'finetune_batch_size': 128, 'finetune_lr': 0.001, 'simclr_batch_size': 128, 'simclr_lr': 0.001}
Mean Accuracy: 0.831 ± 0.039
Mean F1 Score: 0.898 ± 0.026

Testing params: {'finetune_batch_size': 256, 'finetune_lr': 0.0001, 'simclr_batch_size': 128, 'simclr_lr': 0.0001}
Mean Accuracy: 0.825 ± 0.041
Mean F1 Score: 0.894 ± 0.026

Testing params: {'finetune_batch_size': 256, 'finetune_lr': 0.0001, 'simclr_batch_size': 128, 'simclr_lr': 0.001}
Mean Accuracy: 0.813 ± 0.048
Mean F1 Score: 0.885 ± 0.032

Testing params: {'finetune_batch_size': 256, 'finetune_lr': 0.001, 'simclr_batch_size': 128, 'simclr_lr': 0.0001}
Mean Accuracy: 0.850 ± 0.026
Mean F1 Score: 0.911 ± 0.018

Testing params: {'finetune_batch_size': 256, 'finetune_lr': 0.001, 'simclr_batch_size': 128, 'simclr_lr': 0.001}
Mean Accuracy: 0.826 ± 0.035
Mean F1 Score: 0.896 ± 0.023

Testing params: {'finetune_batch_size': 512, 'finetune_lr': 0.0001, 'simclr_batch_size': 128, 'simclr_lr': 0.0001}
Mean Accuracy: 0.820 ± 0.044
Mean F1 Score: 0.892 ± 0.029

Testing params: {'finetune_batch_size': 512, 'finetune_lr': 0.0001, 'simclr_batch_size': 128, 'simclr_lr': 0.001}
Mean Accuracy: 0.796 ± 0.061
Mean F1 Score: 0.874 ± 0.041

Testing params: {'finetune_batch_size': 512, 'finetune_lr': 0.001, 'simclr_batch_size': 128, 'simclr_lr': 0.0001}
Mean Accuracy: 0.823 ± 0.049
Mean F1 Score: 0.893 ± 0.033

Testing params: {'finetune_batch_size': 512, 'finetune_lr': 0.001, 'simclr_batch_size': 128, 'simclr_lr': 0.001}
Mean Accuracy: 0.833 ± 0.044
Mean F1 Score: 0.901 ± 0.030

Testing params: {'finetune_batch_size': 1024, 'finetune_lr': 0.0001, 'simclr_batch_size': 128, 'simclr_lr': 0.0001}
Mean Accuracy: 0.827 ± 0.048
Mean F1 Score: 0.895 ± 0.033

Testing params: {'finetune_batch_size': 1024, 'finetune_lr': 0.0001, 'simclr_batch_size': 128, 'simclr_lr': 0.001}
Mean Accuracy: 0.806 ± 0.060
Mean F1 Score: 0.881 ± 0.040

Testing params: {'finetune_batch_size': 1024, 'finetune_lr': 0.001, 'simclr_batch_size': 128, 'simclr_lr': 0.0001}
Mean Accuracy: 0.831 ± 0.052
Mean F1 Score: 0.900 ± 0.034

Testing params: {'finetune_batch_size': 1024, 'finetune_lr': 0.001, 'simclr_batch_size': 128, 'simclr_lr': 0.001}
Mean Accuracy: 0.836 ± 0.045
Mean F1 Score: 0.903 ± 0.031

'''
# configs = [
#     {'simclr_batch_size': 1024, 'simclr_lr': 1e-2, 'finetune_batch_size': 512, 'finetune_lr': 1e-3},  # Big Bet
#     {'simclr_batch_size': 64,   'simclr_lr': 5e-2, 'finetune_batch_size': 16,  'finetune_lr': 1e-2},  # Tiny Chaos
#     {'simclr_batch_size': 256,  'simclr_lr': 1e-3, 'finetune_batch_size': 256, 'finetune_lr': 5e-5},  # Balanced Gambler
#     {'simclr_batch_size': 512,  'simclr_lr': 1e-4, 'finetune_batch_size': 512, 'finetune_lr': 1e-5},  # Slow & Steady
#     {'simclr_batch_size': 128,  'simclr_lr': 5e-2, 'finetune_batch_size': 32,  'finetune_lr': 1e-5},  # Wildcard
#     {'simclr_batch_size': 256,  'simclr_lr': 1e-3, 'finetune_batch_size': 64,  'finetune_lr': 1e-3},  # classic mid
#     {'simclr_batch_size': 128,  'simclr_lr': 1e-4, 'finetune_batch_size': 128, 'finetune_lr': 1e-4},  # smooth & stable
#     {'simclr_batch_size': 512,  'simclr_lr': 5e-4, 'finetune_batch_size': 256, 'finetune_lr': 1e-3},  # GPU-heavy, safe
#     {'simclr_batch_size': 64,   'simclr_lr': 1e-3, 'finetune_batch_size': 64,  'finetune_lr': 5e-4},  # light setup
#     {'simclr_batch_size': 256,  'simclr_lr': 5e-4, 'finetune_batch_size': 128, 'finetune_lr': 5e-4},  # balanced default
# ]

# for i, cfg in enumerate(configs, 1):
#     print(f'\nTesting config {i}/{len(configs)}: {cfg}')
#     try:
#         evaluate_TempCNN_binary_with_optional_simclr(
#             data_npy, infos_npy,
#             use_simclr_pretrain=True,
#             simclr_epochs=50,
#             simclr_batch_size=cfg['simclr_batch_size'],
#             simclr_lr=cfg['simclr_lr'],
#             n_epochs_supervised=200,
#             batch_size=cfg['finetune_batch_size'],
#             lr_supervised=cfg['finetune_lr'],
#         )
#         print(f'Config {i} finished')
#     except RuntimeError as e:
#         if 'CUDA out of memory' in str(e):
#             print(f'Config {i} failed with OOM')
#         else:
#             print(f'Config {i} failed with RuntimeError: {e}')
#         torch.cuda.empty_cache()
#     except Exception as e:
#         print(f'Config {i} crashed with {type(e).__name__}: {e}')