# Cloud ML parameters
ml_pythonVersion = '3.5'
ml_runtimeVersion = '1.10'
ml_region = 'europe-west1'
# Useful for preprocessing and predict, not for training. ml_scalerTier_train is used for training
# ml_scaleTier = 'BASIC'
ml_preprocess = {'ml_scaleTier_train': 'CUSTOM',
                 'ml_masterType': 'large_model'}
ml_predict = {'ml_scaleTier_train': 'CUSTOM',
              'ml_masterType': 'large_model'}
typology_machine = {'S': {'ml_scaleTier_train': 'CUSTOM',
                          'ml_masterType': 'n1-standard-8',
                          'ml_workerType': 'large_model',
                          'ml_workerCount': 0,
                          'ml_parameterServerType': 'standard',
                          'ml_parameterServerCount': 0},
                    'M': {'ml_scaleTier_train': 'CUSTOM',
                          'ml_workerCount': 3,
                          'ml_workerType': 'standard_gpu',
                          'ml_masterType': 'standard_gpu',
                          'ml_parameterServerCount': 1,
                          'ml_parameterServerType': 'standard_gpu'},
                    'L': {'ml_scaleTier_train': 'CUSTOM',
                          'ml_workerCount': 10,
                          'ml_workerType': 'standard_gpu',
                          'ml_masterType': 'standard_gpu',
                          'ml_parameterServerCount': 3,
                          'ml_parameterServerType': 'standard_gpu'}
                     }
train_ratio = 0.75
random_seed = 6

augment = True
deep_analysis = False
dump_model = True
dump_attribution_results = False
to_preprocess = True

model_param = {'hidden_dim': 128,
               'bucket': [1, 2, 3, 4, 13, 30],
               'batch_dim': 64,
               'input_dim': 2,
               'target_dim': 2,
               'embedding_dim': 150,
               'learning_rate': 0.001,
               'layers': 2,
               'p_dropout': 0.2,
               'epochs': 10}

vocabulary_size = {'channel_init': 0,
                   'env_first_url': 0,
                   'env_last_url': 0}

idx_to_embed = []

features_to_encode = ['channel_init', 'env_first_url', 'env_last_url']

saving_folder = 'attribution_model/'
model_folder = 'attribution_model/model_checkpoint_e10'

working_directory = 'att_comportementale'
local_dir = '/tmp/'
path_data = '/data/'
path_results = '/results/'
path_model = '/model/'
path_preproc = '/preprocessing/'

augmented_data = 'augmented_data.pkl'
# dataset_name = 'full_small_paths.csv'
# dataset_name = 'baseline_data.csv'
dataset_name = 'full_smaller_path'
# encoder_name = 'encoder.pkl'

padding_value = 0

mode = 'train'

attribution_model = 'attention'
