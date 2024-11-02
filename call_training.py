import logging
import parameters as param
from preprocessing import preprocessing
from train_and_val import train_and_val
from data_loader import data_loader
from model import CustomLSTM
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import argparse
import utilities as utilities
import pickle

logger = logging.getLogger(__name__)


def training(project_id, bucket_name):
    """
    :param project_id: str, the name of the project id
    :param bucket_name: str, the name of the bucket where to store data
    :return:
    """
    logger.info('Loading the dataset')
    if param.augmented_data:
        logger.info('Checking existence of augmented dataset in storage')
        augmented_data = param.dataset_name + '_augmented.pkl'
        if utilities.check_existence(augmented_data, project_id, bucket_name,
                                     param.working_directory + param.path_preproc):
            logger.info('Loading augmented dataset')
            utilities.load_from_storage(augmented_data, project_id, bucket_name,
                                        param.working_directory + param.path_preproc, param.local_dir)
            df = pd.read_pickle(param.local_dir + augmented_data)
            to_preprocess = False
        else:
            logger.info('Loading dataset')
            utilities.load_from_storage(param.dataset_name + '.csv', project_id, bucket_name,
                                        param.working_directory + param.path_data, param.local_dir)
            df = pd.read_csv(param.local_dir + param.dataset_name + '.csv')
            to_preprocess = True
    else:
        logger.info('Checking existence of preprocessed dataset in storage')
        preprocessed_data = param.dataset_name + '_preprocessed.pkl'
        if utilities.check_existence(preprocessed_data, project_id, bucket_name,
                                     param.working_directory + param.path_preproc):
            logger.info('Loading preprocessed dataset')
            utilities.load_from_storage(preprocessed_data, project_id, bucket_name,
                                        param.working_directory + param.path_preproc, param.local_dir)
            df = pd.read_pickle(param.local_dir + preprocessed_data)
            to_preprocess = False
        else:
            logger.info('Loading dataset')
            utilities.load_from_storage(param.dataset_name + '.csv', project_id, bucket_name,
                                        param.working_directory + param.path_data, param.local_dir)
            df = pd.read_csv(param.local_dir + param.dataset_name + '.csv')
            to_preprocess = True
    logger.info('Dataset loaded')

    # Preprocessing the dataset

    if to_preprocess:
        logger.info('Preprocessing')
        df, encoder = preprocessing(df, project_id, bucket_name, 'train', param.augment)
        vocabulary_size = dict.fromkeys(param.features_to_encode)
        for n, i in enumerate(param.features_to_encode):
            vocabulary_size[i] = len(encoder[i].classes_)
        logger.info('Data preprocessed')
    else:
        logger.info('Loading the encoders')
        vocabulary_size = dict.fromkeys(param.features_to_encode)
        for i in param.features_to_encode:
            encoder_name = 'encoder_' + i + '.pkl'
            utilities.load_from_storage(encoder_name, project_id, bucket_name,
                                        param.working_directory + param.path_preproc, param.local_dir)
            # Loading the encoder if already existing
            with open(param.local_dir + encoder_name, 'rb') as load_encoder:
                encoder = pickle.load(load_encoder)
            vocabulary_size[i] = len(encoder.classes_)

    logger.info('Positive classes: ' + str(df[df.conversion == 1].path.count()) + '; Negative classes: ' +
                str(df[df.conversion == 0].path.count()))
    # Separating the data in a training and validation set
    df_train = df.sample(frac=param.train_ratio, random_state=param.random_seed)
    idx = df_train.index
    logger.info('Positive classes in training set: ' + str(df_train[df_train.conversion == 1].path.count()) +
                '; Negative classes in training set: ' + str(df_train[df_train.conversion == 0].path.count()))
    df_val = df.iloc[df.index.difference(idx)]
    logger.info('Positive classes in validation set: ' + str(df_val[df_val.conversion == 1].path.count()) +
                '; Negative classes in validation set: ' + str(df_val[df_val.conversion == 0].path.count()))
    # Preparing the data for the neural network
    logger.info('Preparing DataLoader')
    dl_train = data_loader(df_train.reset_index(drop=True), shuffle=True)
    dl_val = data_loader(df_val.reset_index(drop=True))
    logger.info('DataLoader ready')
    logger.info('Defining the model')
    input_dim = len(df.columns) - len(param.features_to_encode) - 1
    logger.info('Saving the model parameters')
    rnn_parameters = {'input_dimension': input_dim,
                      'hidden_dimension': param.model_param['hidden_dim'],
                      'embedding_dimension': param.model_param['embedding_dim'],
                      'output_dimension': param.model_param['target_dim'],
                      'nb_layers': param.model_param['layers'],
                      'vocabulary_size': vocabulary_size}
    with open(param.local_dir + 'rnn_parameters.pkl', 'wb') as dump_parameters:
        pickle.dump(rnn_parameters, dump_parameters)
    logger.info('Pushing the model parameters to storage')
    utilities.push_to_storage('rnn_parameters.pkl', project_id, bucket_name, param.local_dir,
                              param.working_directory + param.path_model)
    rnn_parameters_name = 'lstm_id_{id}_hd_{hd}_ed_{ed}_od_{od}_nl_{nl}.pkl'.format(
        id=rnn_parameters['input_dimension'],
        hd=rnn_parameters['hidden_dimension'],
        ed=rnn_parameters['embedding_dimension'],
        od=rnn_parameters['output_dimension'],
        nl=rnn_parameters['nb_layers']
    )
    rnn = CustomLSTM(rnn_parameters['input_dimension'],
                     rnn_parameters['hidden_dimension'],
                     rnn_parameters['embedding_dimension'],
                     rnn_parameters['output_dimension'],
                     rnn_parameters['nb_layers'],
                     rnn_parameters['vocabulary_size'])
    logger.info('Trainable model parameters: ' + str(sum(p.numel() for p in rnn.parameters() if p.requires_grad)))
    # Declaring the loss function. We are using the negative log likelihood
    loss_function = nn.NLLLoss()
    # As optimizer we are using a stochastic gradient descent
    optimizer = optim.SGD(rnn.parameters(), lr=param.model_param['learning_rate'])
    # Training the model
    logger.info('Start training')
    output_name = 'model_results_{data}_hd_{hd}_ed_{ed}_nl_{nl}.csv'.format(data=param.dataset_name,
                                                                            hd=str(param.model_param['hidden_dim']),
                                                                            ed=str(param.model_param['embedding_dim']),
                                                                            nl=str(param.model_param['layers']))
    train_and_val(dl_train, dl_val, rnn, optimizer, loss_function, output_name, rnn_parameters, rnn_parameters_name)
    logger.info('Training ended')
    logger.info('Pushing results to storage')
    utilities.push_to_storage(output_name, project_id, bucket_name, param.local_dir,
                              param.working_directory + param.path_results)
    logger.info('Pushing model checkpoint to storage')
    utilities.push_to_storage(rnn_parameters_name, project_id, bucket_name, param.local_dir,
                              param.working_directory + param.path_model)
    logger.info('Results in storage')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--global_proj_id', help='global_project_id')
    parser.add_argument('--bucket_name', help='bucket_name')

    args = parser.parse_args()

    training(project_id=args.global_proj_id,
             bucket_name=args.bucket_name)
