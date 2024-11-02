import logging
import numpy as np
import pandas as pd
from itertools import combinations
from data_loader import data_loader
import torch


logger = logging.getLogger(__name__)


def get_impact_attribution(dataset, attribution, model):
    """"
    :param dataset: pandas.DataFrame. The dataset with the paths
    :param attribution: numpy array. The array with the values of autonomy, attribution and presence for the channels
    :param model: torch.lstm. The trained model
    :return: numpy array. The array with the updated values of autonomy, attribution and presence for the channels
    """
    for i in range(dataset.shape[0]):
        data_dict = {j: [] for j in dataset.columns.difference(['conversion'], sort=False)}
        for j in dataset.columns.difference(['conversion'], sort=False):
            data_dict[j] = [dataset[j].values[i][0:k] for k in range(len(dataset[j].values[i]) + 1)]
        data_dict['conversion'] = [0.] * len(data_dict['path'])
        data_subpath = pd.DataFrame.from_dict(data_dict)
        dl_subpath = data_loader(data_subpath)
        model.eval()
        proba = []
        # Since the model requires a non-zero length input, we have set the empty path as a single element path
        # filled with the padded value. Nevertheless, the output of the model is the output for a zero input
        # vector since the embedding layer set the single value padded vector to zero in the embedding space.
        # So the final output is effectively the probability of conversion without any channel
        # (the bias of the model).
        for j, (x, y, l) in enumerate(dl_subpath):
            output = model(x, l)
            proba.extend(torch.exp(output).detach().numpy()[:, 1])
        data_subpath['conversion'] = proba
        for j in range(1, data_subpath.shape[0]):
            attribution[data_subpath.path[j][-1]] += \
                [0., data_subpath.conversion.values[j] / np.sum(data_subpath.conversion.values[1:]), 0.]
    return attribution


def get_attention_attribution(df, attribution, model):
    """
    :param df: pandas.DataFrame. The dataset with the paths
    :param attribution: numpy array. The array with the attribution scores
    :param model: torch.lstm. The trained model for evaluating the probability of conversions
    :return: numpy array. The updated array with the attribution scores
    """

    path_idx = df.columns.get_loc('path')
    dl_path = data_loader(df.reset_index(drop=True))
    model.eval()
    for x, y, l in dl_path:
        path_length = l.detach().numpy()
        output, att_weights = model(x, l)
        for i in range(len(path_length)):
            for j in range(0, path_length[i]):
                attribution[x.detach().numpy()[i, j][path_idx]][1] += att_weights.detach().numpy()[i, j][-1]
    return attribution


def initialize_attribution(df, encoder):
    """
    :param df: pandas.DataFrame. The dataframe with the paths
    :param encoder: scikit-learn LabelEncoder. The encoder for the channels
    :return: channels: numpy array. The array with the value of autonomy, attribution and presence for
    all the channels
    """
    channels = np.zeros((len(encoder.classes_) + 1, 3))
    for i in range(df.shape[0]):
        if len(np.unique(df.path.values[i])) == 1:
            channels[np.unique(df.path.values[i])] += [1., 0., 1.]
        else:
            channels[np.unique(df.path.values[i])] += [0., 0., 1.]
    return channels


def check_attribution(results_dataset, total_kpi):
    """
    :param results_dataset: pandas.DataFrame. The dataset containing the autonomy, attribution, presence of each channel
    :param total_kpi: int. The number of conversions in the dataset
    :return:
    """
    if results_dataset.Attribution.sum() - total_kpi > 1e-3:
        logger.info('The sum of attribution is not equal to the total kpi')
    if not np.all(results_dataset.Attribution >= results_dataset.Autonomy):
        logger.info('The attribution is not greater or equal to the autonomy everywhere')
    if not np.all(results_dataset.Attribution <= results_dataset.Presence):
        logger.info('The attribution is not smaller or equal to the presence everywhere')


def attribution_model(df, encoders, type_attribution, model):
    """
    :param df: pandas.DataFrame. The dataset with the paths
    :param encoders: dict of scikit LabelEncoder. The encoders for the channel, the env_first_url and env_last_url
    :param type_attribution: str. The type of attribution model to use
    :param model: torch.lstm. The trained model to evaluate the probability of conversion
    :return: pandas.DataFrame. The dataset with the attribution results, the autonomy and the presence
    """
    logger.info('Initializing attribution scores')
    channels_attrib = initialize_attribution(df, encoders['channel_init'])
    logger.info('Evaluating attributions')
    if type_attribution == 'impact':
        logger.info('Evaluating impact-based attribution')
        channels_attrib = get_impact_attribution(df, channels_attrib, model)
    elif type_attribution == 'attention':
        logger.info('Evaluating attention-based attribution')
        channels_attrib = get_attention_attribution(df, channels_attrib, model)
    logger.info('Attribution evaluated')
    df_results = pd.DataFrame(channels_attrib, columns=['Autonomy', 'Attribution', 'Presence'])
    check_attribution(df_results, df.shape[0])
    return df_results
