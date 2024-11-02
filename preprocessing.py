import parameters as param
from imblearn.over_sampling import RandomOverSampler
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle
import utilities as utilities

logger = logging.getLogger(__name__)


def preprocessing(df, project_id, bucket_name, mode=None, augment=False):
    """
    :param df: pandas.dataframe. The full dataset
    :param project_id: str, the name of the project id
    :param bucket_name: str, the name of the bucket where to store the data
    :param mode: str. If in evaluation mode or training mode
    :param augment: boolean. Stating if the dataset has to be augmented
    :return: df: the preprocessed dataframe. encoder: the LabelEncoder for the channels
    """

    if mode == 'eval':
        df = df[df.conversion == 1]
        return df

    # Preprocessing the dataset
    classes = {i: df[i].unique() for i in param.features_to_encode}
    param.idx_to_embed = [df.columns.get_loc(i)-1 for i in param.features_to_encode]
    for i in param.vocabulary_size:
        param.vocabulary_size[i] = classes[i].shape[0]
    # Preparing the encoders
    encoders = dict.fromkeys(param.features_to_encode)
    for i in encoders.keys():
        encoder_name = 'encoder_' + i + '.pkl'
        # Checking presence of encoders in storage
        if utilities.check_existence(encoder_name, project_id, bucket_name,
                                     param.working_directory + param.path_preproc):
            logger.info('Encoder ' + encoder_name + ' is in storage')
            logger.info('Loading the encoder')
            utilities.load_from_storage(encoder_name, project_id, bucket_name,
                                        param.working_directory + param.path_preproc, param.local_dir)
            # Loading the encoder if already existing
            with open(param.local_dir + encoder_name, 'rb') as load_encoder:
                encoders[i] = pickle.load(load_encoder)
        else:
            # Defining the encoder
            logger.info('Encoder ' + encoder_name + ' is not in storage')
            logger.info('Creating the encoder')
            encoders[i] = LabelEncoder()
            encoders[i].fit(classes[i])
            # Saving the encoder
            with open(param.local_dir + encoder_name, 'wb') as encoder_file:
                pickle.dump(encoders[i], encoder_file)
            logger.info('Pushing ' + encoder_name + ' in storage')
            utilities.push_to_storage(encoder_name, project_id, bucket_name, param.local_dir,
                                      param.working_directory + param.path_preproc)
    logger.info('Encoders loaded')

    # Applying the encoder to the channel column
    for i in param.features_to_encode:
        df[i] = encoders[i].transform(df[i].values)

    # Increasing the values of channel_init and nb_uniques_pages_session by 1, so that the 0 can be used as
    # padding value

    df[df.columns.difference(['cluster_id', 'path_progression', 'is_positive_path'],
                             sort=False)] = df[df.columns.difference(['cluster_id',
                                                                      'path_progression',
                                                                      'is_positive_path'], sort=False)].apply(
        lambda x: x + 1)

    # Create entries with a list of channel and the other features

    aggregated_df = df.sort_values(by=['cluster_id', 'path_progression'])[
        df.columns.difference(['path_progression', 'is_positive_path'], sort=False)].groupby('cluster_id').aggregate(
        list).reset_index()
    # Merge the aggregated dataset with the is_positive_path column
    joined_df = aggregated_df.merge(df[['cluster_id', 'is_positive_path']].drop_duplicates(),
                                    left_on='cluster_id', right_on='cluster_id', how='inner')
    # Set the column as int
    joined_df['is_positive_path'] = joined_df['is_positive_path'].astype('int')
    # Drop the cluster_id column
    final_df = joined_df.drop(columns='cluster_id')
    # Rename the necessary columns
    final_df.rename(columns={'channel_init': 'path', 'is_positive_path': 'conversion'}, inplace=True)
    if augment:
        logger.info('Augmenting the dataset')
        augmented_data = param.dataset_name.rstrip('.csv') + '_augmented.pkl'
        ros = RandomOverSampler()
        x_augmented, y_augmented = ros.fit_resample(final_df[final_df.columns.difference(['conversion'],
                                                                                         sort=False)].values,
                                                    final_df.conversion)
        augmented_dict = {final_df[final_df.columns.difference(['conversion'], sort=False)].columns[i]:
                              x_augmented[:, i] for i in range(x_augmented.shape[1])}
        final_df = pd.DataFrame(augmented_dict)
        final_df['conversion'] = y_augmented
        # Saving the augmented dataset
        final_df.to_pickle(param.local_dir + augmented_data)
        logger.info('Pushing augmented data to storage')
        utilities.push_to_storage(augmented_data, project_id, bucket_name, param.local_dir,
                                  param.working_directory + param.path_preproc, timeout=3600)
    else:
        logger.info('Pushing preprocessed data to storage')
        preprocessed_data = param.dataset_name.rstrip('.csv') + '_preprocessed.pkl'
        final_df.to_pickle(param.local_dir + preprocessed_data)
        utilities.push_to_storage(preprocessed_data, project_id, bucket_name, param.local_dir,
                                  param.working_directory + param.path_preproc, timeout=3600)
    return final_df, encoders
