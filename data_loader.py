from torch.utils.data import Dataset, DataLoader
import numpy as np
import parameters as param
import torch
from torch.utils.data.sampler import Sampler
from torch.nn.utils.rnn import pad_sequence


# Defining a custom dataset class that takes the path to the pandas dataset and returns a padded pytorch dataset
class CustomData(Dataset):

    def __init__(self, df, cut_len=None):
        """
        :param df: pandas dataframe. Dataset to load
        :param encoder: scikit LabelEncoder. The encoder for the channels
        :param cut_len: int. The maximum length admitted for a path
        """
        super(CustomData, self).__init__()
        self.df = df
        self.cut_len = cut_len
        if self.cut_len:
            self.df = self.df[self.df.path.apply(len) <= self.cut_len].reset_index(drop=True)
        self.max_len = max([len(p) for p in self.df.path.values])
        # Create a bucket column in the dataset, containing the index of the bucket in which the row belong
        self.bucketize()

    def __len__(self):
        """
        :return: The number of rows in the dataset
        """
        return self.df.shape[0]

    def __getitem__(self, idx):
        """
        :param idx: The index of the row of a dataset
        :return: X: The encoded path for row idx. y: The conversion value for row idx. l: The minimum between the length
        of the path and the maximum length admitted
        """
        X = np.stack(self.df[self.df.columns.difference(['conversion', 'bucket'], sort=False)].values[idx])
        y = self.df.conversion[idx]
        # The value 1 is required for the case in which the input is an empty list. In that case it is replaced by a
        # single element list with the padding value. This is necessary because pack_padded_sequence requires a
        # length greater than zero
        l = max(1, len(self.df.path[idx]))
        return X, y, l

    def bucketize(self):
        # Create a "bucket" column in the dataset containing the index of the bucket to which the row belongs
        self.df['bucket'] = np.digitize(self.df.path.apply(len),
                                        np.sort(param.model_param['bucket']+[self.max_len+1]), right=True)
        return


class BatchBucketSampler(Sampler):
    def __init__(self, data_source, batch_size=1, shuffle=False):
        """
        :param data_source: pandas DataFrame. The dataset to process
        :param batch_size: int. The batch dimension
        """
        super(BatchBucketSampler, self).__init__(data_source)
        self.data = data_source
        self.batch_dim = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        # Getting a dictionary with key the bucket and values the rows belonging to the bucket
        bucket_to_ids = self.get_bucket_idx()
        # Batching the elements in each bucket according to batch_size
        ids_list = []
        for k in bucket_to_ids.keys():
            # Shuffling the index in each bucket
            np.random.shuffle(bucket_to_ids[k])
            # Creating an array with the batched indices for each bucket
            batched_bucket = np.array_split(bucket_to_ids[k], np.ceil(len(bucket_to_ids[k]) / self.batch_dim))
            ids_list += list(batched_bucket)
        if self.shuffle:
            np.random.shuffle(ids_list)
        for i in ids_list:
            yield i

    def get_bucket_idx(self):
        return self.data.groupby(by='bucket').indices


def collate(batches):
    padded_sequence = pad_sequence([torch.tensor(i[0].T) for i in batches], batch_first=True,
                                   padding_value=param.padding_value)
    return padded_sequence, torch.tensor([i[1] for i in batches]), torch.tensor([i[2] for i in batches])


def data_loader(df, cut_len=None, shuffle=False):
    """
    :param df: pandas DataFrame. The dataset to load.
    :param cut_len: int. The maximum path length admitted
    :param shuffle: bool. The condition for shuffling the dataset
    :return: pytorch DataLoader
    """
    ds = CustomData(df, cut_len)
    sampler = BatchBucketSampler(df, param.model_param['batch_dim'], shuffle)
    dl = DataLoader(dataset=ds, batch_sampler=sampler, collate_fn=collate)
    return dl
