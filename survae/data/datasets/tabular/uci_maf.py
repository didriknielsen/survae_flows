'''
Contains datasets used in MAF paper.
See https://github.com/gpapamak/maf.
'''

import os
import h5py
import tarfile
import numpy as np
import pandas as pd
from collections import Counter

import torch
from survae.data import DATA_PATH


class UCI_MAF_Dataset(torch.utils.data.Dataset):

    url = 'https://zenodo.org/record/1161203/files/data.tar.gz?download=1'
    folder = 'uci_maf'
    download_file = 'data.tar.gz'
    raw_folder = None
    raw_file = None

    def __init__(self, root=DATA_PATH, split='train', download=False):
        assert split in {'train', 'validation', 'test'}
        self.root = os.path.expanduser(root)
        self.split = split

        if not self._check_download():
            if download:
                self.download()
            else:
                raise RuntimeError('Dataset not found.' +
                                   ' You can use download=True to download it')

        if not self._check_raw():
            self.extract()

        self.x = torch.from_numpy(self.load_data(split))

    def __getitem__(self, index):
        return self.x[index]

    def __len__(self):
        return len(self.x)

    @property
    def download_data_path(self):
        return os.path.join(self.root, self.folder, self.download_file)

    @property
    def raw_data_path(self):
        return os.path.join(self.root, self.raw_folder, self.raw_file)

    def _check_download(self):
        return os.path.exists(self.download_data_path)

    def _check_raw(self):
        return os.path.exists(self.raw_data_path)

    def download(self):
        """Download the data if it doesn't exist in parent_folder already."""
        from six.moves import urllib

        if not os.path.exists(os.path.join(self.root, self.folder)):
            os.makedirs(os.path.join(self.root, self.folder))

        print('Downloading', self.download_file)
        urllib.request.urlretrieve(self.url, self.download_data_path)

    def extract(self):

        print("Extracting data...")
        tar = tarfile.open(self.download_data_path)
        tar.extractall(os.path.join(self.root, self.folder))
        tar.close()

    def load_data(self, split):
        raise NotImplementedError()


class BSDS300Dataset(UCI_MAF_Dataset):

    raw_folder = 'uci_maf/data/BSDS300'
    raw_file = 'BSDS300.hdf5'

    def load_data(self, split):
        # Taken from https://github.com/bayesiains/nsf/blob/master/data/bsds300.py
        file = h5py.File(self.raw_data_path, 'r')
        return np.array(file[split]).astype(np.float32)


class GasDataset(UCI_MAF_Dataset):

    raw_folder = 'uci_maf/data/gas'
    raw_file = 'ethylene_CO.pickle'

    def load_data(self, split):
        # Taken from https://github.com/bayesiains/nsf/blob/master/data/gas.py

        def load_data(file):
            data = pd.read_pickle(file)
            data.drop("Meth", axis=1, inplace=True)
            data.drop("Eth", axis=1, inplace=True)
            data.drop("Time", axis=1, inplace=True)
            return data

        def get_correlation_numbers(data):
            C = data.corr()
            A = C > 0.98
            B = A.sum(axis=1)
            return B

        def load_data_and_clean(file):
            data = load_data(file)
            B = get_correlation_numbers(data)

            while np.any(B > 1):
                col_to_remove = np.where(B > 1)[0][0]
                col_name = data.columns[col_to_remove]
                data.drop(col_name, axis=1, inplace=True)
                B = get_correlation_numbers(data)
            data = (data - data.mean()) / data.std()

            return data.values

        def load_data_and_clean_and_split(file):
            data = load_data_and_clean(file)
            N_test = int(0.1 * data.shape[0])
            data_test = data[-N_test:]
            data_train = data[0:-N_test]
            N_validate = int(0.1 * data_train.shape[0])
            data_validate = data_train[-N_validate:]
            data_train = data_train[0:-N_validate]

            return data_train, data_validate, data_test

        data_train, data_validate, data_test = load_data_and_clean_and_split(self.raw_data_path)

        if split == 'train':
            return np.array(data_train).astype(np.float32)
        if split == 'validation':
            return np.array(data_validate).astype(np.float32)
        if split == 'test':
            return np.array(data_test).astype(np.float32)


class HEPMASSDataset(UCI_MAF_Dataset):

    raw_folder = 'uci_maf/data/hepmass'
    raw_file = '1000_train.csv' # ['1000_train.csv', '1000_test.csv']

    def load_data(self, split):
        # Taken from https://github.com/bayesiains/nsf/blob/master/data/hepmass.py
        def load_data(path):

            data_train = pd.read_csv(filepath_or_buffer=os.path.join(path, '1000_train.csv'),
                                     index_col=False)
            data_test = pd.read_csv(filepath_or_buffer=os.path.join(path, '1000_test.csv'),
                                    index_col=False)

            return data_train, data_test

        def load_data_no_discrete(path):
            """Loads the positive class examples from the first 10% of the dataset."""
            data_train, data_test = load_data(path)

            # Gets rid of any background noise examples i.e. class label 0.
            data_train = data_train[data_train[data_train.columns[0]] == 1]
            data_train = data_train.drop(data_train.columns[0], axis=1)
            data_test = data_test[data_test[data_test.columns[0]] == 1]
            data_test = data_test.drop(data_test.columns[0], axis=1)
            # Because the data_ set is messed up!
            data_test = data_test.drop(data_test.columns[-1], axis=1)

            return data_train, data_test

        def load_data_no_discrete_normalised(path):

            data_train, data_test = load_data_no_discrete(path)
            mu = data_train.mean()
            s = data_train.std()
            data_train = (data_train - mu) / s
            data_test = (data_test - mu) / s

            return data_train, data_test

        def load_data_no_discrete_normalised_as_array(path):

            data_train, data_test = load_data_no_discrete_normalised(path)
            data_train, data_test = data_train.values, data_test.values

            i = 0
            # Remove any features that have too many re-occurring real values.
            features_to_remove = []
            for feature in data_train.T:
                c = Counter(feature)
                max_count = np.array([v for k, v in sorted(c.items())])[0]
                if max_count > 5:
                    features_to_remove.append(i)
                i += 1
            data_train = data_train[:, np.array(
                [i for i in range(data_train.shape[1]) if i not in features_to_remove])]
            data_test = data_test[:, np.array(
                [i for i in range(data_test.shape[1]) if i not in features_to_remove])]

            N = data_train.shape[0]
            N_validate = int(N * 0.1)
            data_validate = data_train[-N_validate:]
            data_train = data_train[0:-N_validate]

            return data_train, data_validate, data_test

        data_train, data_validate, data_test = load_data_no_discrete_normalised_as_array(os.path.join(self.root, self.raw_folder))

        if split == 'train':
            return np.array(data_train).astype(np.float32)
        if split == 'validation':
            return np.array(data_validate).astype(np.float32)
        if split == 'test':
            return np.array(data_test).astype(np.float32)


class MiniBooNEDataset(UCI_MAF_Dataset):

    raw_folder = 'uci_maf/data/miniboone'
    raw_file = 'data.npy'

    def load_data(self, split):
        # Taken from https://github.com/bayesiains/nsf/blob/master/data/miniboone.py
        def load_data(path):
            # NOTE: To remember how the pre-processing was done.
            # data_ = pd.read_csv(root_path, names=[str(x) for x in range(50)], delim_whitespace=True)
            # print data_.head()
            # data_ = data_.as_matrix()
            # # Remove some random outliers
            # indices = (data_[:, 0] < -100)
            # data_ = data_[~indices]
            #
            # i = 0
            # # Remove any features that have too many re-occuring real values.
            # features_to_remove = []
            # for feature in data_.T:
            #     c = Counter(feature)
            #     max_count = np.array([v for k, v in sorted(c.iteritems())])[0]
            #     if max_count > 5:
            #         features_to_remove.append(i)
            #     i += 1
            # data_ = data_[:, np.array([i for i in range(data_.shape[1]) if i not in features_to_remove])]
            # np.save("~/data_/miniboone/data_.npy", data_)

            data = np.load(path)
            N_test = int(0.1 * data.shape[0])
            data_test = data[-N_test:]
            data = data[0:-N_test]
            N_validate = int(0.1 * data.shape[0])
            data_validate = data[-N_validate:]
            data_train = data[0:-N_validate]

            return data_train, data_validate, data_test

        def load_data_normalised(path):
            data_train, data_validate, data_test = load_data(path)
            data = np.vstack((data_train, data_validate))
            mu = data.mean(axis=0)
            s = data.std(axis=0)
            data_train = (data_train - mu) / s
            data_validate = (data_validate - mu) / s
            data_test = (data_test - mu) / s

            return data_train, data_validate, data_test

        data_train, data_validate, data_test = load_data_normalised(self.raw_data_path)

        if split == 'train':
            return np.array(data_train).astype(np.float32)
        if split == 'validation':
            return np.array(data_validate).astype(np.float32)
        if split == 'test':
            return np.array(data_test).astype(np.float32)


class PowerDataset(UCI_MAF_Dataset):

    raw_folder = 'uci_maf/data/power'
    raw_file = 'data.npy'

    def load_data(self, split):
        # Taken from https://github.com/bayesiains/nsf/blob/master/data/power.py
        def load_data():
            file = self.raw_data_path
            return np.load(file)

        def load_data_split_with_noise():
            rng = np.random.RandomState(42)

            data = load_data()
            rng.shuffle(data)
            N = data.shape[0]

            data = np.delete(data, 3, axis=1)
            data = np.delete(data, 1, axis=1)
            ############################
            # Add noise
            ############################
            # global_intensity_noise = 0.1*rng.rand(N, 1)
            voltage_noise = 0.01 * rng.rand(N, 1)
            # grp_noise = 0.001*rng.rand(N, 1)
            gap_noise = 0.001 * rng.rand(N, 1)
            sm_noise = rng.rand(N, 3)
            time_noise = np.zeros((N, 1))
            # noise = np.hstack((gap_noise, grp_noise, voltage_noise, global_intensity_noise, sm_noise, time_noise))
            # noise = np.hstack((gap_noise, grp_noise, voltage_noise, sm_noise, time_noise))
            noise = np.hstack((gap_noise, voltage_noise, sm_noise, time_noise))
            data += noise

            N_test = int(0.1 * data.shape[0])
            data_test = data[-N_test:]
            data = data[0:-N_test]
            N_validate = int(0.1 * data.shape[0])
            data_validate = data[-N_validate:]
            data_train = data[0:-N_validate]

            return data_train, data_validate, data_test

        def load_data_normalised():
            data_train, data_validate, data_test = load_data_split_with_noise()
            data = np.vstack((data_train, data_validate))
            mu = data.mean(axis=0)
            s = data.std(axis=0)
            data_train = (data_train - mu) / s
            data_validate = (data_validate - mu) / s
            data_test = (data_test - mu) / s

            return data_train, data_validate, data_test

        data_train, data_validate, data_test = load_data_normalised()

        if split == 'train':
            return np.array(data_train).astype(np.float32)
        if split == 'validation':
            return np.array(data_validate).astype(np.float32)
        if split == 'test':
            return np.array(data_test).astype(np.float32)
