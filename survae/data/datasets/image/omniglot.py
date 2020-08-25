import torch
import torch.utils.data as data
import os
import urllib.request
from scipy.io import loadmat
from survae.data import DATA_PATH


class OMNIGLOTDataset(data.Dataset):
    """
    The OMNIGLOT dataset of
    (Lake et al., 2013): https://papers.nips.cc/paper/5128-one-shot-learning-by-inverting-a-compositional-causal-process
    as processed in
    (Burda et al., 2015): https://arxiv.org/abs/1509.00519
    """
    url = 'https://github.com/yburda/iwae/blob/master/datasets/OMNIGLOT/chardata.mat'
    file = 'chardata.mat'

    def __init__(self, root=DATA_PATH, train=True, transform=None):
        self.root = os.path.join(os.path.expanduser(root), 'omniglot')
        self.train = train
        self.transform = transform

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' + ' You can download it from https://github.com/yburda/iwae/tree/master/datasets/OMNIGLOT')

        train_x, test_x = self._load_data()
        if train: data = train_x
        else: data = test_x

        c, w, h = 1, 28, 28
        self.data = torch.tensor(data, dtype=torch.float).view(-1, c, h, w)

    def __getitem__(self, index):
        sample = self.data[index]
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.data)

    @property
    def local_file(self):
        return os.path.join(self.root, self.file)

    def download(self):
        if self._check_exists():
            return
        if not os.path.exists(self.root):
            os.makedirs(self.root)

        print('Downloading Omniglot...')
        urllib.request.urlretrieve(self.url, self.local_file)
        print('Done!')

    def _check_exists(self):
        return os.path.exists(self.local_file)

    def _load_data(self):
        data = loadmat(self.local_file)

        train_x = data['data'].astype('float32').T
        test_x = data['testdata'].astype('float32').T

        return train_x, test_x
