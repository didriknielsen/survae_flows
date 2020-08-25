import h5py
import torch
import torch.utils.data as data
from torchvision import transforms
import os
import numpy as np
from PIL import Image
import urllib.request
from survae.data import DATA_PATH


class FixedBinaryMNISTDataset(data.Dataset):
    """
    The MNIST dataset of
    (LeCun, 1998): http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf
    with a fixed binarization as used in
    (Larochelle & Murray, 2011): http://proceedings.mlr.press/v15/larochelle11a/larochelle11a.pdf

    See Footnote 2 on page 6 and Appendix D of (Burda et al., 2015):
    https://arxiv.org/pdf/1509.00519.pdf
    for a remark on the different versions of MNIST.
    """
    train_file = 'binarized_mnist_train.amat'
    valid_file = 'binarized_mnist_valid.amat'
    test_file = 'binarized_mnist_test.amat'

    def __init__(self, root=DATA_PATH, split='train', download=False):
        assert split in {'train', 'valid', 'test'}
        self.root = os.path.join(os.path.expanduser(root), 'static_mnist')
        self.split = split

        if download: self.download()
        if not self._check_exists():
            raise RuntimeError('Dataset not found.' + ' You can use download=True to download it')

        self.data = self._get_data(split=split)

    def __getitem__(self, index):
        img = self.data[index]
        img = Image.fromarray(img)
        img = transforms.ToTensor()(img).type(torch.FloatTensor)
        return img

    def __len__(self):
        return len(self.data)

    def _get_data(self, split=True):
        with h5py.File(os.path.join(self.root, 'data.h5'), 'r') as hf:
            data = hf.get(split)
            data = np.array(data)
        return data

    def download(self):
        if self._check_exists():
            return
        if not os.path.exists(self.root):
            os.makedirs(self.root)

        print('Downloading MNIST with fixed binarization...')
        for dataset in ['train', 'valid', 'test']:
            filename = 'binarized_mnist_{}.amat'.format(dataset)
            url = 'http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_{}.amat'.format(dataset)
            print('Downloading from {}...'.format(url))
            local_filename = os.path.join(self.root, filename)
            urllib.request.urlretrieve(url, local_filename)
            print('Saved to {}'.format(local_filename))

        def filename_to_np(filename):
            with open(filename) as f:
                lines = f.readlines()
            return np.array([[int(i)for i in line.split()] for line in lines]).astype('int8')

        train_data = filename_to_np(os.path.join(self.root, self.train_file))
        valid_data = filename_to_np(os.path.join(self.root, self.valid_file))
        test_data = filename_to_np(os.path.join(self.root, self.test_file))
        with h5py.File(os.path.join(self.root, 'data.h5'), 'w') as hf:
            hf.create_dataset('train', data=train_data.reshape(-1, 28, 28))
            hf.create_dataset('valid', data=valid_data.reshape(-1, 28, 28))
            hf.create_dataset('test', data=test_data.reshape(-1, 28, 28))
        print('Done!')

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, 'data.h5'))
