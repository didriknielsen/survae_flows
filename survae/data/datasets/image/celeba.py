import os
import torch
import torch.utils.data as data
import numpy as np
import pandas as pd
import errno
import zipfile
import io
from PIL import Image
from torchvision.transforms.functional import crop, resize
from survae.data import DATA_PATH


class CelebADataset(data.Dataset):
    """
    The CelebA dataset of
    (Liu et al., 2015): https://arxiv.org/abs/1411.7766
    preprocessed to 64x64 as in
    (Larsen et al. 2016): https://arxiv.org/abs/1512.09300
    (Dinh et al., 2017): https://arxiv.org/abs/1605.08803

    From https://github.com/laurent-dinh/models/blob/master/real_nvp/celeba_formatting.py:
    "
    Download img_align_celeba.zip from
    http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html under the
    link "Align&Cropped Images" in the "Img" directory and list_eval_partition.txt
    under the link "Train/Val/Test Partitions" in the "Eval" directory. Then do:
    unzip img_align_celeba.zip
    "

    Subsequently, move the files img_align_celeba.zip and list_eval_partition.txt
    into folder [root]/celeba/raw/

    """

    raw_folder = 'celeba/raw'
    processed_folder = 'celeba/processed'

    def __init__(self, root=DATA_PATH, split='train', transform=None):
        assert split in {'train','valid','test'}
        self.root = os.path.expanduser(root)
        self.split = split
        self.transform = transform

        if not self._check_raw():
            raise RuntimeError('Dataset not found.\n\nFrom docstring:\n\n' + self.__doc__)

        if not self._check_processed():
            self.process()

        if self.split=='train':
            self.data = torch.load(self.processed_train_file)
        elif self.split=='valid':
            self.data = torch.load(self.processed_valid_file)
        elif self.split=='test':
            self.data = torch.load(self.processed_test_file)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tensor: image
        """

        img = self.data[index]

        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.data)

    @property
    def raw_data_folder(self):
        return os.path.join(self.root, self.raw_folder)

    @property
    def raw_zip_file(self):
        return os.path.join(self.raw_data_folder, 'img_align_celeba.zip')

    @property
    def raw_txt_file(self):
        return os.path.join(self.raw_data_folder, 'list_eval_partition.txt')

    def _check_raw(self):
        return os.path.exists(self.raw_zip_file) and os.path.exists(self.raw_txt_file)

    @property
    def processed_data_folder(self):
        return os.path.join(self.root, self.processed_folder)

    @property
    def processed_train_file(self):
        return os.path.join(self.processed_data_folder, 'train.pt')

    @property
    def processed_valid_file(self):
        return os.path.join(self.processed_data_folder, 'valid.pt')

    @property
    def processed_test_file(self):
        return os.path.join(self.processed_data_folder, 'test.pt')

    def _check_processed(self):
        return os.path.exists(self.processed_train_file) and os.path.exists(self.processed_valid_file) and os.path.exists(self.processed_test_file)

    def process_file_list(self, zipfile_object, file_list, processed_filename):
        images = []
        for i, jpg_file in enumerate(file_list):

            if (i+1)%1000 == 0:
                print('File', i+1, '/', len(file_list), end='\r')

            ## Read file
            img_bytes = zipfile_object.read('img_align_celeba/' + jpg_file)
            img = Image.open(io.BytesIO(img_bytes))

            ## Crop image
            # Coordinates taken from Line 981 in
            # https://github.com/laurent-dinh/models/blob/master/real_nvp/real_nvp_multiscale_dataset.py
            # Coordinates of upper left corner: (40, 15)
            # Size of cropped image: (148, 148)
            cropped_img = crop(img, 40, 15, 148, 148)

            ## Resize image
            # Resizing taken from Line 995-996 in
            # https://github.com/laurent-dinh/models/blob/master/real_nvp/real_nvp_multiscale_dataset.py
            resized_img = resize(img, size=(64,64), interpolation=Image.BILINEAR)

            images.append(resized_img)

        torch.save(images, processed_filename)

    def process(self):

        print("Reading filenames...")
        train_files = []
        valid_files = []
        test_files = []
        for line in open(self.raw_txt_file, 'r'):
            a, b = line.split()
            if b=='0':
                train_files.append(a)
            elif b=='1':
                valid_files.append(a)
            elif b=='2':
                test_files.append(a)

        print("Reading zip file...")
        zip = zipfile.ZipFile(self.raw_zip_file, 'r')

        if not os.path.exists(self.processed_data_folder):
            os.mkdir(self.processed_data_folder)

        print("Preparing training data...")
        self.process_file_list(zipfile_object=zip, file_list=train_files, processed_filename=self.processed_train_file)

        print("Preparing validation data...")
        self.process_file_list(zipfile_object=zip, file_list=valid_files, processed_filename=self.processed_valid_file)

        print("Preparing test data...")
        self.process_file_list(zipfile_object=zip, file_list=test_files, processed_filename=self.processed_test_file)

        zip.close()
