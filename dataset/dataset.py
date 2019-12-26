import os
import numpy as np
from PIL import Image
from scipy.io import loadmat

from torch.utils.data import Dataset
import torchvision.transforms as transforms


class FlowerData(Dataset):
    def __init__(self, args, split: str='train'):
        """
        main_class == -1: 80 images for each class
        main_class == k (0 <= k <= 16): 80 images for k-th class and 10 for each other class
        """
        self.args = args
        self.split = split
        self.main_class = self.args.main_class
        self.idx = self.get_split()
        self.transform = transforms.Compose(
            [transforms.Resize(size=(self.args.image_height, self.args.image_width)),
             transforms.ToTensor(),  # pixel value range [0, 1]
             transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3, inplace=True)])

    def get_split(self):
        splits_path = os.path.join(self.args.data_path, 'datasplits.mat')
        datasplits = loadmat(splits_path)
        index = None

        if self.split == 'train':
            index = datasplits['trn1'][0]
            index = np.random.choice(index, size=len(index), replace=True)
        elif 'test' in self.split:
            index = np.hstack((datasplits['tst1'], datasplits['val1']))[0]

        return index

    def __getitem__(self, item):
        item %= self.idx.shape[0]
        img_num = self.idx[item]
        class_id = (img_num - 1) // self.args.num_image_per_class
        class_id = class_id.astype(np.float32)
        img_name = 'image_%.4d.jpg' % img_num
        img_path = os.path.join(self.args.data_path, 'jpg', img_name)
        img = Image.open(img_path)  # RGB
        img = self.transform(img)

        return img, class_id

    def __len__(self):

        return self.idx.shape[0]
