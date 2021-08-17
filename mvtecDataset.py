# This is a modified version of original  https://github.com/pytorch/vision/blob/master/torchvision/datasets/cifar.py
# This file and the mvtec data directory must be in the same directory, such that:
# /.../this_directory/mvtecDataset.py
# /.../this_directory/mvtec/bottle/...
# /.../this_directory/mvtec/cable/...
# and so on

from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import torch.utils.data as data
import matplotlib.image as mpimg
from torchvision import transforms


class MVTEC(data.Dataset):
    """`MVTEC <https://www.mvtec.com/company/research/datasets/mvtec-ad/>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directories
            ``bottle``, ``cable``, etc., exists.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        resize (int, optional): Desired output image size.
        interpolation (int, optional): Interpolation method for downsizing image.
        category: bottle, cable, capsule, etc.
    """


    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 category='carpet', resize=None, interpolation=2):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.resize = resize
        self.interpolation = interpolation
        
        # load images for training
        if self.train:
            self.train_data = []
            self.train_labels = []
            cwd = os.getcwd()
            trainFolder = self.root+'/'+category+'/train/good/'
            os.chdir(trainFolder)
            filenames = [f.name for f in os.scandir()]
            for file in filenames:
                img = mpimg.imread(file)
                img = img*255
                img = img.astype(np.uint8)
                self.train_data.append(img)
                self.train_labels.append(1)                 
            os.chdir(cwd)
                
            self.train_data = np.array(self.train_data)      
        else:
        # load images for testing
            self.test_data = []
            self.test_labels = []
            
            cwd = os.getcwd()
            testFolder = self.root+'/'+category+'/test/'
            os.chdir(testFolder)
            subfolders = [sf.name for sf in os.scandir() if sf.is_dir()]
            cwsd = os.getcwd()
            
            # for every subfolder in test folder
            for subfolder in subfolders:
                label = 0
                if subfolder == 'good':
                    label = 1
                testSubfolder = './'+subfolder+'/'
                os.chdir(testSubfolder)
                filenames = [f.name for f in os.scandir()]
                for file in filenames:
                    img = mpimg.imread(file)
                    img = img*255
                    img = img.astype(np.uint8)
                    self.test_data.append(img)
                    self.test_labels.append(label)
                os.chdir(cwsd)
            os.chdir(cwd)
                
            self.test_data = np.array(self.test_data)
                
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        
        #if resizing image
        if self.resize is not None:
            resizeTransf = transforms.Resize(self.resize, self.interpolation)
            img = resizeTransf(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target

    def __len__(self):
        """
        Args:
            None
        Returns:
            int: length of array.
        """
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)
