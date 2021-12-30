import torchvision.datasets as datasets
import os
from torchvision import transforms

'''This file contains two simple functions for loading the MNIST anf Omnilog datasets.
Inputs are the base url to save the data. They make base_dir/data subdirectory and store the 
datasets in it. They return train and test data in the form of VisionDataset class.
'''


def load_MNIST(base_dir):
    save_path = os.path.join(base_dir, 'data')
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    train = datasets.MNIST(save_path, train=True, download=True, transform=transforms.ToTensor())
    test = datasets.MNIST(save_path, train=True, download=True, transform=transforms.ToTensor())
    return train, test


def load_Omniglot(base_dir):
    save_path = os.path.join(base_dir, 'data')
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    back = datasets.Omniglot(save_path, download=True, background=True, transform=transforms.ToTensor())
    evaluation = datasets.Omniglot(save_path, download=True, background=False, transform=transforms.ToTensor())
    return back, evaluation

