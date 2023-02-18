from torchvision import datasets, transforms
from torch import Generator
from torch.utils.data import random_split
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


from typing import Callable

def load_CIFAR10(data_set: Callable, train_val_split=0.9, data_path='../data/', pre_processing=None):

    if pre_processing is None:
        preprocessor = transforms.Compose([
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize((0.4915, 0.4823, 0.4468),
                                (0.2470, 0.2435, 0.2616))
        ])
    
    # load datasets
    data_train_val = datasets.CIFAR10(
        data_path,       
        train=True,      
        download=True,
        transform=pre_processing)

    data_test = datasets.CIFAR10(
        data_path, 
        train=False,
        download=True,
        transform=pre_processing)

    # train/validation split
    n_train = int(len(data_train_val) * train_val_split)
    n_val =  len(data_train_val) - n_train

    data_train, data_val = random_split(
        data_train_val, 
        [n_train, n_val],
        generator=Generator().manual_seed(123)
    )

    print("Size of the train dataset:        ", len(data_train))
    print("Size of the validation dataset:   ", len(data_val))
    print("Size of the test dataset:         ", len(data_test))
    
    return (data_train, data_val, data_test)

def subset_dataset(data_train, data_val, data_test):
    label_map = {0: 0, 2: 1}
    class_names = ['airplane', 'bird']

    # For each dataset, keep only airplanes and birds
    cifar2_train = [(img, label_map[label]) for img, label in data_train if label in [0, 2]]
    cifar2_val = [(img, label_map[label]) for img, label in data_val if label in [0, 2]]
    cifar2_test = [(img, label_map[label]) for img, label in data_test if label in [0, 2]]

    return cifar2_train, cifar2_val, cifar2_test

load_CIFAR10("CIFAR10")
