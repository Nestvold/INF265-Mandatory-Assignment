from torchvision.datasets import MNIST
from torch import Generator

from typing import Callable

def load_MNIST(data_set: Callable, train_val_split=0.9, data_path='../data/', pre_processing: Callable=None):
    
    # load datasets
    data_train_val = datasets.data_set(
        data_path,       
        train=True,      
        download=True,
        transform=preprocessor)

    data_test = datasets.data_set(
        data_path, 
        train=False,
        download=True,
        transform=preprocessor)

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
