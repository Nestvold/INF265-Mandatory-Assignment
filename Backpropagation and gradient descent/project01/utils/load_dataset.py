from torch.utils.data import random_split
from torchvision import transforms
from typing import Callable
from torch import Generator
import ssl



def load_dataset(data_set: Callable, train_val_split=0.9, data_path='../data/', preprocessor=None):

    if preprocessor is None:
        preprocessor = transforms.Compose([
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize((0.4915, 0.4823, 0.4468), # Normalizing by mean and 
                                (0.2470, 0.2435, 0.2616))  # standard deviation.
        ])
    
    # load datasets
    data_train_val = data_set(
        data_path,       
        train=True,      
        download=True,
        transform=preprocessor)

    data_test = data_set(
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
    
    label_map = {0: 0, 2: 1}
    class_names = ['airplane', 'bird']

    # For each dataset, keep only airplanes and birds
    data_train_ = [(img, label_map[label]) for img, label in data_train if label in [0, 2]]
    data_val_ = [(img, label_map[label]) for img, label in data_val if label in [0, 2]]
    data_test_ = [(img, label_map[label]) for img, label in data_test if label in [0, 2]]
    
    print(f"Dataset: {str(data_set)}")
    print(f"Size of the train dataset:       , {len(data_train)}")
    print(f"Size of the validation dataset:  , {len(data_val)}")
    print(f"Size of the test dataset:        , {len(data_test)}")
    
    return (data_train_, data_val_, data_test_)
