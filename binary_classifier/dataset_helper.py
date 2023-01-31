import numpy as np

import torch
from torch.utils.data import Dataset, Subset
from torch.utils.data.sampler import SubsetRandomSampler

class BinaryFashionDataset(Dataset):
    """User defined class to build a datset using Pytorch class Dataset."""

    def __init__(self, X, Y, transform = None):
        """Method to initilaize variables."""
        self.images = X
        self.labels = Y
        self.transform = transform

    def __getitem__(self, index):
        label = self.labels[index]
        image = self.images[index]
        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.images)

def create_poison_dataset(x_train, y_train, selected_classes, selected_indices=None, percentage_of_infected=0.1):
    if selected_indices is None:
        selected_indices =  np.arange(len(x_train))
    # generate the infected random indices from the training sample
    generatePoisonIdx = np.random.choice(selected_indices, size=(int(percentage_of_infected * len(selected_indices)),), replace=False)

    # indexing all the infected
    x_pois_train, y_pois_train = x_train[generatePoisonIdx], y_train[generatePoisonIdx]
    # indexing all the noromal
    unpoisoned_indices = np.setdiff1d(selected_indices, generatePoisonIdx)
    x_train_new, y_train_new = x_train[unpoisoned_indices], y_train[unpoisoned_indices]

    # mix up the labels(infecting the infected)
    random_labels = np.random.choice(len(selected_classes), size=(len(y_pois_train),), replace=True)
    for idx in range(len(y_pois_train)):
        if y_pois_train[idx] == random_labels[idx]:
            random_labels[idx] = (random_labels[idx] + 1) % len(selected_classes)
        y_pois_train[idx] = random_labels[idx]

    x_poison_train = np.concatenate((x_train_new, x_pois_train))
    y_poison_train = np.concatenate((y_train_new, y_pois_train))

    print("Training samples after infection : ", x_poison_train.shape)
    print("Labels samples after infection : ", y_poison_train.shape)

    return x_poison_train, y_poison_train

def get_data_loaders(trainset, validation_split, device):
    # Preparing for validaion test
    indices = list(range(len(trainset)))
    np.random.shuffle(indices)
    # to get 20% of the train set for validation
    split = int(np.floor(validation_split * len(trainset)))
    train_sample = SubsetRandomSampler(indices[split:])
    valid_sample = SubsetRandomSampler(indices[:split])

    # Data Loader
    trainloader = torch.utils.data.DataLoader(trainset, sampler=train_sample, batch_size=64, generator=torch.Generator(device))
    validloader = torch.utils.data.DataLoader(trainset, sampler=valid_sample, batch_size=64, generator=torch.Generator(device))
    return trainloader, validloader

def get_class_subset(trainset, testset, selected_classes):
    train_idx = np.zeros(0, dtype=int)
    test_idx = np.zeros(0, dtype=int)

    for clazz in selected_classes:
        train_idx_sub = np.where((trainset.targets==clazz))[0]
        train_idx = np.concatenate((train_idx, train_idx_sub), axis=0)
        
        test_idx_sub = np.where((testset.targets==clazz))[0]
        test_idx = np.concatenate((test_idx, test_idx_sub), axis=0)

    train_subset = Subset(trainset, train_idx)
    test_subset = Subset(testset, test_idx)

    for i in range(len(selected_classes)):
        train_subset.dataset.targets[train_subset.dataset.targets==selected_classes[i]] = i
        test_subset.dataset.targets[test_subset.dataset.targets==selected_classes[i]] = i
    
    return train_idx, train_subset, test_subset