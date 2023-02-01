import numpy as np
import torch

from cnn_classifier import FashionCNN
from torch import nn
from helper import train_model, print_accuracy, get_output_labels

from torch.utils.data.sampler import SubsetRandomSampler

# Preparing the splits
def split_dataset(train_set, device, n_splits):
    indices = list(range(len(train_set)))
    np.random.shuffle(indices)
    split_size = int(np.floor(1/n_splits * len(train_set)))
    datasets = []
    dataloaders = []
    
    for i in range(n_splits):
        start = i * split_size
        end = start + split_size
        curr_indices = indices[start:end]
        
        # Get valid indicies and train indicies from the curr
        np.random.shuffle(curr_indices)
        # to get 20% of the train set
        split = int(np.floor(0.2 * len(curr_indices)))
        
        valid_sample = SubsetRandomSampler(curr_indices[:split])
        train_sample = SubsetRandomSampler(curr_indices[split:])
        datasets.append((train_sample, valid_sample))

        train_loader = torch.utils.data.DataLoader(train_set, sampler=train_sample, batch_size=64, generator=torch.Generator(device))
        valid_loader = torch.utils.data.DataLoader(train_set, sampler=valid_sample, batch_size=64, generator=torch.Generator(device))
        dataloaders.append((train_loader, valid_loader))
        
    return datasets, dataloaders

def train_split_models(train_set, testloader, classes, device, epochs=10, n_splits=10, class_labels=None):
    models = [FashionCNN(len(classes)).to(device) for i in range(n_splits)]
    datasets, dataloaders = split_dataset(train_set, device, n_splits)

    if class_labels is None:
        class_labels = get_output_labels(classes)
    model_class_acc = {}
    
    model_avg_train_loss = []
    model_avg_valid_loss = []
    model_avg_valid_acc = []

    # initialize the ensemble model
    for idx, model in enumerate(models):
        optimizer = torch.optim.Adam(model.parameters(), lr = 0.005)
        criterion = nn.CrossEntropyLoss()
        train_loss, valid_loss, valid_acc = train_model(
            model, dataloaders[0][0], dataloaders[0][1],
            optimizer, criterion, epochs, device
        )
        model_avg_train_loss.append(train_loss)
        model_avg_valid_loss.append(valid_loss)
        model_avg_valid_acc.append(valid_acc)

        model_class_acc[idx] = print_accuracy(model, testloader, device, class_labels)

    
    avg_train_loss = sum(model_avg_train_loss)/n_splits
    avg_valid_loss = sum(model_avg_valid_loss)/n_splits
    avg_valid_acc = sum(model_avg_valid_acc)/n_splits
    return avg_train_loss, avg_valid_loss, avg_valid_acc, model_class_acc