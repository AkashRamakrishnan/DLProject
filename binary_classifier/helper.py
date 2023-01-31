import itertools
import numpy as np
import torch
from torch.autograd import Variable
from tqdm import tqdm
import matplotlib.pyplot as plt


def train_model(model, train_loader, valid_loader, optimizer, criterion, epochs, device):
        # Train the model
        print("Training model...")
        for e in range(epochs):
            # Set the model to training mode
            model.train()
            train_loss = 0
            # Iterate over the training data
            for images, labels in tqdm(train_loader):
                # Move the data to the device
                images, labels = images.to(device), labels.to(device)
                # Zero the gradients
                optimizer.zero_grad()
                # Forward pass
                output = model(images)
                loss = criterion(output, labels)
                # Backward pass
                if torch.cuda.is_available():
                    with torch.cuda.amp.autocast():
                        loss.backward()
                else:
                    loss.backward()
                # Update the weights
                optimizer.step()
                train_loss += loss.item()
            else:
                # Set the model to evaluation mode
                model.eval()
                valid_loss, correct = 0, 0
                total = 0
                # Iterate over the validation data
                with torch.no_grad():
                    for images, labels in valid_loader:
                        images, labels = images.to(device), labels.to(device)
                        output = model(images)
                        valid_loss += criterion(output, labels).item()
                        correct += torch.sum(torch.argmax(output, dim=1) == labels).item()
                        total += len(labels)
                accuracy = correct / total
                # Save the model if the validation loss is the lowest so far
                print(f"Epoch: {e+1}/{epochs}  Training loss: {train_loss/len(train_loader):.4f}  Validation loss: {valid_loss/len(valid_loader):.4f}  Validation accuracy: {accuracy:.4f}")
        return train_loss/len(train_loader), valid_loss/len(valid_loader), accuracy

def set_torch_device():
    # Check that MPS is available, if not, check if CUDA is available, if not, CPU
    device = torch.device('cpu')
    display_devices = False

    if not torch.backends.mps.is_available():
        # If cuda is available...
        if torch.cuda.is_available():
            # Find GPU with most free memory and set that as the device
            mem_usage_list = [torch.cuda.mem_get_info(f'cuda:{gpu_num}')[0] for gpu_num in range(torch.cuda.device_count())]
            most_free = mem_usage_list.index(max(mem_usage_list))
            device = torch.device(f'cuda:{most_free}')
            print(f'Setting the device to {device}...\n')

            if display_devices:
                # Print GPU info on all
                for gpu_num in range(torch.cuda.device_count()):
                    available_mem, total_mem = torch.cuda.mem_get_info(f'cuda:{gpu_num}')
                    print(f'cuda:{gpu_num}')
                    print('Memory Usage:')
                    print('Total:', round(total_mem/1024**3,2), 'GB')
                    print('Allocated:', round((total_mem-available_mem)/1024**3,2), 'GB')
                    print('Free:   ', round(available_mem/1024**3,2), 'GB')
                    print()
            # Set the default tensor type to gpu
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
    else:
        device = torch.device("mps")
    return device

# provide a mapping from label to class name
def get_output_labels(selected_classes=None):
    output_mapping = {
                 0: "T-shirt/Top",
                 1: "Trouser",
                 2: "Pullover",
                 3: "Dress",
                 4: "Coat", 
                 5: "Sandal", 
                 6: "Shirt",
                 7: "Sneaker",
                 8: "Bag",
                 9: "Ankle Boot"
                 }
    if selected_classes:
        output_mapping = {key: output_mapping[key] for key in selected_classes}
        for i in range(len(selected_classes)):
            output_mapping[i] = output_mapping.pop(selected_classes[i])
    return output_mapping


# get predictions
def print_accuracy(network, testloader, device, classes):
    class_correct = [0. for _ in range(10)]
    total_correct = [0. for _ in range(10)]

    acc_str = []

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            test = Variable(images)
            outputs = network(test)
            predicted = torch.max(outputs, 1)[1]
            correct = (predicted == labels).squeeze()

            for idx, label in enumerate(labels):
                class_correct[label] += correct[idx].item()
                total_correct[label] += 1
    for i in [0, 1]:
        msg = f"Accuracy of {classes[i]}: {class_correct[i] * 100 / total_correct[i]:.2f}%"
        acc_str.append(msg)
        print(msg)
    return acc_str


# define some helper functions
def get_item(preds, labels):
    """function that returns the accuracy of our architecture"""
    return preds.argmax(dim=1).eq(labels).sum().item()


@torch.no_grad()  # turn off gradients during inference for memory effieciency
def get_all_preds(network, dataloader):
    """function to return the number of correct predictions across data set"""
    all_preds = torch.tensor([])
    model = network
    for batch in dataloader:
        images, _ = batch
        preds = model(images)  # get preds
        # join along existing axis
        all_preds = torch.cat((all_preds, preds), dim=0)

    return all_preds


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """

    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(15, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()