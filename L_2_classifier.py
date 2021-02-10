from datetime import datetime

now = datetime.now()
time = now.strftime("%b-%d-%Y_%H-%M-%S")

import torch
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import yaml


import numpy as np

import importlib.util

from feature_eval.resnet_wider import resnet50x4

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)

class_similarity_config = yaml.load(open('./config_class_similarity.yaml', "r"), Loader=yaml.FullLoader)


def load_cifar10(download=True, shuffle=False, batch_size=512):
    train_dataset = datasets.CIFAR10('./data', download=download, train=True,
                                     transform=transforms.ToTensor())

    # train_dataset = datasets.CIFAR10('./data', download=download, train=True,
    #                                  transform=transforms.Compose([transforms.Resize(256), transforms.CenterCrop(256),
    #                                                                transforms.ToTensor()]))

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              num_workers=0, drop_last=False, shuffle=shuffle)

    test_dataset = datasets.CIFAR10('./data', download=download, train=False,
                                    transform=transforms.ToTensor())

    # test_dataset = datasets.CIFAR10('./data', download=download, train=False,
    #                                 transform=transforms.Compose([transforms.Resize(256), transforms.CenterCrop(256),
    #                                                               transforms.ToTensor()]))

    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             num_workers=0, drop_last=False, shuffle=shuffle)
    return train_loader, test_loader


def from_loader_to_np_array(loader):
    X_array = []
    y_array = []
    for batch_x, batch_y in loader:
        y_array.extend(batch_y)
        X_array.extend(batch_x.cpu().detach().numpy())

    X_array = np.array(X_array)
    y_array = np.array(y_array)

    print("Original data shape {}".format(X_array.shape))
    return X_array, y_array


# load CIFAR-10 train and test data. These are RAW data, not the representations learned by f in the contrastive setting
train_loader, test_loader = load_cifar10()
X_train, y_train = from_loader_to_np_array(train_loader)
X_test, y_test = from_loader_to_np_array(test_loader)

# Load the neural net module

# Code to load our default SimCLR pre-trained resnet:
spec = importlib.util.spec_from_file_location("model", './models/resnet_simclr.py')
resnet_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(resnet_module)

model = resnet_module.ResNetSimCLR(**class_similarity_config['model'])
model.eval()

state_dict = torch.load('./runs/Jul09_16-50-25_math-aogpu2/checkpoints/model.pth', map_location=torch.device('cpu'))
model.load_state_dict(state_dict)
model = model.to(device)

# Code to load ResNet50 (4x) converted to PyTorch from https://github.com/google-research/simclr
# by https://github.com/tonylins/simclr-converter :
# model = resnet50x4()
# model.eval()  #
# sd = './feature_eval/resnet50-4x.pth'
# sd = torch.load(sd, map_location='cpu')
# model.load_state_dict(sd['state_dict'])
# model = model.to(device)  #


def next_batch(X, y, batch_size):
    for i in range(0, X.shape[0], batch_size):
        # X_batch = torch.tensor(X[i: i + batch_size]) / 255.
        X_batch = torch.tensor(X[i: i + batch_size])
        y_batch = torch.tensor(y[i: i + batch_size])
        yield X_batch.to(device), y_batch.to(device)


X_train_feature = []

for batch_x, batch_y in next_batch(X_train, y_train, batch_size=512):
    features, _ = model(batch_x)
    # features = model(batch_x)
    X_train_feature.extend(features.cpu().detach().numpy())

X_train_feature = np.array(X_train_feature)

X_test_feature = []

for batch_x, batch_y in next_batch(X_test, y_test, batch_size=512):
    features, _ = model(batch_x)
    # features = model(batch_x)
    X_test_feature.extend(features.cpu().detach().numpy())

X_test_feature = np.array(X_test_feature)

# Now we have X_train_feature, y_train, X_test_feature, y_test as numpy arrays and with the correct format:
# X_train_feature has dimensions N_train x Rep.dimension; X_test_feature has dimension N_test x Rep.dimension.

# Normalizing the features
for row in range(X_train_feature.shape[0]):
    norm = np.linalg.norm(X_train_feature[row])
    X_train_feature[row] /= norm

for row in range(X_test_feature.shape[0]):
    norm = np.linalg.norm(X_test_feature[row])
    X_test_feature[row] /= norm

print("Processing the training set...")
# Defining the class weights for the training set
n_classes = np.unique(y_train).size
w = []
for _class in range(n_classes):
    class_indices = np.where(y_train == _class)[0]
    relevant_X_train_feature = X_train_feature[class_indices]
    class_weights_vector = np.average(relevant_X_train_feature, axis=0)
    class_weights_vector = class_weights_vector.tolist()
    w.append(class_weights_vector)

w = np.array(w)
# normalizing the weight vectors
for row in range(w.shape[0]):
    norm = np.linalg.norm(w[row])
    w[row] /= norm

# Computing the inner products between the different classes:
running_average = 0
counter = 0
for i in range(n_classes):
    for j in range(i + 1, n_classes):
        running_average += np.dot(w[i], w[j])
        counter += 1

running_average /= counter
print(f"Average (cosine) similarity between different class weights on the training set: {running_average}")

print("---------------------------------------------------------------------------------------")

print("Processing the test set...")
# Defining the class weights for the test set
n_classes = np.unique(y_test).size
w = []
for _class in range(n_classes):
    class_indices = np.where(y_test == _class)[0]
    relevant_X_test_feature = X_test_feature[class_indices]
    class_weights_vector = np.average(relevant_X_test_feature, axis=0)
    class_weights_vector = class_weights_vector.tolist()
    w.append(class_weights_vector)

w = np.array(w)
# normalizing the weight vectors
for row in range(w.shape[0]):
    norm = np.linalg.norm(w[row])
    w[row] /= norm

# Computing the inner products between the different classes:
running_average = 0
counter = 0
for i in range(n_classes):
    for j in range(i + 1, n_classes):
        running_average += np.dot(w[i], w[j])
        counter += 1

running_average /= counter
print(f"Average (cosine) similarity between different class weights on the test set: {running_average}")
