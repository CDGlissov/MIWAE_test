import os
import gzip
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch
from sklearn import preprocessing

# Function to read stupid MNIST files
def read_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels

#Loading FMNIST into np.array
def mnist_numpy(labels=False, data_type = "fmnist", normalize=False, binarize=False):
    """
    Read stupid fmnist files.
    :param labels: Output labels or not
    :return:
    """
    root = os.getcwd()
    data_type=data_type.lower()
    
    if(data_type=="fmnist"):
        x_train, y_train = read_mnist(root + '/data/FashionMNIST/raw', kind='train')
        x_test, y_test = read_mnist(root + '/data/FashionMNIST/raw', kind='t10k')
    elif(data_type=="mnist"):
        x_train, y_train = read_mnist(root + '/data/MNIST/raw', kind='train')
        x_test, y_test = read_mnist(root + '/data/MNIST/raw', kind='t10k')
        
    x_train = x_train[:].astype('float32') / 255
    y_train = y_train[:].astype('int32')
    
    x_test = x_test[:].astype('float32') / 255
    y_test = y_test[:].astype('int32')
    
    if normalize==True:
        x_train = (x_train-0.5)/0.5 #can also use x_train.mean()/x_train.std()
        x_test = (x_test-0.5)/0.5
    
    if binarize == True:
        binarizer = preprocessing.Binarizer(threshold=0.05)
        x_train=binarizer.transform(x_train)
        x_test=binarizer.transform(x_test)
        
    
    if labels:
        return x_train, y_train, x_test, y_test
    else:
        return x_train, x_test


#Adding missing values to data as zeros.
def data_corruption(data_train, data_test, mode='SnP', percentage=0.5, block_size=7):
    """
    Introduces missing values in the data.
    data expected to be quadratic

    :param data_train: Train data set
    :param data_test: Test data set
    :param mode: Salt and pepper or blocks of missing
    :param percentage: Percentage of SnP
    :param block_size: Size of block
    :return:
    """

    orig_dim = np.sqrt(data_train.shape[1])
    print('---------- Corrupting data ----------')
    if mode.lower() == 'snp':
        bernoulli = torch.distributions.bernoulli.Bernoulli(probs=percentage)
        mask_train = np.ones(data_train.shape) # Data corruption of train data
        for i, sample in enumerate(tqdm(data_train)):
            # mean_value = np.mean(sample)
            mask_train[i] = bernoulli.sample([data_train.shape[1]]).numpy()

        print('Test data:')
        mask_test = np.ones(data_test.shape) # Data corruption of test
        for i, sample in enumerate(tqdm(data_test)):
            # mean_value = np.mean(sample)
            mask_test[i] = bernoulli.sample([data_test.shape[1]]).numpy()

    if mode.lower() == 'block':

        dim_max = orig_dim - block_size
        mask_train = np.ones((data_train.shape[0], 28, 28))
        for i in tqdm(range(data_train.shape[0])):
            dim1 = np.random.randint(dim_max)
            dim2 = np.random.randint(dim_max)
            if np.random.uniform(0, 1) < percentage:
                mask_train[i, dim1:dim1 + block_size, dim2:dim2 + block_size] = 0
            else:
                mask_train[i, dim1:dim1 + block_size, dim2:dim2 + block_size] = 1
        mask_train = mask_train.reshape(data_train.shape[0], 784)

        mask_test = np.ones((data_test.shape[0], 28, 28))
        for i in tqdm(range(data_test.shape[0])):
            dim1 = np.random.randint(dim_max)
            dim2 = np.random.randint(dim_max)
            if np.random.uniform(0, 1) < percentage:
                mask_test[i, dim1:dim1 + block_size, dim2:dim2 + block_size] = 0
            else:
                mask_test[i, dim1:dim1 + block_size, dim2:dim2 + block_size] = 1
            # plt.imshow(mask_test[i])
        mask_test = mask_test.reshape(data_test.shape[0], 784)

    return mask_train, mask_test



