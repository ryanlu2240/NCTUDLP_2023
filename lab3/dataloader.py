import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


def read_bci_data():
    S4b_train = np.load('data/S4b_train.npz')
    X11b_train = np.load('data/X11b_train.npz')
    S4b_test = np.load('data/S4b_test.npz')
    X11b_test = np.load('data/X11b_test.npz')

    train_data = np.concatenate((S4b_train['signal'], X11b_train['signal']), axis=0)
    train_label = np.concatenate((S4b_train['label'], X11b_train['label']), axis=0)
    test_data = np.concatenate((S4b_test['signal'], X11b_test['signal']), axis=0)
    test_label = np.concatenate((S4b_test['label'], X11b_test['label']), axis=0)


    train_label = train_label - 1
    test_label = test_label -1
    train_data = np.transpose(np.expand_dims(train_data, axis=1), (0, 1, 3, 2))
    test_data = np.transpose(np.expand_dims(test_data, axis=1), (0, 1, 3, 2))
   

    mask = np.where(np.isnan(train_data))
    train_data[mask] = np.nanmean(train_data)

    mask = np.where(np.isnan(test_data))
    test_data[mask] = np.nanmean(test_data)

   

    return train_data, train_label, test_data, test_label

def create_dataset(args, device, train_data, train_label, test_data, test_label):
    """Create batch dataset."""
    print("Creating dataset...\n")
    ## Convert numpy array to torch.Tensor, move data to GPU
    x_train = torch.Tensor(train_data).to(device)
    x_test  = torch.Tensor(test_data).to(device)
    y_train = torch.LongTensor(train_label).to(device)
    y_test  = torch.LongTensor(test_label).to(device)

    train_dataset = TensorDataset(x_train, y_train)
    test_dataset  = TensorDataset(x_test , y_test)

    ## Create batch dataset
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader  = DataLoader(dataset=test_dataset , batch_size=args.batch_size, shuffle=False)

    return train_loader, test_loader


