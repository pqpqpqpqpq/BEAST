#import scipy.sparse as sp
import numpy as np
#from scipy.sparse.linalg.eigen.arpack import eigsh, ArpackNoConvergence
import boto3
import os
import torch
import torch.nn.functional as F
from sklearn.model_selection import ShuffleSplit,StratifiedShuffleSplit


def normalize_adj_tensor(adj, symmetric=True):
    """
    Normalize the adjacency matrix.
    If symmetric=True, it normalizes as D^(-1/2) * A * D^(-1/2),
    otherwise, it normalizes as D^(-1) * A.
    """
    if symmetric:
        row_sum = adj.sum(1)  # Sum along the rows (degree matrix)
        d_inv_sqrt = torch.pow(row_sum, -0.5)  # D^(-1/2)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0  # Handle division by zero
        d_inv_sqrt = torch.diag(d_inv_sqrt)  # Create diagonal matrix D^(-1/2)
        a_norm = torch.matmul(torch.matmul(d_inv_sqrt, adj), d_inv_sqrt)  # D^(-1/2) * A * D^(-1/2)
    else:
        row_sum = adj.sum(1)  # Sum along the rows (degree matrix)
        d_inv = torch.pow(row_sum, -1)  # D^(-1)
        d_inv[torch.isinf(d_inv)] = 0  # Handle division by zero
        d_inv = torch.diag(d_inv)  # Create diagonal matrix D^(-1)
        a_norm = torch.matmul(d_inv, adj)  # D^(-1) * A

    return a_norm


def preprocess_adj_tensor_with_identity(adj_tensor, symmetric=True):
    """
    Preprocess the adjacency tensor with identity matrix added and normalized.
    adj_tensor: [batch, num_frames, V, V] (e.g., [batch_size, 6, 23, 23])
    """
    batch_size, num_frames, num_nodes, _ = adj_tensor.shape  # Get shape of the tensor
    adj_out_tensor = []

    for i in range(batch_size):
        adj_one_tensor = []
        adj = adj_tensor[i]  # adj is now shape [num_frames, num_nodes, num_nodes]

        for j in range(num_frames):
            adj_one = adj[j]  # Shape [num_nodes, num_nodes]

            # Add identity matrix
            identity_matrix = torch.eye(adj_one.size(0), device=adj.device)
            adj_one = adj_one + identity_matrix  # Add identity matrix I

            # Normalize the adjacency matrix
            adj_one = normalize_adj_tensor(adj_one, symmetric)

            adj_one_tensor.append(adj_one)  # Append processed adjacency matrix for the current frame

        adj_out_tensor.append(torch.stack(adj_one_tensor))  # Stack the processed frames for the current batch

    return torch.stack(adj_out_tensor)  # Shape [batch_size, num_frames, num_nodes+1, num_nodes]


def initialize_filters(A):
    """
    Initialize filters by preprocessing the adjacency matrix A.
    :param A: adjacency matrix tensor [N, V, V]
    :return: filters [N, V+1, V]
    """
    filters = preprocess_adj_tensor_with_identity(A, symmetric=True)
    return filters


def kmer_parser(fn, exclude_base=None):
    '''
    Function parses kmer file and returns
    Parameters
    ----------
    fn: str
        path to file
    exclude_base : str
        base to exclude from kmer_list. The base selected will
        be removed regardless its position. All kmers containing
        that base will not be returned

    Returns
    ----------
    kmer_list: array, list of kmers in the order they appear in the file
    pA_list: array, list of pA values (floats) in the same order
    '''
    kmer_list = []
    pA_list = []
    label_list = []
    with open(fn, 'r') as f:
        lines = f.readlines()

        for line in lines:

            if '\t' in line:
                line = line.split('\t')
            elif ' ' in line:
                line = line.split(' ')
            if len(line) > 2:
                label = int(line[2].strip())
                label_list += [label]

            kmer = str(line[0]).strip()
            pA = float(line[1].strip())
            if exclude_base is not None:
                if exclude_base in kmer:
                    continue

            kmer_list += [kmer]
            pA_list += [pA]

    if len(label_list) == 0:
        label_list = None

        return np.array(kmer_list), np.array(pA_list), label_list
    else:
        return np.array(kmer_list), np.array(pA_list), np.array(label_list)


def cv_folds(X, Y, folds, test_sizes, labels=None ):
    '''
    Parameters
    -----------
    X : array
        list of samples

    Y : array
         list of values to predict

    labels : array
         list of labels to be used for stratified split

    folds : int
         number of CV folds to be made

    test_sizes : array
        Array to test sizes

    Returns
    -----------
    test_size : float

    kmer_train_mat : mat
        shape(folds, train_size) for each train/test split

    kmer_test_mat : mat
        shape(folds, train_size) for each train/test split

    pA_train_mat : mat
        shape(folds, test_size) for each train/test split

    pA_test_mat : mat
        shape(folds, test_size) for each train/test split

    '''


    for test_size in test_sizes:  # np.arange(0.05,1.0.05),

        # The following matrices contain the train/test kmer and corresponding pA values
        # for the folds produced. Shape is (folds, train/test size)
        kmer_train_mat = []
        kmer_test_mat = []

        pA_train_mat = []
        pA_test_mat = []

        if labels is not None:

            splitter = StratifiedShuffleSplit(n_splits=folds, test_size=test_size, random_state=42).split(X, labels)

        else:
            splitter = ShuffleSplit(n_splits=folds, test_size=test_size, random_state=42).split(X)


        for train_idx, test_idx in (splitter):
            x_train = X[train_idx]
            x_test = X[test_idx]

            y_train = Y[train_idx]
            y_test = Y[test_idx]

            if all(isinstance(kmer, str) for kmer in x_train):
                print("worked")
                x_train = x_train.flatten()
                x_test = x_test.flatten()

                kmer_train_mat += [x_train]
                kmer_test_mat += [x_test]

            else:

                kmer_train_mat += [np.vstack(x_train)]
                kmer_test_mat += [np.vstack(x_test)]

            pA_train_mat += [y_train]
            pA_test_mat += [y_test]

        pA_train_mat = np.vstack(pA_train_mat)
        pA_test_mat = np.vstack(pA_test_mat)

        if all(isinstance(kmer, str) for kmer in x_train):
            kmer_train_mat = np.vstack(kmer_train_mat)
            kmer_test_mat = np.vstack(kmer_test_mat)

        else:
            kmer_train_mat = np.array(kmer_train_mat)
            kmer_test_mat = np.array(kmer_test_mat)

        yield test_size, kmer_train_mat, kmer_test_mat, pA_train_mat, pA_test_mat

