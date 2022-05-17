import numpy as np
import json
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()



def process_grad(grads):
    '''
    Args:
        grads: grad
    Return:
        a flattened grad in numpy (1-D array)
    '''

    client_grads = grads[0]

    for i in range(1, len(grads)):
        client_grads = np.append(client_grads, grads[i]) # output a flattened array


    return client_grads

def l2_clip(cgrads):

    # input: grads (or model updates, or models) from all selected clients
    # output: clipped grads in the same shape

    flattened_grads = []
    for i in range(len(cgrads)):
        flattened_grads.append(process_grad(cgrads[i]))
    norms = [np.linalg.norm(u) for u in flattened_grads]

    clipping_threshold = np.median(norms)

    clipped = []
    for grads in cgrads:
        norm_ = np.linalg.norm(process_grad(grads), 2)
        if norm_ > clipping_threshold:
            clipped.append([u * (clipping_threshold * 1.0) / (norm_ + 1e-10) for u in grads])
        else:
            clipped.append(grads)

    return clipped


def get_stdev(parameters):

    # input: the model parameters
    # output: the standard deviation of the flattened vector

    flattened_param = process_grad(parameters)
    return np.std(flattened_param)


def read_data(train_data_dir, test_data_dir):
    '''parses data in given train and test data directories

    assumes:
    - the data in the input directories are .json files with
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users

    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    '''
    clients = []
    groups = []  # del
    train_data = {}
    test_data = {}

    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith('.json')]
    for f in train_files:
        file_path = os.path.join(train_data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])  # ["f_00010", "f_00011", "f_00012", ... ,]
        if 'hierarchies' in cdata:  # del
            groups.extend(cdata['hierarchies'])
        train_data.update(cdata['user_data'])  # {"f_00010":{'y':[],'x':[[]]}...}

    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith('.json')]
    for f in test_files:
        file_path = os.path.join(test_data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        test_data.update(cdata['user_data'])

    clients = list(sorted(train_data.keys()))

    return clients, groups, train_data, test_data

