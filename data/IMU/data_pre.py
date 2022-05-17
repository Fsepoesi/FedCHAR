import os
import numpy as np
import json
from sklearn.model_selection import train_test_split
import random
random.seed(0)

cluster_set = ['hsh_', 'mmw_']
class_set = ['_walk','_up','_down']

NUM_OF_USERS = 7
NUM_OF_CLASS = 3
DIMENSION_OF_FEATURE = 900
all_train = {'num_samples': [], 'users': [], 'user_data': {}}
all_test = {'num_samples': [], 'users': [], 'user_data': {}}

np.random.seed(0)
num_test = 50
num_train_set = np.random.randint(low=10, high=50, size=NUM_OF_USERS)
print('num_train_set', num_train_set)

for user_id in range(0, NUM_OF_USERS):
    if user_id < 4:
        cluster_id = 0
        intra_user_id = user_id + 1  # 0,1,2,3 to 1,2,3,4
    else:
        cluster_id = 1
        intra_user_id = user_id - 3  # 4,5,6 to 1,2,3

    cluster_des = cluster_set[cluster_id]
    uname = 'f_{0:05d}'.format(user_id)
    one_usr_X = []
    one_usr_Y = []

    count_one_usr_sample = 0
    for class_id in range(NUM_OF_CLASS):
        read_path = './raw_data/' + cluster_des + str(intra_user_id) + class_set[class_id] + '_nor.txt'
        if os.path.exists(read_path):
            temp_original_data = np.loadtxt(read_path, delimiter=',')
            one_class_X = temp_original_data.reshape(-1, DIMENSION_OF_FEATURE)
            count_img = one_class_X.shape[0]
            count_one_usr_sample += count_img
            one_class_Y = class_id * np.ones(count_img)
            one_usr_X.extend(one_class_X)
            one_usr_Y.extend(one_class_Y)

    one_usr_X = np.array(one_usr_X)
    one_usr_Y = np.array(one_usr_Y)

    test_percent = 0.4
    x_train, x_test, y_train, y_test = train_test_split(one_usr_X, one_usr_Y, test_size=test_percent, random_state=0)
    print('user {}-> all train num: {}, all test num: {}'.format(user_id, y_train.shape[0], y_test.shape[0]))
    np.random.seed(user_id)
    train_idx = np.random.choice(range(y_train.shape[0]), size=num_train_set[user_id], replace=False)
    test_idx = np.random.choice(range(y_test.shape[0]), size=num_test, replace=False)
    print('user {}-> train_idx: {}, test_idx: {}'.format(user_id, train_idx, test_idx))


    x_train = list(x_train[train_idx])
    y_train = list(y_train[train_idx])
    x_test = list(x_test[test_idx])
    y_test = list(y_test[test_idx])

    for i in range(len(x_train)):
        x_train[i] = list(x_train[i])
    for i in range(len(x_test)):
        x_test[i] = list(x_test[i])

    all_train['users'].append(uname)
    all_test['users'].append(uname)
    all_train['num_samples'].append(len(y_train))
    all_test['num_samples'].append(len(y_test))
    all_train['user_data'][uname] = {'y': y_train, 'x': x_train}
    all_test['user_data'][uname] = {'y': y_test, 'x': x_test}

with open('./data/train/train.json', 'w') as outfile:
    json.dump(all_train, outfile)

with open('./data/test/test.json', 'w') as outfile:
    json.dump(all_test, outfile)
