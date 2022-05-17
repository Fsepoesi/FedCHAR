import numpy as np
from tqdm import tqdm
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import copy
import math
from .base import BaseFedarated
from .utils import get_stdev, l2_clip


class Server(BaseFedarated):#------------1
    def __init__(self, params, learner, dataset):
        if params['dataset'] == 'FMCW':
            self.inner_opt = tf.train.AdamOptimizer(params['learning_rate'])
        else:
            self.inner_opt = tf.train.GradientDescentOptimizer(params['learning_rate'])
        super(Server, self).__init__(params, learner, dataset)
        self.num_classes = params['model_params'][0]
    def train(self):
        print('---{} workers per communication round---'.format(self.clients_per_round))
        np.random.seed(self.corrupt_seed)
        corrupt_id = np.random.choice(range(len(self.clients)), size=self.num_corrupted, replace=False)
        print("corrupt_id: {}".format(corrupt_id))

        for idx, c in enumerate(self.clients):
            if idx in corrupt_id and self.label_flipping == 1:
                c.train_data['y'] = np.asarray(c.train_data['y'])
                if self.dataset == 'HARBox':
                    c.train_data['y'] = np.random.randint(0, 5, len(c.train_data['y']))
                elif self.dataset == 'IMU':
                    c.train_data['y'] = np.random.randint(0, 3, len(c.train_data['y']))
                elif self.dataset == 'Depth':
                    c.train_data['y'] = np.random.randint(0, 5, len(c.train_data['y']))
                elif self.dataset == 'UWB':
                    c.train_data['y'] = 1 - c.train_data['y']
                elif self.dataset == 'FMCW':
                    c.train_data['y'] = np.random.randint(0, 6, len(c.train_data['y']))

        #initial_rounds
        for i in range(self.initial_rounds):

            indices, selected_clients = self.select_clients(round=i, num_clients=self.clients_per_round)
            print("initial_round {} selected clients indices: {}".format(i, indices))
            csolns = []

            for idx in indices:
                w_global_idx = copy.deepcopy(self.global_model)
                c = self.clients[idx]
                for epoch_i in range(self.epoch):
                    num_batch = int(math.ceil(len(c.train_data['y']) / self.batch_size))
                    train_x, train_y = self.shuffle_training_data(c.train_data['x'], c.train_data['y'], epoch_i * i + epoch_i)
                    for s in range(num_batch):
                        batch_xs = train_x[s * self.batch_size: (s + 1) * self.batch_size]
                        batch_ys = train_y[s * self.batch_size: (s + 1) * self.batch_size]
                        data_batch = (batch_xs, batch_ys)
                        # local
                        self.client_model.set_params(self.local_models[idx])
                        _, grads, _ = c.solve_sgd(data_batch)

                        for layer in range(len(grads[1])):
                            eff_grad = grads[1][layer] + self.lam * (self.local_models[idx][layer] - self.global_model[layer])
                            self.local_models[idx][layer] = self.local_models[idx][layer] - self.learning_rate * eff_grad

                        # global
                        self.client_model.set_params(w_global_idx)
                        _, grads, _ = c.solve_sgd(data_batch)
                        w_global_idx = self.client_model.get_params()

                # get the difference (global model updates)
                diff = [u - v for (u, v) in zip(w_global_idx, self.global_model)]


                # send the malicious updates
                if idx in corrupt_id:
                    if self.boosting:
                        # scale malicious updates
                        diff = [10 * u for u in diff]
                    elif self.random_updates:
                        # send random updates
                        stdev_ = get_stdev(diff)
                        diff = [np.random.normal(0, stdev_, size=u.shape) for u in diff]
                    elif self.inner_opt:
                        diff = [-1 * u for u in diff]
                csolns.append(diff)

            avg_updates = self.simple_average(csolns)

            # update the global model
            for layer in range(len(avg_updates)):
                self.global_model[layer] += avg_updates[layer]
        print("initial rounds are over")
        indices = np.arange(len(self.clients))
        csolns = []

        #clustering stage
        for idx in indices:
            w_global_idx = copy.deepcopy(self.global_model)
            c = self.clients[idx]
            for epoch_i in range(self.epoch):
                num_batch = int(math.ceil(len(c.train_data['y']) / self.batch_size))
                train_x, train_y = self.shuffle_training_data(c.train_data['x'], c.train_data['y'], self.epoch * self.initial_rounds + epoch_i)
                for s in range(num_batch):
                    batch_xs = train_x[s * self.batch_size: (s + 1) * self.batch_size]
                    batch_ys = train_y[s * self.batch_size: (s + 1) * self.batch_size]
                    data_batch = (batch_xs, batch_ys)
                    # global
                    self.client_model.set_params(w_global_idx)
                    _, grads, _ = c.solve_sgd(data_batch)
                    w_global_idx = self.client_model.get_params()

            # get the difference (model updates)
            diff = [u - v for (u, v) in zip(w_global_idx, self.global_model)]

            # send the malicious updates
            if idx in corrupt_id:
                if self.boosting:
                    # scale malicious updates
                    diff = [10 * u for u in diff]
                elif self.random_updates:
                    # send random updates
                    stdev_ = get_stdev(diff)
                    diff = [np.random.normal(0, stdev_, size=u.shape) for u in diff]
                elif self.inner_opt:
                    diff = [-1 * u for u in diff]
            csolns.append(diff)
        G = self.hierarchical_clustering(csolns, indices)
        print("Clusters' Structure: {}".format(G))

        Groups_details={}
        CM = []
        test_CM = []
        for g in range(len(G)):
            group_c = []

            for g_usr in G[g]:
                group_c.append(self.clients[g_usr])

            g_train_acc = []
            g_train_loss = []
            g_val_acc = []
            g_val_loss = []
            g_benign_train_acc = []
            g_benign_train_loss = []
            g_benign_val_acc = []
            g_benign_val_loss = []
            g_malicious_train_acc = []
            g_malicious_train_loss = []
            g_malicious_val_acc = []
            g_malicious_val_loss = []
            g_var_of_performance = []

            print("Now {}th group in G starts training".format(g))
            self.group_model = copy.deepcopy(self.global_model)

            if len(G[g]) < self.clients_per_round:
                num_clients = len(G[g])
            else:
                num_clients = self.clients_per_round

            for i in range(self.remain_rounds + 1):
                if i % self.eval_every == 0 and i > 0:
                    tmp_models = []
                    for idx in G[g]:
                        tmp_models.append(self.local_models[idx])

                    num_train, num_correct_train, train_loss_vector = self.train_error(tmp_models, group_c)
                    num_val, num_correct_val, val_loss_vector, Group_cm = self.validate(tmp_models, group_c)
                    if i == self.remain_rounds:
                        CM.append(Group_cm)
                        num_test, num_correct_test, test_loss_vector, test_Group_cm = self.test(tmp_models, group_c)
                        test_CM.append(test_Group_cm)
                        test_acc = np.sum(num_correct_test) * 1.0 / np.sum(num_test)
                        tqdm.write('Final group {} test accu: {}'.format(i, g, test_acc))
                    train_acc_once = np.sum(num_correct_train) * 1.0 / np.sum(num_train)
                    avg_train_loss = np.dot(train_loss_vector, num_train) / np.sum(num_train)

                    val_acc_once = np.sum(num_correct_val) * 1.0 / np.sum(num_val)
                    avg_val_loss = np.dot(val_loss_vector, num_val) / np.sum(num_val)

                    tqdm.write('At round {} group {} training accu: {}, loss: {}'.format(i, g, train_acc_once, avg_train_loss))
                    tqdm.write('At round {} group {} val accu: {}, loss: {}'.format(i, g, val_acc_once, avg_val_loss))

                    c_id = []
                    nc_id = []

                    for idx, j in enumerate(G[g]):
                        if j in corrupt_id:
                            c_id.append(idx)
                        else:
                            nc_id.append(idx)
                    if nc_id == []:
                        break
                    malicious_train_loss_once = np.dot(train_loss_vector[c_id], num_train[c_id]) / np.sum(
                        num_train[c_id])
                    benign_train_loss_once = np.dot(train_loss_vector[nc_id], num_train[nc_id]) / np.sum(
                        num_train[nc_id])
                    malicious_val_loss_once = np.dot(val_loss_vector[c_id], num_val[c_id]) / np.sum(
                        num_val[c_id])
                    benign_val_loss_once = np.dot(val_loss_vector[nc_id], num_val[nc_id]) / np.sum(
                        num_val[nc_id])

                    malicious_train_acc_once = np.sum(num_correct_train[c_id]) * 1.0 / np.sum(num_train[c_id])
                    malicious_val_acc_once = np.sum(num_correct_val[c_id]) * 1.0 / np.sum(num_val[c_id])
                    benign_train_acc_once = np.sum(num_correct_train[nc_id]) * 1.0 / np.sum(num_train[nc_id])
                    benign_val_acc_once = np.sum(num_correct_val[nc_id]) * 1.0 / np.sum(num_val[nc_id])
                    if i == self.remain_rounds:
                        malicious_test_acc = np.sum(num_correct_test[c_id]) * 1.0 / np.sum(num_test[c_id])
                        benign_test_acc = np.sum(num_correct_test[nc_id]) * 1.0 / np.sum(num_test[nc_id])
                        tqdm.write('Final group {} malicious test accu: {}'.format(g, malicious_test_acc))
                        tqdm.write('Final group {} benign test accu: {}'.format(g, benign_test_acc))
                        Groups_details["Group" + str(g)] = (num_correct_test[nc_id], num_test[nc_id], num_correct_test[nc_id] / num_test[nc_id])

                    var_of_performance_once = np.var(num_correct_val[nc_id] / num_val[nc_id])

                    usr_in_group_acc = []
                    for user_id, user_acc in zip(np.array(G[g])[np.array(nc_id)], num_correct_val[nc_id]/num_val[nc_id]):
                        usr_in_group_acc.append((user_id, user_acc))

                    tqdm.write('At round {} group {} malicious training accu: {}, loss: {}'.format(i, g, malicious_train_acc_once,
                                                                                          malicious_train_loss_once))
                    tqdm.write('At round {} group {} malicious val accu: {}, loss: {}'.format(i, g, malicious_val_acc_once,
                                                                                      malicious_val_loss_once))

                    tqdm.write('At round {} group {} benign training accu: {}, loss: {}'.format(i, g, benign_train_acc_once,
                                                                                       benign_train_loss_once))
                    tqdm.write(
                        'At round {} group {} benign val accu: {}, loss: {}'.format(i, g, benign_val_acc_once,
                                                                                         benign_val_loss_once))


                    tqdm.write('At round {} users in group {} benign val accu: {}'.format(i, g, usr_in_group_acc))

                    tqdm.write("group {} variance of the performance: {}".format(g, var_of_performance_once))  # fairness

                    g_train_acc.append(train_acc_once)
                    g_train_loss.append(avg_train_loss)
                    g_val_acc.append(val_acc_once)
                    g_val_loss.append(avg_val_loss)

                    g_benign_train_acc.append(benign_train_acc_once)
                    g_benign_val_acc.append(benign_val_acc_once)
                    g_benign_train_loss.append(benign_train_loss_once)
                    g_benign_val_loss.append(benign_val_loss_once)

                    g_malicious_train_acc.append(malicious_train_acc_once)
                    g_malicious_val_acc.append(malicious_val_acc_once)
                    g_malicious_train_loss.append(malicious_train_loss_once)
                    g_malicious_val_loss.append(malicious_val_loss_once)
                    g_var_of_performance.append(var_of_performance_once)

                    if i == self.remain_rounds:
                        break
                indices, selected_clients = self.select_clients_intraG(i+self.initial_rounds, num_clients, G[g], group_c)

                csolns = []

                for idx in indices:
                    w_global_idx = copy.deepcopy(self.group_model)
                    c = self.clients[idx]
                    for epoch_i in range(self.epoch):
                        num_batch = int(math.ceil(len(c.train_data['y']) / self.batch_size))
                        train_x, train_y = self.shuffle_training_data(c.train_data['x'], c.train_data['y'], epoch_i + self.epoch * self.initial_rounds + self.epoch)
                        for s in range(num_batch):
                            batch_xs = train_x[s * self.batch_size: (s + 1) * self.batch_size]
                            batch_ys = train_y[s * self.batch_size: (s + 1) * self.batch_size]
                            data_batch = (batch_xs, batch_ys)
                            # local
                            self.client_model.set_params(self.local_models[idx])
                            _, grads, _ = c.solve_sgd(data_batch)  # weights,grads,loss

                            for layer in range(len(grads[1])):
                                eff_grad = grads[1][layer] + self.lam * (self.local_models[idx][layer] - self.group_model[layer])
                                self.local_models[idx][layer] = self.local_models[idx][layer] - self.learning_rate * eff_grad

                            # global
                            self.client_model.set_params(w_global_idx)

                            _, grads, _ = c.solve_sgd(data_batch)
                            w_global_idx = self.client_model.get_params()

                    # get the difference (global model updates)
                    diff = [u - v for (u, v) in zip(w_global_idx, self.group_model)]

                    # send the malicious updates
                    if idx in corrupt_id:
                        if self.boosting:
                            # scale malicious updates
                            diff = [10 * u for u in diff]
                        elif self.random_updates:
                            # send random updates
                            stdev_ = get_stdev(diff)
                            diff = [np.random.normal(0, stdev_, size=u.shape) for u in diff]
                        elif self.inner_opt:
                            diff = [-1 * u for u in diff]
                    csolns.append(diff)

                if self.gradient_clipping:
                    csolns = l2_clip(csolns)

                expected_num_mali = int(self.clients_per_round * self.num_corrupted / len(self.clients))

                if self.median:
                    avg_updates = self.median_average(csolns)
                elif self.k_norm:
                    avg_updates = self.k_norm_average(self.clients_per_round - expected_num_mali, csolns)
                elif self.krum:
                    avg_updates = self.krum_average(self.clients_per_round - expected_num_mali - 2, csolns)
                elif self.multi_krum:
                    m = self.clients_per_round - expected_num_mali
                    avg_updates = self.mkrum_average(self.clients_per_round - expected_num_mali - 2, m, csolns)
                else:
                    avg_updates = self.simple_average(csolns)

                # update the global model
                for layer in range(len(avg_updates)):
                    self.group_model[layer] += avg_updates[layer]


            print("group {} train_acc:{}".format(g, g_train_acc))
            print("group {} train_loss:{}".format(g, g_train_loss))
            print("group {} val_acc:{}".format(g, g_val_acc))
            print("group {} val_loss:{}".format(g, g_val_loss))
            print("group {} benign_train_acc:{}".format(g, g_benign_train_acc))
            print("group {} benign_train_loss:{}".format(g, g_benign_train_loss))
            print("group {} benign_val_acc:{}".format(g, g_benign_val_acc))
            print("group {} benign_val_loss:{}".format(g, g_benign_val_loss))
            print("group {} malicious_train_acc:{}".format(g, g_malicious_train_acc))
            print("group {} malicious_val_acc:{}".format(g, g_malicious_val_acc))
            print("group {} malicious_train_loss:{}".format(g, g_malicious_train_loss))
            print("group {} malicious_val_loss:{}".format(g, g_malicious_val_loss))
            print("group {} var_of_performance:{}".format(g, g_var_of_performance))
            print("Now {}th group in G ends training".format(g))

        print("**********************************")
        print("CM:\n", CM)


        all_num_correct = np.array([])
        all_num_test = np.array([])
        group_acc_vec = np.array([])
        for g in range(len(G)):
            try:
                all_num_correct = np.append(all_num_correct, Groups_details["Group"+str(g)][0])
                all_num_test = np.append(all_num_test, Groups_details["Group"+str(g)][1])
                group_acc_vec = np.append(group_acc_vec, Groups_details["Group"+str(g)][2])
            except KeyError:
                pass
        benign_test_acc = (np.sum(all_num_correct) / np.sum(all_num_test))
        var_of_performance = np.var(group_acc_vec)
        print("benign_test_acc: {}".format(benign_test_acc))
        print("var_of_performance: {}".format(var_of_performance))



        sum_test_CM = np.zeros((self.num_classes, self.num_classes))
        for i in range(len(test_CM)):
            sum_test_CM += test_CM[i].sum(axis=0)

        print("[NC]Confusion Matrix:\n ", sum_test_CM)
        sum_test_CM = sum_test_CM.astype('float') / sum_test_CM.sum(axis=1)[:, np.newaxis]
        sum_test_CM = np.around(sum_test_CM, decimals=2)
        print("[ACC]Confusion Matrix:\n ", sum_test_CM)

