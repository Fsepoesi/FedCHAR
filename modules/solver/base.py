import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import copy
from .client import Client
from .utils import process_grad
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster


class BaseFedarated(object):
    def __init__(self, params, learner, dataset):
        # transfer parameters to self
        for key, val in params.items():
            setattr(self, key, val)

        # create worker nodes
        tf.reset_default_graph()
        self.client_model = learner(*params['model_params'], self.inner_opt, self.seed)
        self.clients = self.setup_clients(dataset, self.client_model)

        print('{} Clients in Total'.format(len(self.clients)))
        self.latest_model = copy.deepcopy(self.client_model.get_params())
        self.local_models = []
        self.global_model = copy.deepcopy(self.latest_model)
        for _ in self.clients:
            self.local_models.append(copy.deepcopy(self.latest_model))


        # # initialize system metrics
        # self.metrics = Metrics(self.clients, params)

    def __del__(self):
        self.client_model.close()

    def setup_clients(self, dataset, model=None):
        '''instantiates clients based on given train and test data directories

        Return:
            list of Clients
        '''
        users, groups, train_data, test_data = dataset
        if len(groups) == 0:
            groups = [None for _ in users]
        all_clients = [Client(u, g, train_data[u], test_data[u], model) for u, g in zip(users, groups)]
        return all_clients

    def train_error(self, models, group_c):
        num_samples = []
        tot_correct = []
        losses = []

        for idx, c in enumerate(group_c):
            self.client_model.set_params(models[idx])
            ct, cl, ns = c.train_error()
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
            losses.append(cl*1.0)

        return np.array(num_samples), np.array(tot_correct), np.array(losses)

    def test(self, models, group_c):
        num_samples = []
        tot_correct = []
        losses = []
        Group_cm = []

        for idx, c in enumerate(group_c):
            self.client_model.set_params(models[idx])
            ct, cl, ns, cm = c.test()
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
            losses.append(cl * 1.0)
            Group_cm.append(cm)
        return np.array(num_samples), np.array(tot_correct), np.array(losses), np.array(Group_cm)

    def validate(self, models, group_c):
        num_samples = []
        tot_correct = []
        losses = []
        Group_cm = []

        for idx, c in enumerate(group_c):
            self.client_model.set_params(models[idx])
            ct, cl, ns, cm = c.validate()
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
            losses.append(cl * 1.0)
            Group_cm.append(cm)
        return np.array(num_samples), np.array(tot_correct), np.array(losses), np.array(Group_cm)

    def save(self):
        pass

    def shuffle_training_data(self, train_data, train_labels, epoch_i):
        seed = epoch_i
        np.random.seed(seed)
        np.random.shuffle(train_data)
        np.random.seed(seed)
        np.random.shuffle(train_labels)
        return np.array(train_data), np.array(train_labels)

    def select_clients(self, round, num_clients=20):

        num_clients = min(num_clients, len(self.clients))

        np.random.seed(round)

        if self.sampling == 1:
            pk = np.ones(num_clients) / num_clients
            indices = np.random.choice(range(len(self.clients)), num_clients, replace=False, p=pk)
            return indices, np.asarray(self.clients)[indices]
        elif self.sampling == 2:
            num_samples = []
            for client in self.clients:
                num_samples.append(client.train_samples)
            total_samples = np.sum(np.asarray(num_samples))
            pk = [item * 1.0 / total_samples for item in num_samples]
            indices = np.random.choice(range(len(self.clients)), num_clients, replace=False, p=pk)
            return indices, np.asarray(self.clients)[indices]

    def select_clients_intraG(self, round, num_clients, group, group_c):

        num_clients = min(num_clients, len(self.clients))

        np.random.seed(round)

        if self.sampling == 1:
            pk = np.ones(len(group)) / len(group)
            indices = np.random.choice(group, num_clients, replace=False, p=pk)
            return indices, np.asarray(self.clients)[indices]

        elif self.sampling == 2:
            num_samples = []
            for client in group_c:
                num_samples.append(client.train_samples)
            total_samples = np.sum(np.asarray(num_samples))
            pk = [item * 1.0 / total_samples for item in num_samples]
            indices = np.random.choice(group, num_clients, replace=False, p=pk)
            return indices, np.asarray(self.clients)[indices]

    def hierarchical_clustering(self, csolns, indices):
        X = np.array([])
        for k in range(len(csolns[0])):
            X = np.concatenate((X, csolns[0][k].flatten()), axis=0)
        X = X.reshape(1, -1)
        for i in range(1, len(csolns)):
            v = np.array([])
            for k in range(len(csolns[0])):
                v = np.concatenate((v, csolns[i][k].flatten()), axis=0)
            v= v.reshape(1, -1)
            X = np.r_[X, v] #this reshape is invalid

        Z = linkage(X, method=self.linkage, metric=self.distance)

        dendrogram(Z, truncate_mode='lastp', p=120, show_leaf_counts=True, leaf_rotation=90, leaf_font_size=10, show_contracted=True)
        k = self.num_of_clusters
        labels = fcluster(Z, t=k, criterion='maxclust')
        G = []
        for i in range(1, k+1):
            tmp = []
            for j in range(len(csolns)):
                if labels[j] == i:
                    tmp.append(indices[j])
            G.append(tmp)
        return G


    def simple_average(self, parameters):

        base = [0] * len(parameters[0])

        for p in parameters:  # for each client
            for i, v in enumerate(p):
                base[i] += v.astype(np.float64)   # the i-th layer

        averaged_params = [v / len(parameters) for v in base]

        return averaged_params

    def median_average(self, parameters):

        num_layers = len(parameters[0])
        aggregated_models = []
        for i in range(num_layers):
            a = []
            for j in range(len(parameters)):
                a.append(parameters[j][i].flatten())
            aggregated_models.append(np.reshape(np.median(a, axis=0), newshape=parameters[0][i].shape))

        return aggregated_models

    def krum_average(self, k, parameters):
        # krum: return the parameter which has the lowest score defined as the sum of distance to its closest k vectors
        flattened_grads = []
        for i in range(len(parameters)):
            flattened_grads.append(process_grad(parameters[i]))
        distance = np.zeros((len(parameters), len(parameters)))
        for i in range(len(parameters)):
            for j in range(i+1, len(parameters)):
                distance[i][j] = np.sum(np.square(flattened_grads[i] - flattened_grads[j]))
                distance[j][i] = distance[i][j]

        score = np.zeros(len(parameters))
        for i in range(len(parameters)):
            score[i] = np.sum(np.sort(distance[i])[:k+1])  # the distance including itself, so k+1 not k

        selected_idx = np.argsort(score)[0]

        return parameters[selected_idx]

    def mkrum_average(self, k, m, parameters):
        flattened_grads = []
        for i in range(len(parameters)):
            flattened_grads.append(process_grad(parameters[i]))
        distance = np.zeros((len(parameters), len(parameters)))
        for i in range(len(parameters)):
            for j in range(i + 1, len(parameters)):
                distance[i][j] = np.sum(np.square(flattened_grads[i] - flattened_grads[j]))
                distance[j][i] = distance[i][j]

        score = np.zeros(len(parameters))
        for i in range(len(parameters)):
            score[i] = np.sum(np.sort(distance[i])[:k + 1])  # the distance including itself, so k+1 not k

        # multi-krum selects top-m 'good' vectors (defined by socre) (m=1: reduce to krum)
        selected_idx = np.argsort(score)[:m]
        selected_parameters = []
        for i in selected_idx:
            selected_parameters.append(parameters[i])

        return self.simple_average(selected_parameters)


    def k_norm_average(self, num_benign, parameters):
        flattened_grads = []
        for i in range(len(parameters)):
            flattened_grads.append(process_grad(parameters[i]))
        norms = [np.linalg.norm(u) for u in flattened_grads]
        selected_idx = np.argsort(norms)[:num_benign]  # filter out the updates with large norms
        selected_parameters = []
        for i in selected_idx:
            selected_parameters.append(parameters[i])
        return self.simple_average(selected_parameters)
