
class Client(object):
    
    def __init__(self, id, group=None, train_data={'x':[],'y':[]}, test_data={'x':[],'y':[]}, model=None):
        self.model = model
        self.id = id # integer
        self.group = group

        train_len = len(train_data['x'])
        Operation = 'result'

        # If you just want to see the results in our paper
        if Operation == 'result':
            self.train_data = train_data
            self.val_data = test_data

        #If you want to perform parameter tuning
        else:
            self.train_data = {'x': train_data['x'][:int(train_len * 0.9)],
                                'y': train_data['y'][:int(train_len * 0.9)]}
            self.val_data = {'x': train_data['x'][int(train_len * 0.9):],
                            'y': train_data['y'][int(train_len * 0.9):]}

        self.test_data = test_data
        self.train_samples = len(self.train_data['y'])
        self.val_samples = len(self.val_data['y'])
        self.test_samples = len(self.test_data['y'])

    def set_params(self, model_params):
        '''set model parameters'''
        self.model.set_params(model_params)

    def get_params(self):
        '''get model parameters'''
        return self.model.get_params()


    def get_loss(self):
        return self.model.get_loss(self.train_data)

    def get_val_loss(self):
        return self.model.get_loss(self.val_data)


    def solve_sgd(self, mini_batch_data):
        '''
        run one iteration of mini-batch SGD
        '''
        grads, loss, weights = self.model.solve_sgd(mini_batch_data)
        return (self.train_samples, weights), (self.train_samples, grads), loss

    def train_error(self):

        tot_correct, loss, _ = self.model.test(self.train_data)
        return tot_correct, loss, self.train_samples


    def test(self):

        tot_correct, loss, cm = self.model.test(self.test_data)
        return tot_correct, loss, self.test_samples, cm

    def validate(self):
        tot_correct, loss, cm = self.model.test(self.val_data)
        return tot_correct, loss, self.val_samples, cm