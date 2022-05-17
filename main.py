import argparse
import importlib
import time
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from modules.solver.utils import read_data

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# GLOBAL PARAMETERS
OPTIMIZERS = ['FedCHAR']
DATASETS = ['HARBox', 'IMU', 'Depth', 'UWB', 'FMCW']


MODEL_PARAMS = {

    'HARBox': (5, ),
    'IMU': (3, ),
    'Depth': (5, ),
    'UWB': (2, ),
    'FMCW': (6, )
}


def read_options():
    ''' Parse command line arguments or load defaults '''
    parser = argparse.ArgumentParser()

    parser.add_argument('--optimizer',
                        help='name of optimizer',
                        type=str,
                        choices=OPTIMIZERS,
                        default='FedCHAR')
    parser.add_argument('--dataset',
                        help='name of dataset',
                        type=str,
                        choices=DATASETS,
                        default='IMU')
    parser.add_argument('--model',
                        help='name of model',
                        type=str,
                        default='IMU.py')
    parser.add_argument('--remain_rounds',
                        help='number of rounds after clustering',
                        type=int,
                        default=50)
    parser.add_argument('--eval_every',
                        help='evaluate every ____ communication rounds',
                        type=int,
                        default=1)
    parser.add_argument('--clients_per_round',
                        help='number of clients trained per communication round',
                        type=int,
                        default=7)
    parser.add_argument('--batch_size',
                        help='batch size of local training',
                        type=int,
                        default=5)
    parser.add_argument('--learning_rate',
                        help='learning rate',
                        type=float,
                        default=0.01)
    parser.add_argument('--seed',
                        help='seed for random initialization',
                        type=int,
                        default=0)
    parser.add_argument('--sampling',
                        help='client sampling methods',
                        type=int,
                        default='2')
    parser.add_argument('--num_corrupted',
                        help='how many corrupted devices',
                        type=int,
                        default=0)
    parser.add_argument('--label_flipping',
                        help='whether do A1',
                        type=int,
                        default=0)
    parser.add_argument('--random_updates',
                        help='whether do A2',
                        type=int,
                        default=0)
    parser.add_argument('--boosting',
                        help='whether do A3',
                        type=int,
                        default=0)
    parser.add_argument('--inner_product',
                        help='whether do A4',
                        type=int,
                        default=0)
    parser.add_argument('--gradient_clipping',
                        type=int,
                        default=0)
    parser.add_argument('--median',
                        type=int,
                        default=0)
    parser.add_argument('--krum',
                        type=int,
                        default=0)
    parser.add_argument('--multi_krum',
                        type=int,
                        default=0)
    parser.add_argument('--k_norm',
                        type=int,
                        default=0)
    parser.add_argument('--epoch',
                        type=int,
                        default=0)
    parser.add_argument('--lam',
                        help='lambda in the objective',
                        type=float,
                        default=0)
    parser.add_argument('--initial_rounds',
                        type=int,
                        default=10)
    parser.add_argument('--linkage',
                        type=str,
                        default='complete')
    parser.add_argument('--distance',
                        type=str,
                        default='cosine')
    parser.add_argument('--corrupt_seed',
                        help='seed for random select corrupt id',
                        type=int,
                        default=0)
    parser.add_argument('--num_of_clusters',
                        type=int,
                        default=0)
    try:
        parsed = vars(parser.parse_args())
    except IOError as msg:
        parser.error(str(msg))

    # load selected model

    model_path = '%s.%s.%s' % ('modules', 'models', parsed['model'])

    mod = importlib.import_module(model_path)
    learner = getattr(mod, 'Model')


    opt_path = 'modules.solver.%s' % parsed['optimizer']


    mod = importlib.import_module(opt_path)
    optimizer = getattr(mod, 'Server')

    # add selected model parameter
    parsed['model_params'] = MODEL_PARAMS['.'.join(model_path.split('.')[2:])]


    # print and return
    maxLen = max([len(ii) for ii in parsed.keys()])
    fmtString = '\t%' + str(maxLen) + 's : %s'
    print('Arguments:')
    for keyPair in sorted(parsed.items()): print(fmtString % keyPair)

    return parsed, learner, optimizer

def main():
    # suppress tf warnings
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)
    
    # parse command line arguments
    options, learner, optimizer = read_options()

    # read data

    train_path = os.path.join('data', options['dataset'], 'data', 'train')
    test_path = os.path.join('data', options['dataset'], 'data', 'test')
    dataset = read_data(train_path, test_path)


    t = optimizer(options, learner, dataset)
    t.train()

    
if __name__ == '__main__':
    main()





