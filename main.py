import time
import argparse
import pickle
from model import *
from utils import *
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'


def init_seed(seed=None):
    if seed is None:
        seed = int(time.time() * 1000 // 1000)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='Tmall', help='Tmall/retailrocket/lastfm')
parser.add_argument('--hiddenSize', type=int, default=100)
parser.add_argument('--epoch', type=int, default=30)
parser.add_argument('--activate', type=str, default='relu')
parser.add_argument('--n_sample_all', type=int, default=12)
parser.add_argument('--n_sample', type=int, default=12)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.001, help='learning rate.')
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay.')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay.')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty ')
parser.add_argument('--n_iter', type=int, default=1)                                    # [1, 2]
parser.add_argument('--dropout_gcn', type=float, default=0, help='Dropout rate.')       # [0, 0.2, 0.4, 0.6, 0.8]
parser.add_argument('--dropout_local', type=float, default=0, help='Dropout rate.')     # [0, 0.5]
parser.add_argument('--dropout_global', type=float, default=0.5, help='Dropout rate.')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=3)
parser.add_argument('--interests', type=int, default=3, help='The number of interests')  # [1, 2, 3, 4, 5]
parser.add_argument('--beta', type=float, default=0.01, help='Beta for the interests regularization')   # [0, 0.001, 0.005, 0.01, 0.02, 0.05]
parser.add_argument('--length', type=float, default=8, help='eta in the paper')          # [8, 10, 12, 14, 16, 18]

opt = parser.parse_args()


def main():
    init_seed(2024)

    if opt.dataset == 'Tmall':
        num_node = 40728
        opt.n_iter = 1
        opt.dropout_global = 1
        opt.dropout_gcn = 1
        opt.dropout_local = 0.7
        opt.beta = 0.02
        opt.interests = 5
        opt.length = 8
    elif opt.dataset == 'retailrocket':
        num_node = 36969
        opt.n_iter = 1
        opt.dropout_global = 0.5
        opt.dropout_gcn = 0.8
        opt.dropout_local = 0.0
        opt.beta = 0.005
        opt.interests = 3
        opt.length = 12
    elif opt.dataset == 'lastfm':
        num_node = 38616
        opt.n_iter = 1
        opt.dropout_global = 0.1
        opt.dropout_gcn = 0
        opt.dropout_local = 0
        opt.length = 18
        opt.beta = 0.005
        opt.interests = 5
    else:
        raise Exception('Unknown Dataset!')

    train_data = pickle.load(open('datasets/' + opt.dataset + '/train.txt', 'rb'))
    if opt.validation:
        train_data, valid_data = split_validation(train_data, opt.valid_portion)
        test_data = valid_data
    else:
        test_data = pickle.load(open('datasets/' + opt.dataset + '/test.txt', 'rb'))

    adj = pickle.load(open('datasets/' + opt.dataset + '/adj_' + str(opt.n_sample_all) + '.pkl', 'rb'))
    num = pickle.load(open('datasets/' + opt.dataset + '/num_' + str(opt.n_sample_all) + '.pkl', 'rb'))
    train_data = Data(train_data)
    test_data = Data(test_data)

    adj, num = handle_adj(adj, num_node, opt.n_sample_all, num)
    model = trans_to_cuda(DMIGNN(opt, num_node, adj, num))

    print(opt)
    start = time.time()
    best_result = [0, 0, 0]
    best_epoch = [0, 0, 0]
    bad_counter = 0

    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        hit, mrr, cov = train_test(model, train_data, test_data)
        cov = cov * 100 / (num_node - 1)
        flag = 0
        if hit >= best_result[0]:
            best_result[0] = hit
            best_epoch[0] = epoch
            flag = 1
        if mrr >= best_result[1]:
            best_result[1] = mrr
            best_epoch[1] = epoch
            flag = 1
        if cov >= best_result[2]:
            best_result[2] = cov
            best_epoch[2] = epoch
            flag = 1
        print('Current Result:')
        print('\tRecall@20:\t%.4f\tMRR@20:\t%.4f\tCov@20:\t%.4f' % (hit, mrr, cov))
        print('Best Result:')
        print('\tRecall@20:\t%.4f\tMRR@20:\t%.4f\tCov@20:%.4f\t\tEpoch:\t%d,\t%d,\t%d' % (
            best_result[0], best_result[1], best_result[2], best_epoch[0], best_epoch[1], best_epoch[2]))
        bad_counter += 1 - flag
        if bad_counter >= opt.patience:
            break
    print('-------------------------------------------------------')
    end = time.time()
    print("Run time: %f s" % (end - start))


if __name__ == '__main__':
    main()
