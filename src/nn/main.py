import argparse
import numpy as np
from data_loader import load_data
from train import train, plot_metrics

np.random.seed(555)


parser = argparse.ArgumentParser()
parser.add_argument('--val_ratio', type = float, default = 0.2, help = 'validation size')
parser.add_argument('--test_ratio', type = float, default = 0.2, help = 'test size')
parser.add_argument('--embed_dim', type = int, default = 2, help = 'dimension of embedding layer')
parser.add_argument('--hidden_dim', type = int, default = 4, help = 'dimension of hidden layer')
parser.add_argument('--n_epochs', type = int, default = 20, help = 'number of epochs')
parser.add_argument('--batch_size', type = int, default = 4096, help = 'batch size')
parser.add_argument('--l2_weight', type = float, default = 1e-8, help = 'weight of l2 regularization')
parser.add_argument('--learning_rate', type = float, default = 0.01, help = 'learning rate')

args = parser.parse_args()


def main():
    data_info = load_data(args)
    rmse_info = train(args, data_info)
    plot_metrics(rmse_info)
    
    
    

if __name__ == '__main__':
    main()