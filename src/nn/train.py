import tensorflow as tf
import random
import numpy as np
import matplotlib.pyplot as plt
from model import NeuralNetwork


def train(args,data_info):
    n_user = data_info[0]
    n_item = data_info[1]
    train_data = data_info[2]
    val_data = data_info[3]
    test_data = data_info[4]
    
    model = NeuralNetwork(args,n_user,n_item)
    
    train_rmse_ls = []
    val_rmse_ls = []
    test_rmse_ls = []
    epoch = []
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for step in range(args.n_epochs):
            np.random.shuffle(train_data)
            start = random.randint(0, train_data.shape[0] - args.batch_size)
            _ = model.train(sess, feed_dict = get_feed_dict(train_data, model, start, start + args.batch_size))

            #start = 0
            #while start < len(train_data):
            #    model.train(sess, feed_dict=get_feed_dict(train_data, model, start, start + args.batch_size))
            #    start += args.batch_size
            
            train_loss, train_rmse = model.validation(sess, feed_dict=get_feed_dict(train_data, model, 0, len(train_data)))
            val_loss, val_rmse = model.validation(sess, feed_dict=get_feed_dict(val_data, model, 0, len(val_data)))
            test_loss, test_rmse = model.validation(sess, feed_dict=get_feed_dict(test_data, model, 0, len(test_data)))
            
            train_rmse_ls.append(train_rmse)
            val_rmse_ls.append(val_rmse)
            test_rmse_ls.append(test_rmse)
            epoch.append(step)
            
            
            
            print('epoch %d    train loss: %.4f  rmse: %.4f    val loss: %.4f  rmse: %.4f    test loss: %.4f  rmse: %.4f'
                  % (step, train_loss, train_rmse, val_loss, val_rmse, test_loss, test_rmse))
                
        return train_rmse_ls, val_rmse_ls, test_rmse_ls, epoch



def get_feed_dict(data, model, start, end):
    feed_dict = dict()
    feed_dict[model.user_indices] = data[start:end, 0]
    feed_dict[model.item_indices] = data[start:end, 1]
    feed_dict[model.ratings] = data[start:end, 2]
    
    return feed_dict


def plot_metrics(rmse_info):
    train_rmse_ls = rmse_info[0]
    val_rmse_ls = rmse_info[1]
    test_rmse_ls = rmse_info[2]
    epoch = rmse_info[3]
    plt.plot(epoch, train_rmse_ls, label = 'train', color = 'red')
    plt.plot(epoch, val_rmse_ls, label = 'valid', color = 'green')
    plt.plot(epoch, test_rmse_ls, label = 'test', color = 'blue')
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("User RMSE")
    plt.show()
