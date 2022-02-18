import torch
import torch.nn as nn

import argparse

def str2bool(v):
    """
    Source code copied from: https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def adjust_learning_rate(optimizer, lr, factor, min_lr):
        """Sets the learning rate to the given LR decayed by factor every every_epochs epochs
        """
        if min_lr is None:
            new_lr = lr * factor
        else:
            new_lr = max(lr * factor, min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        return new_lr


def build_fc_network(layer_dims, activation="relu", dropout_prob=0.):
    """
    Stacks multiple fully-connected layers with an activation function and a dropout layer in between.

    - Source used as orientation: https://github.com/eelxpeng/UnsupervisedDeepLearning-Pytorch/blob/master/udlp/clustering/vade.py

    Args:
        layer_dims: A list of integers, where (starting from 1) the (i-1)th and ith entry indicates the input
                    and output dimension of the ith layer, respectively.
        activation: Activation function to choose.
        dropout_prob: Dropout probability between every fully connected layer with activation.

    Returns:
        An nn.Sequential object of the layers.
    """
    # Note: possible alternative: OrderedDictionary
    net = []
    for i in range(1, len(layer_dims)):
        net.append(nn.Linear(layer_dims[i-1], layer_dims[i]))
        if activation == "relu":
            net.append(nn.ReLU())
        elif activation == "sigmoid":
            net.append(nn.Sigmoid())
        elif activation == "elu":
            net.append(nn.ELU())
        elif activation == "leakyrelu":
            net.append(nn.LeakyReLU())
        elif activation == "tanh":
            net.append(nn.Tanh())
        net.append(nn.Dropout(dropout_prob))
    net = nn.Sequential(*net)  # unpacks list as separate arguments to be passed to function

    return net