
import argparse
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import torch.distributions as D


from torch.autograd import Variable

import numpy as np
import os
import matplotlib.pyplot as plt
from pytorch_lightning import Trainer, seed_everything
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

import pickle

torch.backends.cudnn.benchmark = True  # for potential speedup, see https://www.reddit.com/r/MachineLearning/comments/kvs1ex/d_here_are_17_ways_of_making_pytorch_training/ (6.)

from utils import str2bool, adjust_learning_rate
from datasets import News, IHDP, ACIC2016
from models import SimpleMatchNet, SimpleNet
from matching import matching_metrics
from datetime import datetime
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import json
import pandas as pd

# avoid weird error
torch.set_num_threads(1)

def train_args_parser():

    parser = argparse.ArgumentParser(description='Neural score matching training')

    # general
    parser.add_argument('--device', type=str, default="cuda",   # e.g. "cuda", "cpu", ...
                        help='device to use for all heavy tensor operations')
    parser.add_argument('--wandb_mode', type=str, default="online", 
                        help="mode of wandb run tracking, either no tracking ('disabled') or with tracking ('online')")
    parser.add_argument('--wandb_dir', type=str, default="../outputs", 
                        help="directory where the wandb folder is created, default same as code.")

    parser.add_argument('--wandb_config', type=str, default='wandb_config.json', 
                        help="wandb config file")
    parser.add_argument('--model_type', type=str, default='simple_match_net', 
                        help="model type to use, in [simple_match_net, simple_net]")
    parser.add_argument('--dataset', type=str, default='acic2016', 
                        help="dataset used during training, one in ['news', 'acic2016', 'ihdp']")
    parser.add_argument('--data_folder', type=str, default='../data', 
                        help="data folder, default is assumed to be ../data/")

    # TODO can this be done in a nicer way?
    parser.add_argument('--dataset_on_gpu', type=str2bool, nargs='?', dest='dataset_on_gpu', const=True, default=True,
                        # special parsing of boolean argument
                        help='whether the dataset is on GPU or not (batches are loaded separately to GPU)')
    parser.add_argument('--save_model', type=str2bool, nargs='?', dest='save_model', const=True, default=True,
                        help='whether to save the model or not')
    parser.add_argument('--load_model', type=str, default=None)
    parser.add_argument('--config_args_path', type=str, default="", 
                        help="the path to the args config dictionary to be loaded. If a path is provided, all specifications of hyperparameters are ignored. \
                             If the argument is an empty string, the hyperparameter specifications above are used as usual.")
    parser.add_argument('--do_val_during_training', type=str2bool, nargs='?', dest='do_val_during_training', const=True,
                        default=True,  # special parsing of boolean argument
                        help='whether to perform evaluation on the test set throughout training, if True, do it for every epoch, otherwise only do it at the end of the training')
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1, 
                        help='every how many epochs to perform a validation')

    parser.add_argument('--balancing_dim', type=int, default=10, 
                        help='dimension of the intermediate balancing score')
    parser.add_argument('--dropout_prob', type=float, default=0., 
                        help='dropout probability')
    parser.add_argument('--activation', type=str, default="elu", 
                        help='activation function to use - see PyTorch documentation for name')
    # TODO : refactor the model chocie
    
    # Simple Match Net
    parser.add_argument('--n_hidden', type=int, default=200,
                        help="number of hidden units in each layer.")
    parser.add_argument('--n_layers', type=int, default=0, help="number of layers.")
    parser.add_argument('--layer_dims', type=int, nargs='+', default=None,
                        help="list of integers, each indicating the output dimensions of each hidden layer.")
    parser.add_argument('--alpha_focal', type=float, default=0, help="alpha of the focal loss, 0 is equivalent to cross-entropy.")
    parser.add_argument('--eval_matching_during_training', type=str2bool, nargs='?', dest='eval_matching_during_training', const=True, default=False, help='evaluate matching metrics during training')
    parser.add_argument('--optimizer', type=str, default="adam", help="optimizer used for training")

    # learning rate configs
    parser.add_argument('--init_lr', type=float, default=0.001,
                        help='initial learning rate for training')
    parser.add_argument('--lr_decay', type=float, default=0.95,
                        help='decay factor of the learning rate every args.update_lr_every_epoch')
    parser.add_argument('--update_lr_every_epoch', type=int, default=10,
                        help='number of epochs after which to update learning rate')
    parser.add_argument('--min_lr', type=float, default=None, # 0.0002
                        help='the minimum learning rate (lower bound of decaying). If None, no minimum applied.')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='weight decay.')

    # batch size configs
    parser.add_argument('--batch_size', type=int, default=100,
                        help='input batch size for training')
    parser.add_argument('--eval_batch_size', type=int, default=100,
                        help='input batch size for evaluation')
    parser.add_argument('--n_test_batches', type=int, default=-1,
                        help='number of test batches to use per evaluation (-1 uses all available batches)')

    parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs to train')
    parser.add_argument('--seed', type=int, default=1000,
                        help='seed for pseudo-randomness')


    parser.add_argument('--dummy', type=int, default=0, help="dummy variable")

    # Common to all datasets
    parser.add_argument('--train_size', type=float, default=0.6, help='train size, between 0 and 1 exclusive')
    parser.add_argument('--val_size', type=float, default=0.2, help='train size, between 0 and 1 exclusive')
    parser.add_argument('--scale_covariates', type=str2bool, nargs='?', dest='scale_covariates', const=True, default=True,
                        help='scale covariates')

    # iHDP configs
    parser.add_argument('--ihdp_exp_num', type=int, default=1, 
                        help='choose an experiment configuration of IHDP between 1 and 50')
    # ACIC 2016 configs
    parser.add_argument('--acic2016_setting', type=int, default=4, 
                        help='choose a setting for ACIC2016 between 1 and 77')
    parser.add_argument('--acic2016_exp_num', type=int, default=1, 
                        help='choose an experiment configuration of ACIC2016 between 1 and 100')

    # News configs
    parser.add_argument('--news_exp_num', type=int, default=1, 
                        help='choose an experiment configuration of News between 1 and 50')
    parser.add_argument('--news_seed', type=int, default=0, 
                        help='choose a seed for train/val/test split for News between 1 and 50')


    # Early stopping
    parser.add_argument('--min_delta', type=float, default=0.00, help='min delta of early stopping')
    parser.add_argument('--patience', type=int, default=0, help='patience of early stopping')

    # matching config
    parser.add_argument('--n_neighbors', type=int, default=1, help='number of neighbors ')

    # sweep config
    parser.add_argument('--sweep_config', type=str, default=None, help='file with sweep config')
    parser.add_argument('--sweep_id', type=str, default=None, help='id of sweep')
    parser.add_argument('--sweep_id_load', type=str, default=None, help='id of sweep containing pre-trained models to load')
    parser.add_argument('--filter_on', type=str, nargs='+', default=None, help='find run in sweep based on given filter')

    # retrieve table
    parser.add_argument('--table_name', type=str, default='results', help='name of the file where results are saved (in retrieve_sweep_table.py)')

    args = parser.parse_args()

    return args

# In a given sweep, find a given run and extract its trained model parameters
def get_model_dict(sweep_id_load, args, wandb, filter_on):
    with open(args.wandb_config) as f:
        wandb_config = json.load(f)
    filters = {}
    print(filter_on)
    for on in filter_on:
        filters['config.'+on] = getattr(args, on)
    filters['sweep'] = sweep_id_load
    print(filters)
    runs = wandb.Api().runs(path=f"{wandb_config['entity']}/{wandb_config['project']}", filters=filters)
    print(len(runs))
    runs[0].file("save_dict.pt").download(replace=True)
    model_dict = torch.load('save_dict.pt')['state_dict']
    return model_dict

# Main training function
def train(args = None):

    if args is None:
        args = train_args_parser()

    import wandb

    # set the right user to login
    with open(args.wandb_config) as f:
        wandb_config = json.load(f)

    wandb.login(key=wandb_config['key'])

    # create W&B logger from with pl support

    # keeping the wandb.init call in, so that everything else keeps working!
    wandb_run = wandb.init(project=wandb_config['project'], entity=wandb_config['entity'], mode=args.wandb_mode, dir=args.wandb_dir)
    wandb_logger = WandbLogger(project=wandb_config['project'], entity=wandb_config['entity'], mode=args.wandb_mode, save_dir=wandb.run.dir)

    # for sweep: don't use such values of args above which are defined by sweep
    # set args value to sweep values instead
    for (key, value) in wandb.config.items():
        setattr(args, key, value)  # args is namespace object
    setattr(args, 'n_dims_tot', None)

    # update configs -> remember hyperparams
    wandb.config.update(args)

    # load config dictionary instead
    if args.config_args_path != "":
        with open(args.config_args_path, 'rb') as file:
            print("NOTE: Loading args configuration dictionary which overrides any specified hyperparameters!")
            args = pickle.load(file)

    # print out args and wandb run dir
    print(args)
    print("wandb.run.dir: ", wandb.run.dir)


    # make device a global variable so that dataset.py can access it
    global device
    # initializing global variable (see above)
    device = torch.device(args.device)

    # Choose right dataset and create train/val/test dataset objects as well as their loaders
    if args.dataset == 'news':
        print(args.data_folder)
        train_data = News(args.news_exp_num, dataset="train", tensor=True, device=args.device, train_size=args.train_size, val_size=args.val_size, data_folder=args.data_folder, seed=args.news_seed)
        val_data = News(args.news_exp_num, dataset="val", tensor=True, device=args.device, train_size=args.train_size, val_size=args.val_size, data_folder=args.data_folder, seed=args.news_seed)
        train_val_data = News(args.news_exp_num, dataset="train_val", tensor=True, device=args.device, train_size=args.train_size, val_size=args.val_size, data_folder=args.data_folder, seed=args.news_seed)
        test_data = News(args.news_exp_num, dataset="test", tensor=True, device=args.device, train_size=args.train_size,val_size=args.val_size, data_folder=args.data_folder, seed=args.news_seed)

        # initialize data loaders
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=4 if args.device == "cpu" else 0)
        train_val_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=4 if args.device == "cpu" else 0)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=4 if args.device == "cpu" else 0)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.eval_batch_size, shuffle=False,
                                                  num_workers=4 if args.device == "cpu" else 0)

    elif args.dataset == 'ihdp':
        print(args.data_folder)
        train_data = IHDP(args.ihdp_exp_num, device=args.device, dataset='train', path_data=args.data_folder, tensor=True, normalise_y=(args.model_type == 'cevae'), train_size=args.train_size, val_size=args.val_size)
        val_data = IHDP(args.ihdp_exp_num, device=args.device, dataset='val', path_data=args.data_folder, tensor=True, normalise_y=(args.model_type == 'cevae'), train_size=args.train_size, val_size=args.val_size)
        train_val_data = IHDP(args.ihdp_exp_num, device=args.device, dataset='train_val', path_data=args.data_folder, tensor=True, normalise_y=(args.model_type == 'cevae'), train_size=args.train_size, val_size=args.val_size)
        test_data = IHDP(args.ihdp_exp_num, device=args.device, dataset='test', path_data=args.data_folder, tensor=True, normalise_y=(args.model_type == 'cevae'), train_size=args.train_size, val_size=args.val_size)

        # initialize data loaders
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=4 if args.device == "cpu" else 0)
        train_val_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                                       num_workers=4 if args.device == "cpu" else 0)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=4 if args.device == "cpu" else 0)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.eval_batch_size, shuffle=False,
                                                  num_workers=4 if args.device == "cpu" else 0)

    elif args.dataset == 'acic2016':
        print(args.data_folder)
        train_data = ACIC2016(setting=args.acic2016_setting, exp_num=args.acic2016_exp_num, device=args.device, dataset='train', path_data=args.data_folder, tensor=True, scale=args.scale_covariates)
        val_data = ACIC2016(setting=args.acic2016_setting, exp_num=args.acic2016_exp_num, device=args.device, dataset='val', path_data=args.data_folder, tensor=True, scale=args.scale_covariates)
        train_val_data = ACIC2016(setting=args.acic2016_setting, exp_num=args.acic2016_exp_num, device=args.device, dataset='train_val', path_data=args.data_folder, tensor=True, scale=args.scale_covariates)
        test_data = ACIC2016(setting=args.acic2016_setting, exp_num=args.acic2016_exp_num, device=args.device, dataset='test', path_data=args.data_folder, tensor=True, scale=args.scale_covariates)

        # initialize data loaders
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=4 if args.device == "cpu" else 0)
        train_val_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                                       num_workers=4 if args.device == "cpu" else 0)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=4 if args.device == "cpu" else 0)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.eval_batch_size, shuffle=False,
                                                  num_workers=4 if args.device == "cpu" else 0)
    else:
        raise ValueError("args.dataset has not chosen an implemented dataset")

    # dataset-specific parameters
    in_dim = train_data.x.shape[1]
    
    # Create appropriate model with appropriate hyperparameters
    if args.model_type == 'simple_match_net':
        model_init_dict = dict(in_dim=in_dim, balancing_dim=args.balancing_dim,
                               layer_dims=args.layer_dims if args.layer_dims is not None else [args.n_hidden] * args.n_layers,
                               dropout_prob=args.dropout_prob, activation=args.activation, weight_decay=args.weight_decay,
                               dataset=args.dataset, init_lr=args.init_lr, alpha_focal=args.alpha_focal,
                               optimizer=args.optimizer, eval_matching_during_training=args.eval_matching_during_training)
        wandb.log({'model_init_dict': model_init_dict})
        print('model_init_dict:')
        print(model_init_dict)
        model = SimpleMatchNet(**model_init_dict)
    elif args.model_type == 'simple_net':
        model_init_dict = dict(in_dim=in_dim,
                               layer_dims=args.layer_dims if args.layer_dims is not None else [args.n_hidden] * args.n_layers,
                               dropout_prob=args.dropout_prob, activation=args.activation, weight_decay=args.weight_decay,
                               dataset=args.dataset, init_lr=args.init_lr, alpha_focal=args.alpha_focal,
                               optimizer=args.optimizer, eval_matching_during_training=args.eval_matching_during_training)
        wandb.log({'model_init_dict': model_init_dict})
        print('model_init_dict:')
        print(model_init_dict)
        model = SimpleNet(**model_init_dict)
    else:
        raise ValueError("No recognised model given")

    # weights&biases tracking (gradients, network topology)
    wandb.watch(model)

    # sets seeds for numpy, torch, python.random and PYTHONHASHSEED.
    seed_everything(args.seed)

    # Train model
    print("Run trainer...")
    now = datetime.now()
    gpus = -1 if args.device == "cuda" else None
    if args.sweep_id_load is None:
        early_stop_callback = EarlyStopping(
            monitor='val_metric/loss',
            min_delta=args.min_delta,
            patience=args.patience,
            verbose=False,
            mode='min'
        )
        trainer = pl.Trainer(deterministic=True, logger=wandb_logger, check_val_every_n_epoch=args.check_val_every_n_epoch, max_epochs=args.n_epochs, gpus=gpus, callbacks=[early_stop_callback])
        # TODO test in the very beginning of training and in the very end!
        trainer.validate(model, train_loader)
        if args.do_val_during_training:
            trainer.validate(model, val_loader)
            trainer.fit(model, train_loader, val_loader)
        else:
            trainer.fit(model, train_val_loader)
        later = datetime.now()
        diff = later - now
        print("Training took {}".format(diff.seconds))

    else:
        model_dict = get_model_dict(sweep_id_load=args.sweep_id_load, args=args, wandb=wandb, filter_on=args.filter_on)
        model.load_state_dict(model_dict)


    # save model
    if args.save_model:
        save_dict_path = os.path.join(wandb.run.dir, "save_dict.pt")
        save_dict = {'state_dict': model.state_dict(),
                     "model_init_dict": model_init_dict,
                     'args': args}  # args dictionary is already part of saving the model
        torch.save(save_dict, save_dict_path)

    # Computing matching metrics
    now = datetime.now()
    if args.dataset in ['news','ihdp','acic2016'] and args.model_type == 'simple_match_net':
        if args.device == 'cuda':
            model = model.cuda()
        print("IN SAMPLE MATCHING METRICS")
        train_val_batch = train_val_data[:]
        matching_results, loss_results, scores = model.matching_metrics(train_val_batch, n_neighbors=args.n_neighbors)
        for metric in matching_results:
            wandb.log({'insample_matching_metric/' + metric: matching_results[metric]})
        for metric in loss_results:
            wandb.log({'insample_loss_metric/' + metric: loss_results[metric]})
        save_args_path = os.path.join(wandb.run.dir, "scores_insample.pickle")
        with open(save_args_path, 'wb') as file:
            pickle.dump(scores, file, protocol=pickle.HIGHEST_PROTOCOL)
        print("HOLD OUT MATCHING METRICS")
        test_batch = test_data[:]
        matching_results, loss_results, scores = model.matching_metrics(test_batch, n_neighbors=args.n_neighbors)
        for metric in matching_results:
            wandb.log({'holdout_matching_metric/' + metric: matching_results[metric]})
        for metric in loss_results:
            wandb.log({'holdout_loss_metric/' + metric: loss_results[metric]})
        save_args_path = os.path.join(wandb.run.dir, "scores_holdout.pickle")
        with open(save_args_path, 'wb') as file:
            pickle.dump(scores, file, protocol=pickle.HIGHEST_PROTOCOL)


    later = datetime.now()
    diff = later - now
    print("Matching / computing ATT took {} seconds".format(diff.seconds))



    # save args config dictionary
    save_args_path = os.path.join(wandb.run.dir, "args.pickle")  # args_dict.py
    with open(save_args_path, 'wb') as file:
        pickle.dump(args, file, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    train()
