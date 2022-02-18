 
import torch
import torch.distributions as D
import torch.nn as nn
import torch.optim as optim
import numpy as np
import abc
import pytorch_lightning as pl
from torch.nn import BCELoss

from matching import Matcher


from modules import BernoulliFC, GaussianFC, GaussianTARNet, LikelihoodFC, PropensityScoreFC, DISTRIBUTIONS, LikelihoodTARNet
from utils import build_fc_network


# Just a propensity score model where the first layer is allowed to take another dimension (balancing_dim) and it will serve as the balancing score
class SimpleMatchNet(pl.LightningModule):

    PROGRAMME_DEFAULT = [
        ('compute_distances', dict(on='propensity')),
        ('compute_distances', dict(metric='euclidean', on=['balancing']))
    ]

    METRICS_DEFAULT = ['att', 'error_att', 'linear_mmd_att']

    AGGREG_FUNCS = {
        'median': np.median,
        'min': np.min,
        'max': np.max
    }

    OPTIMIZERS = {
            "adam": optim.Adam,
            "sgd": optim.SGD,
        }

    def __init__(self, in_dim, balancing_dim, layer_dims, dataset, init_lr, activation='relu', dropout_prob=0., weight_decay=0., alpha_focal=0, optimizer="adam", eval_matching_during_training=True):

        super(SimpleMatchNet, self).__init__()
        self.optimizer = self.OPTIMIZERS[optimizer]
        self.balancing_dim = balancing_dim
        self.layer_dims = layer_dims
        self.dataset = dataset
        self.init_lr = init_lr
        self.weight_decay = weight_decay
        self.alpha_focal = alpha_focal
        self.eval_matching_during_training = eval_matching_during_training

        self.ps_net = PropensityScoreFC(in_dim=in_dim, out_dim=1, layer_dims= [balancing_dim] + layer_dims,
                                        dropout_prob=dropout_prob, activation=activation).to(self.device)

    def forward(self, x):

        ps = self.ps_net(x)

        return ps

    def compute_loss(self, ps, t, numpy=False):
        if not torch.is_tensor(ps):
            ps = torch.Tensor(ps)
        if not torch.is_tensor(t):
            t = torch.Tensor(t)
        loss = -(t*torch.pow(1-ps,self.alpha_focal)*torch.log(ps) + (1-t)*torch.pow(ps,self.alpha_focal)*torch.log(1-ps))
        loss = torch.mean(loss)
        if numpy:
            loss = loss.cpu().detach().numpy()
        return loss

    def compute_calibration_error(self, ps_pred, ps_gt, numpy=False):
        if ps_gt is None:
            return None
        if not torch.is_tensor(ps_pred):
            ps_pred = torch.Tensor(ps_pred)
        if not torch.is_tensor(ps_gt):
            ps_gt = torch.Tensor(ps_gt)
        calibration_error = torch.mean(torch.abs(ps_gt - ps_pred))
        if numpy:
            calibration_error = calibration_error.cpu().detach().numpy()
        return calibration_error

    def training_step(self, batch, batch_idx):
        if self.dataset in ["news","ihdp"]:
            x, y, t, mu0, mu1, cate_true = batch
            ps_gt = None
        elif self.dataset in ["acic2016"]:
            x, y, t, mu0, mu1, cate_true, ps_gt = batch

        ps = self.forward(x)
        loss = self.compute_loss(ps, t)
        calibration_error = self.compute_calibration_error(ps, ps_gt)

        self.log('train_metric/loss', loss, on_step=False, on_epoch=True)
        if calibration_error is not None:
            self.log('train_metric/calibration_error', calibration_error, on_step=False, on_epoch=True)

        if self.eval_matching_during_training:
            matching_results, _, _ = self.matching_metrics(batch, n_neighbors=10)
            for metric in matching_results:
                if 'error_att' in metric or 'linear_mmd' in metric:
                    self.log('train_metric/' + metric, matching_results[metric], on_step=False, on_epoch=True)


        return loss


    def validation_step(self, batch, batch_idx):
        if self.dataset in ["news","ihdp"]:
            x, y, t, mu0, mu1, cate_true = batch
            ps_gt = None
        elif self.dataset in ["acic2016"]:
            x, y, t, mu0, mu1, cate_true, ps_gt = batch

        ps = self.forward(x)
        loss = self.compute_loss(ps, t)
        calibration_error = self.compute_calibration_error(ps, ps_gt)

        self.log('val_metric/loss', loss, on_step=False, on_epoch=True)
        if calibration_error is not None:
            self.log('val_metric/calibration_error', calibration_error, on_step=False, on_epoch=True)

        if self.eval_matching_during_training:
            matching_results, _, _ = self.matching_metrics(batch, n_neighbors=10)
            for metric in matching_results:
                if 'error_att' in metric or 'linear_mmd' in metric:
                    self.log('val_metric/' + metric, matching_results[metric], on_step=False, on_epoch=True)

        return loss


    def configure_optimizers(self):
        # define optimizer and scheduler
        optimizer = self.optimizer(filter(lambda p: p.requires_grad, self.parameters()), lr=self.init_lr, weight_decay=self.weight_decay)

        # training and evaluation loop
        epoch_lr = optimizer.param_groups[0]['lr']

        # TODO add support for learning rate scheduler
        # adjust learning rate
        # if epoch % args.update_lr_every_epoch == 0 and not epoch == 0:
        #     adjust_learning_rate(optimizer, epoch_lr, args.lr_decay, args.min_lr)

        return optimizer

    def get_scores(self, x):
        scores_ps_net = self.ps_net.get_scores(x)
        scores = {
            'balancing': scores_ps_net['balancing_1'],
            'propensity': scores_ps_net['propensity']
        }
        return scores

    def matching_metrics(self, batch, programme=None, metrics=None, att=True, atc=False, n_neighbors=10):
        if self.dataset in ["news","ihdp"]:
            x, y, t, mu0, mu1, cate_true = batch
            ps_gt = None
        elif self.dataset in ["acic2016"]:
            x, y, t, mu0, mu1, cate_true, ps_gt = batch

        if metrics is None:
            metrics = self.METRICS_DEFAULT

        self.eval()
        scores = self.get_scores(x)

        y_numpy = y.cpu().detach().numpy()
        t_numpy = t.cpu().detach().numpy()
        x_numpy = x.cpu().detach().numpy()
        cate_numpy = cate_true.cpu().detach().numpy()
        ps_gt_numpy = ps_gt.cpu().detach().numpy() if ps_gt is not None else None

        if programme is None:
            programme = [el for el in self.PROGRAMME_DEFAULT]
        programme += [
            ('nearest_neighbor_matching_replacement', dict(n_neighbors=n_neighbors, on=['balancing', 'propensity'])),
            ('get_treatment_effects', dict(y=y_numpy, evaluate=True, ites=cate_numpy)),
            ('get_balance_metrics', dict(x=x_numpy, add_nothing=True)),
        ]

        m = Matcher(scores, t_numpy, att=att, atc=atc)

        results = m.apply_programme(programme)
        if ps_gt_numpy is not None:
            loss_results = {
                'loss': self.compute_loss(scores['propensity'], t_numpy, numpy=True),
                'calibration_error': self.compute_calibration_error(scores['propensity'], ps_gt_numpy, numpy=True),
            }
        else:
            loss_results = {
                'loss': self.compute_loss(scores['propensity'], t_numpy, numpy=True),
            }

        balance_df = results[-1]
        te_df = results[-2]

        matching_results = {}
        for df in [te_df, balance_df]:
            matching_results.update(
                {
                    method + '_' + metric: df.loc[method, metric] \
                    for metric in df.columns if metric in metrics \
                    for method in df.index
                }
            )


        return matching_results, loss_results, scores

# Similar network but without a given balancing score layer
class SimpleNet(SimpleMatchNet):

    def __init__(self, in_dim, layer_dims, dataset, init_lr, activation='relu', dropout_prob=0.,
                 weight_decay=0., alpha_focal=0, optimizer="adam", eval_matching_during_training=True):

        super(SimpleMatchNet, self).__init__()

        self.optimizer = self.OPTIMIZERS[optimizer]
        self.layer_dims = layer_dims
        self.dataset = dataset
        self.init_lr = init_lr
        self.weight_decay = weight_decay
        self.alpha_focal = alpha_focal
        self.eval_matching_during_training = eval_matching_during_training

        self.ps_net = PropensityScoreFC(in_dim=in_dim, out_dim=1, layer_dims=layer_dims,
                                        dropout_prob=dropout_prob, activation=activation).to(self.device)