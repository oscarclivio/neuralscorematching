import torch
import torch.distributions as D
import torch.nn as nn
import numpy as np
import abc

from utils import build_fc_network

# TARNet-style network returning model parameters for given likelihoods
class LikelihoodTARNet(nn.Module):

    def __init__(self, in_dim, layer_dims_core, layer_dims_branch, out_dim, likelihoods=[], activation = "relu", dropout_prob = 0., scale=True):

        super(LikelihoodTARNet, self).__init__()

        self.core = build_fc_network([in_dim] + layer_dims_core, activation=activation, dropout_prob=dropout_prob)  # most layers of the model

        # t = 1 branch
        self.branch_1 = LikelihoodFC(in_dim=layer_dims_core[-1] if len(layer_dims_core) > 0 else in_dim, layer_dims=layer_dims_branch, out_dim=out_dim, likelihoods=likelihoods, activation=activation, dropout_prob=dropout_prob, scale=scale)

        # t = 0 branch
        self.branch_0 = LikelihoodFC(in_dim=layer_dims_core[-1] if len(layer_dims_core) > 0 else in_dim, layer_dims=layer_dims_branch, out_dim=out_dim, likelihoods=likelihoods, activation=activation, dropout_prob=dropout_prob, scale=scale)

    def forward(self, x, t, return_all=False):
        h = self.core(x)

        # t = 1 branch
        params_1 = self.branch_1(h)

        # t = 0 branch
        params_0 = self.branch_0(h)

        # merge and return
        params = {}
        for lkl in params_1.keys():
            params[lkl] = {}
            for param in params_1[lkl].keys():
                params[lkl][param] =t*params_1[lkl][param] + (1-t)*params_0[lkl][param]

        if return_all:
            return params, params_1, params_0
        else:
            return params

# TARNet-style network returning parameters of Gaussian distribution
class GaussianTARNet(nn.Module):

    def __init__(self, in_dim, layer_dims_core, layer_dims_branch, out_dim, activation = "relu", dropout_prob = 0.):

        super(GaussianTARNet, self).__init__()

        self.core = build_fc_network([in_dim] + layer_dims_core, activation=activation, dropout_prob=dropout_prob)  # most layers of the model

        # t = 1 branch
        self.branch_1 = build_fc_network([layer_dims_core[-1]] + layer_dims_branch, activation=activation, dropout_prob=dropout_prob)
        self.fc_mu_1 = nn.Linear(layer_dims_branch[-1],
                                 out_dim)  # layer that outputs the mean parameter of the Gaussian distribution; no activation function, since on continuous scale
        self.fc_log_sigma_1 = nn.Linear(layer_dims_branch[-1],
                                        out_dim)  # layer that outputs the logarithm of the Gaussian distribution (diagonal covariance structure); no activation function, since on continuous scale

        # t = 0 branch
        self.branch_0 = build_fc_network([layer_dims_core[-1]] + layer_dims_branch, activation=activation,
                                         dropout_prob=dropout_prob)
        self.fc_mu_0 = nn.Linear(layer_dims_branch[-1],
                                 out_dim)  # layer that outputs the mean parameter of q(z | x); no activation function, since on continuous scale
        self.fc_log_sigma_0 = nn.Linear(layer_dims_branch[-1],
                                        out_dim)  # layer that outputs the logarithm of the variance parameter of q(z | x) (diagonal covariance structure); no activation function, since on continuous scale

    def forward(self, x, t, return_all=False):
        h = self.core(x)

        # t = 1 branch
        h_1 = self.branch_1(h)
        mu_1 = self.fc_mu_1(h_1)
        log_sigma_square_1 = self.fc_log_sigma_1(h_1)

        # t = 0 branch
        h_0 = self.branch_0(h)
        mu_0 = self.fc_mu_0(h_0)
        log_sigma_square_0 = self.fc_log_sigma_0(h_0)

        # merge and return
        mu = t*mu_1 + (1-t)*mu_0
        log_sigma_square = t*log_sigma_square_1 + (1-t)*log_sigma_square_0

        if return_all:
            return mu, log_sigma_square, mu_1, log_sigma_square_1, mu_0, log_sigma_square_0
        else:
            return mu, log_sigma_square


DISTRIBUTIONS = {
    "cont": D.Normal,
    "bin": D.Bernoulli,
    "poisson": D.Poisson
}

# Feed-forward full-connected network returning parameters of given likelihoods
class LikelihoodFC(nn.Module):
    def __init__(self, in_dim, layer_dims, out_dim, likelihoods=[], activation="relu", dropout_prob=0., scale=True):

        if isinstance(likelihoods, str):
            likelihoods = np.array([likelihoods])
        assert (len(likelihoods) == out_dim)
        self.masks = {}
        self.lengths = {}
        for lkl in DISTRIBUTIONS.keys():
            self.masks[lkl] = (likelihoods == lkl).ravel()
            self.lengths[lkl] = int((likelihoods == lkl).sum())

        super(LikelihoodFC, self).__init__()

        self.core = build_fc_network([in_dim] + layer_dims, activation=activation,
                                     dropout_prob=dropout_prob)  # most layers of the model
        self.lkl_head_activations = {
            "cont": {"loc": None, "scale": (lambda s: torch.sqrt(torch.exp(scale*s)))},
            "bin": {"logits": None},
            "poisson": {"rate": (lambda s: torch.clamp(torch.exp(s), min=1e-10))},
        }

        self.lkl_head_layers = nn.ModuleDict({
            lkl: nn.ModuleDict({param: nn.Linear(layer_dims[-1], self.lengths[lkl]) for param in
                                self.lkl_head_activations[lkl].keys()}) for lkl in self.lkl_head_activations.keys()
        })
        assert len([lkl for lkl in self.lkl_head_activations.keys() if lkl not in DISTRIBUTIONS]) == 0

    def forward(self, x):
        h = self.core(x)
        params = {}
        for lkl in self.lkl_head_layers.keys():
            params[lkl] = {param: self.lkl_head_layers[lkl][param](h) for param in self.lkl_head_layers[lkl].keys()}
            for param in params[lkl].keys():
                if self.lkl_head_activations[lkl][param] is not None:
                    params[lkl][param] = self.lkl_head_activations[lkl][param](params[lkl][param])

        return params

# Feed-forward full-connected network returning parameters of Gaussian distribution
class GaussianFC(nn.Module):

    def __init__(self, in_dim, layer_dims, out_dim, activation = "relu", dropout_prob = 0.):

        super(GaussianFC, self).__init__()

        self.core = build_fc_network([in_dim] + layer_dims, activation=activation, dropout_prob=dropout_prob)  # most layers of the model
        self.fc_mu = nn.Linear(layer_dims[-1],
                               out_dim)  # layer that outputs the mean parameter of the Gaussian distribution ; no activation function, since on continuous scale
        self.fc_log_sigma = nn.Linear(layer_dims[-1],
                                      out_dim)  # layer that outputs the logarithm of the Gaussian distribution (diagonal covariance structure); no activation function, since on continuous scale


    def forward(self, x):
        h = self.core(x)
        mu = self.fc_mu(h)
        log_sigma_square = self.fc_log_sigma(h)

        return mu, log_sigma_square

# Feed-forward full-connected network returning parameter of Bernoulli distribution
class BernoulliFC(nn.Module):

    def __init__(self, in_dim, layer_dims, out_dim, activation = "relu", dropout_prob = 0.):

        super(BernoulliFC, self).__init__()

        if len(layer_dims) > 0:
            self.core = build_fc_network([in_dim] + layer_dims, activation=activation,
                                         dropout_prob=dropout_prob)  # most layers of the model
            self.fc_p = nn.Linear(layer_dims[-1],
                                  out_dim)  # layer that outputs the logit of the Bernoulli parameter
        else:
            self.core = build_fc_network([], activation=activation, dropout_prob=dropout_prob)
            self.fc_p = nn.Linear(in_dim, out_dim)



    def forward(self, x):

        h = self.core(x)
        p = self.fc_p(h)
        p = torch.sigmoid(p)

        return p

# Feed-forward full-connected network returning parameters of Bernoulli distribution (here a propensity score model)
# but is also able to return intermediate balancing scores
class PropensityScoreFC(BernoulliFC):

    def __init__(self, *args, **kwargs):

        super(PropensityScoreFC, self).__init__(*args, **kwargs)

        self.scores = {}
        self.create_hooks()

    def _get_scores(self, name, sigmoid=False):
        def hook(model, input, output):
            if sigmoid:
                output = torch.sigmoid(output)
            self.scores[name] = output.cpu().detach().numpy()
        return hook

    def create_hooks(self):
        ind_balancing = 1
        for ind in range(len(self.core)):
            if "linear" in self.core[ind].__str__().lower():
                self.core[ind].register_forward_hook(self._get_scores(name='balancing_{}'.format(ind_balancing)))
                ind_balancing += 1
        self.fc_p.register_forward_hook(self._get_scores(name='propensity'.format(ind_balancing), sigmoid=True))

    def get_scores(self, x):
        p = self.forward(x)
        return dict(self.scores)