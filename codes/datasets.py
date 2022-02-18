
import torch
import numpy as np
import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from scipy.special import expit
import urllib
import zipfile

# Class reproducing the ACIC 2016 dataset
class ACIC2016(object):
    def __init__(self, setting=1, exp_num=1, one_hot_factors=True, device='cpu', dataset='train', path_data="../data", tensor=True, seed=0, train_size=0.6, val_size=0.2, scale=True):

        self.path_data = path_data
        self.one_hot_factors = one_hot_factors
        self.covariates = pd.read_csv(os.path.join(path_data,'ACIC2016/covariates.csv'))
        self.info = pd.read_csv(os.path.join(path_data,'ACIC2016/info.csv'))
        self.dgp_data = pd.read_csv(os.path.join(path_data,f'ACIC2016/dgp_data/setting{setting}_dataset{exp_num}.csv'))
        self.X_df = self._process_covariates(self.covariates) # turn factor variables into one-hot binary variables

        attrs = {}
        attrs['x'] = np.array(self.X_df)
        attrs['t'] = np.array(self.dgp_data['z']).reshape((-1, 1))
        attrs['y'] = np.array(self.dgp_data['y']).reshape((-1,1))
        attrs['y0'] = np.array(self.dgp_data['y.0']).reshape((-1, 1))
        attrs['y1'] = np.array(self.dgp_data['y.1']).reshape((-1, 1))
        attrs['mu0'] = np.array(self.dgp_data['mu.0']).reshape((-1, 1))
        attrs['mu1'] = np.array(self.dgp_data['mu.1']).reshape((-1, 1))
        attrs['cate_true'] = attrs['mu1'] - attrs['mu0']
        attrs['ps'] = np.array(self.dgp_data['e']).reshape((-1, 1))


        n_samples = self.X_df.shape[0]
        self.rng = np.random.default_rng(seed=seed)
        original_indices = self.rng.permutation(n_samples)
        n_train = int(train_size * n_samples)
        n_val = int(val_size * n_samples)
        itr = original_indices[:n_train] # train set
        iva = original_indices[n_train:n_train + n_val] # val set
        itrva = original_indices[:n_train + n_val] # train+val set
        ite = original_indices[n_train + n_val:]  # test set

        # Check that each factor is in train, val and test sets
        for covariate_name in self.covariates.columns:
            if not pd.to_numeric(self.covariates[covariate_name], errors='coerce').notnull().all():
                factors_tr = pd.unique(self.covariates[covariate_name].iloc[itr])
                factors_va = pd.unique(self.covariates[covariate_name].iloc[iva])
                factors_te = pd.unique(self.covariates[covariate_name].iloc[ite])
                assert set(factors_va).issubset(set(factors_tr))
                assert set(factors_te).issubset(set(factors_tr))

        if dataset == 'train':
            original_indices = itr
        elif dataset == 'val':
            original_indices = iva
        elif dataset == "train_val":
            original_indices = itrva
        else:
            original_indices = ite
        self.original_indices = original_indices

        # Find binary covariates
        self.binary = []
        for ind in range(attrs['x'].shape[1]):
            self.binary.append(len(np.unique(attrs['x'][:,ind])) == 2)
        self.binary = np.array(self.binary)

        # Normalise - continuous data
        self.scale = scale
        self.xm = np.zeros(self.binary.shape)
        self.xs = np.ones(self.binary.shape)
        if self.scale:
            self.xm[~self.binary] = np.mean(attrs['x'][itrva][:,~self.binary], axis=0)
            self.xs[~self.binary] = np.std(attrs['x'][itrva][:,~self.binary], axis=0)
        attrs['x'] -= self.xm
        attrs['x'] /= self.xs

        # Subsample data and convert to torch.Tensor with the right device
        for key, value in attrs.items():
            value = value[original_indices]
            if tensor:
                value = torch.Tensor(value).to(device)
            setattr(self, key, value)

    def __getitem__(self, index, attrs=None):
        if attrs is None:
            attrs = ['x', 'y', 't', 'mu0', 'mu1', 'cate_true', 'ps']
        res = []
        for attr in attrs:
            res.append(getattr(self, attr)[index])
        return (*res,)

    def __len__(self):
        return len(self.original_indices)




    def _process_covariates(self, covariates):
        covariates_done = {}
        for ind,covariate_name in enumerate(covariates.columns):
            if not 'x_' in covariate_name:
                continue
            if pd.to_numeric(covariates[covariate_name], errors='coerce').notnull().all():
                covariates_done[covariate_name] = covariates[covariate_name]
            else:
                if self.one_hot_factors:
                    for item in sorted(pd.unique(covariates[covariate_name])):
                       covariates_done[covariate_name + '_' + item] = (covariates[covariate_name] == item).astype(int)
                else:
                    covariates_done[covariate_name] = pd.Series([0]*len(covariates[covariate_name]))
                    for idx, item in sorted(enumerate(pd.unique(covariates[covariate_name]))):
                        covariates_done[covariate_name][covariates[covariate_name] == item] = idx

        return pd.DataFrame(covariates_done)




class IHDP(object):
    def __init__(self, exp_num, device='cpu', dataset='train', path_data="../data", tensor=True, normalise_y=True, split_original=False, rescale_cates=True, train_size=0.6, val_size=0.2, scale=True):
        self.path_data = path_data

        attrs = {}
        data = np.loadtxt(os.path.join(self.path_data,'IHDP/csv/ihdp_npci_' + str(exp_num) + '.csv'), delimiter=',')
        t, y, y_cf = data[:, 0][:, np.newaxis], data[:, 1][:, np.newaxis], data[:, 2][:, np.newaxis]
        mu_0, mu_1, x = data[:, 3][:, np.newaxis], data[:, 4][:, np.newaxis], data[:, 5:]
        x[:, 13] -= 1 # binary variable with values 1 and 2
        attrs['t'] = t
        attrs['y'] = y
        attrs['y_cf'] = y_cf
        attrs['mu0'] = mu_0
        attrs['mu1'] = mu_1
        attrs['cate_true'] = mu_1 - mu_0
        attrs['x'] = x

        if split_original: # unused in experiments - divide data as in code for CEVAE : https://github.com/AMLab-Amsterdam/CEVAE/blob/master/datasets.py
            idxtrain, ite = train_test_split(np.arange(x.shape[0]), test_size=0.1, random_state=1)
            itr, iva = train_test_split(idxtrain, test_size=0.3, random_state=1)
        else:
            n_samples = x.shape[0]
            rng = np.random.default_rng(seed=0)
            original_indices = rng.permutation(n_samples)
            n_train = int(train_size * n_samples)
            n_val = int(val_size * n_samples)
            itr = original_indices[:n_train] # train set
            iva = original_indices[n_train:n_train + n_val] # val set
            idxtrain = original_indices[:n_train + n_val] # train + val set
            ite = original_indices[n_train + n_val:]  # test set

        if rescale_cates: # rescale CATEs as in Curth et al. 2021 - see paper for ref
            sd_cate = np.sqrt(np.var(attrs['cate_true'][idxtrain]))
            if sd_cate > 1:
                error = y - t*mu_1 - (1-t)*mu_0
                mu_0 = mu_0 / sd_cate
                mu_1 = mu_1 / sd_cate
                y  = t*mu_1 + (1-t)*mu_0 + error
                attrs['y'] = y
                attrs['mu0'] = mu_0
                attrs['mu1'] = mu_1
                attrs['cate_true'] = mu_1 - mu_0

        ytr = y[itr]
        ym, ys = np.mean(ytr), np.std(ytr)
        self.ym = ym
        self.ys = ys
        if dataset == 'train':
            original_indices = itr
        elif dataset == 'val':
            original_indices = iva
        elif dataset == "train_val":
            original_indices = idxtrain
        else:
            original_indices = ite
        self.original_indices = original_indices

        # Find binary covariates
        self.binary = []
        for ind in range(attrs['x'].shape[1]):
            self.binary.append(len(np.unique(attrs['x'][:, ind])) == 2)
        self.binary = np.array(self.binary)

        # Normalise - continuous variables only
        self.scale = scale
        self.xm = np.zeros(self.binary.shape)
        self.xs = np.ones(self.binary.shape)
        if self.scale:
            self.xm[~self.binary] = np.mean(attrs['x'][idxtrain][:, ~self.binary], axis=0)
            self.xs[~self.binary] = np.std(attrs['x'][idxtrain][:, ~self.binary], axis=0)
        attrs['x'] -= self.xm
        attrs['x'] /= self.xs

        # Subsample data and convert to torch.Tensor with the right device
        for key, value in attrs.items():
            value = value[original_indices]
            if tensor:
                value = torch.Tensor(value).to(device)
            setattr(self, key, value)

        if normalise_y:
            self.y = (self.y - ym) / ys


    def __getitem__(self, index, attrs=None):
        if attrs is None:
            attrs = ['x', 'y', 't', 'mu0', 'mu1', 'cate_true']
        res = []
        for attr in attrs:
            res.append(getattr(self, attr)[index])
        return (*res,)

    def __len__(self):
        return len(self.original_indices)


class News():

    def __init__(self, exp_num, dataset='train', tensor=True, device="cpu", train_size=0.6, val_size=0.2,
                 data_folder=None, scale=True, seed=0):

        if data_folder is None:
            data_folder = '../data'

        # Create data if it does not exist
        if not os.path.isdir(os.path.join(data_folder, 'News/numpy_dicts/')):
            self._create_data(data_folder)

        with open(os.path.join(data_folder, 'News/numpy_dicts/data_as_dicts_with_numpy_seed_{}'.format(exp_num)),
                  'rb') as file:
            data = pickle.load(file)
        data['cate_true'] = data['mu1'] - data['mu0']

        # Create and store indices
        x = data['x']
        n_samples = x.shape[0]
        rng = np.random.default_rng(seed=seed)
        original_indices = rng.permutation(n_samples)
        n_train = int(train_size * n_samples)
        n_val = int(val_size * n_samples)
        itr = original_indices[:n_train] # train set
        iva = original_indices[n_train:n_train + n_val] # val set
        idxtrain = original_indices[:n_train + n_val] # train + val set
        ite = original_indices[n_train + n_val:]  # test set

        if dataset == 'train':
            original_indices = itr
        elif dataset == 'val':
            original_indices = iva
        elif dataset == "train_val":
            original_indices = idxtrain
        else:
            original_indices = ite
        self.original_indices = original_indices

        # Subsample data and convert to torch.Tensor with the right device
        for key, value in data.items():
            value = value[original_indices]
            if tensor:
                value = torch.Tensor(value).to(device)
            setattr(self, key, value)

    @staticmethod
    def _create_data(data_folder):

        print('News : no data, creating it')
        print('Downloading zipped csvs')
        urllib.request.urlretrieve('http://www.fredjo.com/files/NEWS_csv.zip', os.path.join(data_folder, 'News/csv.zip'))

        print('Unzipping csvs with sparse data')
        with zipfile.ZipFile(os.path.join(data_folder, 'News/csv.zip'), 'r') as zip_ref:
            zip_ref.extractall(os.path.join(data_folder, 'News'))

        print('Densifying the sparse data')
        os.mkdir(os.path.join(data_folder, 'News/numpy_dicts/'))

        for f_index in range(1, 50 + 1):
            mat = pd.read_csv(os.path.join(data_folder,'News/csv/topic_doc_mean_n5000_k3477_seed_{}.csv.x'.format(f_index)))
            n_rows, n_cols = int(mat.columns[0]), int(mat.columns[1])
            x = np.zeros((n_rows, n_cols)).astype(int)
            for i, j, val in zip(mat.iloc[:, 0], mat.iloc[:, 1], mat.iloc[:, 2]):
                x[i - 1, j - 1] = val
            data = {}
            data['x'] = x
            meta = pd.read_csv(
                os.path.join(data_folder, 'News/csv/topic_doc_mean_n5000_k3477_seed_{}.csv.y'.format(f_index)),
                names=['t', 'y', 'y_cf', 'mu0', 'mu1'])
            for col in ['t', 'y', 'y_cf', 'mu0', 'mu1']:
                data[col] = np.array(meta[col]).reshape((-1, 1))
            with open(os.path.join(data_folder, 'News/numpy_dicts/data_as_dicts_with_numpy_seed_{}'.format(f_index)), 'wb') as file:
                pickle.dump(data, file)

        print('Done!')

    def __getitem__(self, index, attrs=None):
        if attrs is None:
            attrs = ['x', 'y', 't', 'mu0', 'mu1', 'cate_true']
        res = []
        for attr in attrs:
            res.append(getattr(self, attr)[index])
        return (*res,)

    def __len__(self):
        return len(self.original_indices)

