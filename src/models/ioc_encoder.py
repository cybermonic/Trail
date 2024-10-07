import os
import sys

import pandas as pd
import torch
from torch import nn

from copy import deepcopy

sys.path.append('..')
from feature_extraction import ip, domain, url
from feature_extraction.utils import order_df_cols

class FeatureSampler(nn.Module):
    '''
    Loads feature vectors projected into lower space from
    pandas dataframe
    '''
    NUM_NODE_TYPES = 5
    def __init__(self, out_dim, feat_dir):
        super().__init__()

        # The other option is to convert all the csvs to dense np files
        # but that takes a LOT of space (~5GB per file) while this stays
        # in the MB range
        print('Loading IPs')
        self.ips = order_df_cols(
            pd.read_csv(
                os.path.join(feat_dir, 'ips.csv'),
                sep='\t'
            ), ip.COL_ORDER
        )
        print('Loading domains')
        self.domains = order_df_cols(
            pd.read_csv(
                os.path.join(feat_dir, 'domains.csv'),
                sep='\t'
            ), domain.COL_ORDER
        )
        print('Loading URLs')
        self.urls = order_df_cols(
            pd.read_csv(
                os.path.join(feat_dir, 'urls.csv'),
                sep='\t'
            ), url.COL_ORDER
        )

        # Not dealing with dates
        '''
        for df in [self.ips, self.domains, self.urls]:
            if 'first_seen' in df:
                df.pop('first_seen')
            if 'last_seen' in df:
                df.pop('last_seen')
        '''

        self.ip_size = ip.to_dense_gnn(self.ips.iloc[:1])[0].shape[1]
        self.domain_size = domain.to_dense_gnn(self.domains.iloc[:1])[0].shape[1]
        self.url_size = url.to_dense_gnn(self.urls.iloc[:1])[0].shape[1]
        self.out_dim = out_dim

        self.ip_net = nn.Linear(self.ip_size, out_dim-self.NUM_NODE_TYPES)
        self.domain_net = nn.Linear(self.domain_size, out_dim-self.NUM_NODE_TYPES)
        self.url_net = nn.Linear(self.url_size, out_dim-self.NUM_NODE_TYPES)

        self.type_map = {
            'ips': (ip.to_dense_gnn, self.ips, self.ip_net),
            'domains': (domain.to_dense_gnn, self.domains, self.domain_net),
            'urls': (url.to_dense_gnn, self.urls, self.url_net)
        }

    def sample_one(self, g, batch, ntype, labels=False):
        nodes_searched = g.x[batch] == g.type_dict[ntype]
        sample = batch[nodes_searched]
        if sample.size(0) == 0:
            return None, None

        df_idx = g.feat_map[sample]
        mask = df_idx != -1
        df_idx = df_idx[mask]
        nodes_searched[nodes_searched==True] = nodes_searched[nodes_searched==True].logical_and(mask)

        if df_idx.size(0) == 0:
            return None, None

        # Has samples that we can grab
        dense, df, net = self.type_map[ntype]
        x,y,_ = dense(df.iloc[df_idx])
        x = torch.from_numpy(x).float()

        if not labels:
            return nodes_searched, net(x)
        return nodes_searched, (net(x), y)

    def forward(self, g, batch):
        x = torch.zeros(batch.size(0), self.out_dim-self.NUM_NODE_TYPES)

        for ntype in ['ips', 'domains', 'urls']:
            idx, value = self.sample_one(g, batch, ntype)
            if idx is not None:
                x[idx] += value

        oh = torch.zeros(batch.size(0), g.x.max()+1)
        oh[torch.arange(batch.size(0)), g.x[batch]] = 1.

        return torch.cat([oh, x], dim=1)

    def node_names(self, g, batch):
        pass

class FeatureGetter(FeatureSampler):
    '''
    Same as above but doesn't encode features with NN
    '''
    def sample_one(self, g, batch, ntype, labels=False):
        nodes_searched = g.x[batch] == g.type_dict[ntype]
        sample = batch[nodes_searched]
        if sample.size(0) == 0:
            return None, None

        df_idx = g.feat_map[sample]
        mask = df_idx != -1
        df_idx = df_idx[mask]
        nodes_searched[nodes_searched==True] = nodes_searched[nodes_searched==True].logical_and(mask)

        if df_idx.size(0) == 0:
            return None, None

        # Has samples that we can grab
        dense, df, net = self.type_map[ntype]
        x,y,_ = dense(df.iloc[df_idx])
        x = torch.from_numpy(x).float()

        if not labels:
            return nodes_searched, x
        return nodes_searched, (x, y)

    def _mask_to_idx(self, mask):
        new = torch.full(mask.size(), -1)
        cnt = 0
        for i in range(mask.size(0)):
            if mask[i]:
                new[i] = cnt
                cnt += 1

        return new

    def forward(self, g, batch):
        ret = []
        for ntype in ['ips', 'domains', 'urls']:
            mask, iocs = self.sample_one(g, batch, ntype)
            idx = self._mask_to_idx(mask)
            ret.append( (idx, iocs) )

        return ret


class FeatureAutoencoder(FeatureSampler):
    def __init__(self, out_dim, feat_dir, hidden=512, layers=2):
        super().__init__(out_dim, feat_dir)

        def layer(in_d, out_d, act=nn.ReLU, act_args=()):
            return nn.Sequential(
                nn.Linear(in_d, out_d),
                nn.Dropout(0.1),
                act(*act_args)
            )

        def decoder(target_dim):
            return nn.Sequential(
                layer(out_dim-self.NUM_NODE_TYPES, hidden),
                *[layer(hidden, hidden) for _ in range(layers-2)],
                layer(hidden, target_dim, act=nn.Identity)
            )

        def encoder(input_dim):
            return nn.Sequential(
                layer(input_dim, hidden),
                *[layer(hidden, hidden) for _ in range(layers-2)],
                layer(hidden, out_dim-self.NUM_NODE_TYPES)
            )

        self.ip_net = encoder(self.ip_size)
        self.url_net = encoder(self.url_size)
        self.domain_net = encoder(self.domain_size)

        self.ip_dec = decoder(self.ip_size)
        self.url_dec = decoder(self.url_size)
        self.domain_dec = decoder(self.domain_size)

        self.type_map = {
            'ips': (ip.to_dense_gnn, self.ips, self.ip_net, self.ip_dec),
            'domains': (domain.to_dense_gnn, self.domains, self.domain_net, self.domain_dec),
            'urls': (url.to_dense_gnn, self.urls, self.url_net, self.url_dec)
        }

        self.criterion = nn.MSELoss()
        self.losses = []

    def sample_one(self, g, batch, ntype, labels=False):
        nodes_searched = g.x[batch] == g.type_dict[ntype]
        sample = batch[nodes_searched]
        if sample.size(0) == 0:
            return None, None

        df_idx = g.feat_map[sample]
        mask = df_idx != -1
        df_idx = df_idx[mask]
        nodes_searched[nodes_searched==True] = nodes_searched[nodes_searched==True].logical_and(mask)

        if df_idx.size(0) == 0:
            return None, None

        # Has samples that we can grab
        dense, df, enc, dec = self.type_map[ntype]
        x,y,_ = dense(df.iloc[df_idx])
        x = torch.from_numpy(x).float()

        z = enc(x)

        # Need to calculate reconstruction loss if training
        if self.training:
            x_hat = dec(z)
            loss = (x - x_hat).pow(2).mean()
            self.losses.append(loss)

        if not labels:
            return nodes_searched, z
        return nodes_searched, (z, y)

    def get_loss(self):
        if self.losses == []:
            return torch.tensor(0)

        loss = torch.stack(self.losses)
        loss = loss.mean()
        self.losses = []
        return loss

class VariationalFeatureAutoencoder(FeatureAutoencoder):
    def __init__(self, out_dim, feat_dir, hidden=512, layers=2):
        super().__init__(out_dim, feat_dir)

        def layer(in_d, out_d, act=nn.LeakyReLU, act_args=()):
            return nn.Sequential(
                nn.Linear(in_d, out_d),
                nn.Dropout(0.1),
                act(*act_args)
            )

        def encoder(input_dim):
            return nn.Sequential(
                layer(input_dim, hidden),
                *[layer(hidden, hidden) for _ in range(layers-2)],
                layer(hidden, hidden)
            )

        self.ip_net = encoder(self.ip_size)
        self.url_net = encoder(self.url_size)
        self.domain_net = encoder(self.domain_size)

        out = out_dim-self.NUM_NODE_TYPES
        self.ip_mu = nn.Linear(hidden, out)
        self.ip_log_var = nn.Linear(hidden, out)
        self.url_mu = nn.Linear(hidden, out)
        self.url_log_var = nn.Linear(hidden, out)
        self.domain_mu = nn.Linear(hidden, out)
        self.domain_log_var = nn.Linear(hidden, out)

        self.type_map = {
            'ips': (ip.to_dense_gnn, self.ips, self.ip_net, self.ip_mu, self.ip_log_var, self.ip_dec),
            'domains': (domain.to_dense_gnn, self.domains, self.domain_net, self.domain_mu, self.domain_log_var, self.domain_dec),
            'urls': (url.to_dense_gnn, self.urls, self.url_net, self.url_mu, self.url_log_var, self.url_dec)
        }

        self.criterion = nn.MSELoss()
        self.losses = []

    def sample_one(self, g, batch, ntype, labels=False):
        nodes_searched = g.x[batch] == g.type_dict[ntype]
        sample = batch[nodes_searched]
        if sample.size(0) == 0:
            return None, None

        df_idx = g.feat_map[sample]
        mask = df_idx != -1
        df_idx = df_idx[mask]
        nodes_searched[nodes_searched==True] = nodes_searched[nodes_searched==True].logical_and(mask)

        if df_idx.size(0) == 0:
            return None, None

        # Has samples that we can grab
        dense, df, in_net, mu_net, var_net, dec = self.type_map[ntype]
        x,y,_ = dense(df.iloc[df_idx])
        x_orig = torch.from_numpy(x).float()

        x = in_net(x_orig)
        mu = mu_net(x)

        # Need to calculate reconstruction loss if training
        if self.training:
            logvar = var_net(x)
            z = self.__reparam(mu, logvar)
            x_hat = dec(z)

            r_loss = (x_orig - x_hat).pow(2).mean()
            kld_loss = self.__kld_loss(mu, logvar)

            # Using 0.1 as weight based on discussions in this thread:
            # https://stats.stackexchange.com/questions/332179/how-to-weight-kld-loss-vs-reconstruction-loss-in-variational-auto-encoder
            self.losses.append(r_loss + 0.1*kld_loss)

        # During eval just use mean as embedding with no randomness
        else:
            z = mu

        if not labels:
            return nodes_searched, z

        return nodes_searched, (z, y)

    def __reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def __kld_loss(self, mu, logvar):
        return torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)