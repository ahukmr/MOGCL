import torch
import torch as t
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np
from torch.nn.init import normal_
import random
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_uniform_initialization
from recbole.config import Config
from recbole.data import create_dataset, data_preparation, FullSortEvalDataLoader
from recbole.utils import init_logger, init_seed, set_color, EvaluatorType
import argparse
from recbole.trainer import Trainer
from recbole.utils import early_stopping, dict2str, set_color, get_gpu_usage
from torch.nn.utils.clip_grad import clip_grad_norm_
from tqdm import tqdm
from time import time
import argparse
from logging import getLogger
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_logger, init_seed, set_color

from recbole.utils import InputType


class MOGCL(GeneralRecommender):
    input_type = InputType.PAIRWISE
    def __init__(self, config, dataset):
        super(MOGCL, self).__init__(config, dataset)

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)

        # load parameters info
        self.latent_dim = config['embedding_size']
        self.n_layers = config['n_layers']
        self.reg_weight =  config['reg_weight']
        self.ssl_temp = config['ssl_temp']
        self.ssl_reg = config['ssl_reg']
        self.hyper_layers = config['hyper_layers']
        self.r = config['r']
        self.ssl_alpha = config['ssl_alpha']
        self.catweight = torch.tensor(config['catweight'])

        self.user_embedding = torch.nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.latent_dim)
        self.item_embedding = torch.nn.Embedding(num_embeddings=self.n_items, embedding_dim=self.latent_dim)

        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()
        self.restore_user_e = None
        self.restore_item_e = None

        self.acc_norm_adj_mat = self.acc_get_norm_adj_mat().to(self.device)
        self.div_norm_adj_mat = self.div_get_norm_adj_mat().to(self.device)

        # parameters initialization
        self.apply(xavier_uniform_initialization)
        self.apply(self._init_weights)
        self.other_parameter_name = ['restore_user_e', 'restore_item_e']

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            normal_(module.weight.data, 0, 0.01)
            if module.bias is not None:
                module.bias.data.fill_(0.0)
        elif isinstance(module, nn.Embedding):
            normal_(module.weight.data, 0, 0.01)

    def  acc_get_norm_adj_mat(self):
        r"""Get the normalized interaction matrix of users and items.

        Construct the square matrix from the training data and normalize it
        using the laplace matrix.

        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}

        Returns:
            Sparse tensor of the normalized interaction matrix.
        """
        # build adj matrix
        A = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix

        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid divide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        self.diag = torch.from_numpy(diag).to(self.device)
        D = sp.diags(diag)
        L = D @ A @ D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor([row, col])
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse_coo_tensor(i, data, torch.Size(L.shape))
        return SparseL

    def div_get_norm_adj_mat(self):
        r"""Get the normalized interaction matrix of users and items.

        Construct the square matrix from the training data and normalize it
        using the laplace matrix.


        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}

        Returns:
            Sparse tensor of the normalized interaction matrix.
        """
        # build adj matrix
        A = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        A._update(data_dict)
        sumArr_left = (A > 0).sum(axis=1)
        diag_left = np.array(sumArr_left.flatten())[0] + 1e-7
        diag_left = np.power(diag_left, -self.r)
        sumArr_right = (A > 0).sum(axis=1)
        diag_right = np.array(sumArr_right.flatten())[0] + 1e-7
        diag_right = np.power(diag_right, -(1-self.r))

        self.diag_left = torch.from_numpy(diag_left).to(self.device)
        self.diag_right = torch.from_numpy(diag_right).to(self.device)
        D_left = sp.diags(diag_left)
        D_right = sp.diags(diag_right)
        L = D_left @ A @ D_right
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor([row, col])
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse_coo_tensor(i, data, torch.Size(L.shape))
        return SparseL


    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.

        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def forward(self):
        acc_all_embeddings = self.get_ego_embeddings()
        acc_embeddings_list = [acc_all_embeddings]
        div_all_embeddings = self.get_ego_embeddings()
        div_embeddings_list = [div_all_embeddings]
        for layer_idx in range(self.n_layers):
            acc_all_embeddings = torch.sparse.mm(self.acc_norm_adj_mat, acc_all_embeddings)
            div_all_embeddings = torch.sparse.mm(self.div_norm_adj_mat, div_all_embeddings)
            acc_embeddings_list.append(acc_all_embeddings)    # 得到每一layer的embedding
            div_embeddings_list.append(div_all_embeddings)
        lightgcn_acc_all_embeddings = torch.stack(acc_embeddings_list[:self.n_layers+1], dim=1)   # 沿着一个新维度对输入张量序列进行连接。 序列中所有的张量都应该为相同形状。
        lightgcn_acc_all_embeddings = torch.mean(lightgcn_acc_all_embeddings, dim=1)

        user_acc_all_embeddings, item_acc_all_embeddings = torch.split(lightgcn_acc_all_embeddings, [self.n_users, self.n_items])

        lightgcn_div_all_embeddings = torch.stack(div_embeddings_list[:self.n_layers + 1],
                                                  dim=1)
        lightgcn_div_all_embeddings = torch.mean(lightgcn_div_all_embeddings, dim=1)

        user_div_all_embeddings, item_div_all_embeddings = torch.split(lightgcn_div_all_embeddings,
                                                               [self.n_users, self.n_items])
        user_fil_emb = self.catweight * user_acc_all_embeddings + (1 - self.catweight) * user_div_all_embeddings
        item_fil_emb = self.catweight * item_acc_all_embeddings + (1 - self.catweight) * item_div_all_embeddings
        return lightgcn_acc_all_embeddings, lightgcn_div_all_embeddings,user_fil_emb,item_fil_emb

    def Ssl_loss(self, acc_embedding, div_embedding, user, item):
        acc_user_embeddings, acc_item_embeddings = torch.split(acc_embedding, [self.n_users, self.n_items])
        div_user_embeddings_all, div_item_embeddings_all = torch.split(div_embedding,
                                                                                 [self.n_users, self.n_items])

        acc_user_embeddings = acc_user_embeddings[user]
        div_user_embeddings = div_user_embeddings_all[user]
        norm_user_emb1 = F.normalize(acc_user_embeddings)
        norm_user_emb2 = F.normalize(div_user_embeddings)
        norm_all_user_emb = F.normalize(div_user_embeddings_all)
        pos_score_user = torch.mul(norm_user_emb1, norm_user_emb2).sum(dim=1)
        ttl_score_user = torch.matmul(norm_user_emb1, norm_all_user_emb.transpose(0, 1))
        pos_score_user = torch.exp(pos_score_user / self.ssl_temp)
        ttl_score_user = torch.exp(ttl_score_user / self.ssl_temp).sum(dim=1)

        ssl_loss_user = -torch.log(pos_score_user / ttl_score_user).sum()

        acc_item_embeddings = acc_item_embeddings[item]
        div_item_embeddings = div_item_embeddings_all[item]
        norm_item_emb1 = F.normalize(acc_item_embeddings)
        norm_item_emb2 = F.normalize(div_item_embeddings)
        norm_all_item_emb = F.normalize(div_item_embeddings_all)
        pos_score_item = torch.mul(norm_item_emb1, norm_item_emb2).sum(dim=1)
        ttl_score_item = torch.matmul(norm_item_emb1, norm_all_item_emb.transpose(0, 1))
        pos_score_item = torch.exp(pos_score_item / self.ssl_temp)
        ttl_score_item = torch.exp(ttl_score_item / self.ssl_temp).sum(dim=1)

        ssl_loss_item = -torch.log(pos_score_item / ttl_score_item).sum()

        ssl_loss = self.ssl_reg * (ssl_loss_user + self.ssl_alpha * ssl_loss_item)
        return ssl_loss
    def calculate_loss(self, interaction):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        acc_all_embeddings, div_all_embeddings,user_all_embeddings,item_all_embeddings= self.forward()
        ssl_loss = self.Ssl_loss(acc_all_embeddings, div_all_embeddings, user, pos_item)

        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]

        # calculate BPR Loss
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)

        mf_loss = self.mf_loss(pos_scores, neg_scores)



        u_ego_embeddings = self.user_embedding(user)
        pos_ego_embeddings = self.item_embedding(pos_item)
        neg_ego_embeddings = self.item_embedding(neg_item)

        reg_loss = self.reg_loss(u_ego_embeddings, pos_ego_embeddings, neg_ego_embeddings)
        return  mf_loss + self.reg_weight * reg_loss,ssl_loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        acc_all_embeddings, div_all_embeddings, user_all_embeddings, item_all_embeddings = self.forward()
        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            _, _, self.restore_user_e, self.restore_item_e = self.forward()

        u_embeddings = self.restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))

        return scores.view(-1)







