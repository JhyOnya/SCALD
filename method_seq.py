import pandas as pd
import numpy as np

from scipy import stats

from utils_jhy import printT

from torch.distributions import normal
from torch import optim
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# set random seeds:
def setup_seed(seed=1000):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # random.seed(seed)
    torch.backends.cudnn.deterministic = True


def save_models(stat_params, cache_dir):
    # save_models(stat_params, args.cache_dir, i)
    stat_params_state_dict = {name: param_pre.state_dict() for name, param_pre in stat_params.items()}
    torch.save(stat_params_state_dict, "%sbest_model.pth" % (cache_dir))


def restore_models(stat_params, cache_dir):
    # restore_models(stat_params, args.cache_dir, i)
    checkpoint = torch.load("%sbest_model.pth" % (cache_dir))
    for name, param_pre in stat_params.items():
        param_pre.load_state_dict(checkpoint[name])
        param_pre.eval()


class LocallyConnected(nn.Module):
    def __init__(self, num_linear, input_features, output_features):
        super(LocallyConnected, self).__init__()

        self.weight = nn.Parameter(torch.rand(num_linear, input_features, output_features))
        self.bias = nn.Parameter(torch.rand(num_linear, output_features))

    def forward(self, input):
        # [n, d, 1, m2] = [n, d, 1, m1] @ [1, d, m1, m2]
        out = torch.matmul(input.unsqueeze(dim=2), self.weight.unsqueeze(dim=0))
        return out.squeeze(dim=2) + self.bias


class p_A_x_func(nn.Module):
    def __init__(self, dim_feat, tf_len=195, dim_h=256, nh=2):
        super().__init__()
        self.dim_feat = dim_feat
        self.tf_len = tf_len

        self.A = torch.nn.Parameter(torch.zeros((tf_len, dim_feat), requires_grad=True))

        dim_list = [dim_feat, int(dim_feat / 2), 1]
        layers = []
        for l in range(len(dim_list) - 1):
            layers.append(LocallyConnected(dim_feat, dim_list[l], dim_list[l + 1]))
        self.hidden = nn.ModuleList(layers)

        self.pad = nn.ZeroPad2d(padding=(0, 0, 0, self.dim_feat - self.tf_len))  # 左右上下

    def forward(self, x, drop=True):
        A = self.pad(torch.sigmoid(self.A)).fill_diagonal_(0)  # [d, md]
        pre = torch.mul(x.unsqueeze(1),A.T.unsqueeze(0))  # [n, d, md] = [n, 1, d] x [1, d, md]
        for module in self.hidden:
            pre = F.elu(module(pre), inplace=True)
        return pre.squeeze(dim=2)

    def get_A(self):
        return self.pad(torch.sigmoid(self.A)).fill_diagonal_(0)


def get_subnets(data_np, device, tf_len):
    spearman_np, p_np = stats.spearmanr(data_np)
    np.fill_diagonal(spearman_np, 0)
    np.fill_diagonal(p_np, 1)

    orient_adv = np.ones_like(spearman_np)
    orient_adv[tf_len:, :] = 0
    orient_adv[:, tf_len:] = 0

    orient_dec = np.ones_like(spearman_np)
    orient_dec[tf_len:, :] = 0
    orient_dec[:, tf_len:] = 0

    if data_np.shape[0] <= 20:
        shd = 0.05
    elif data_np.shape[0] <= 100:
        shd = 0.001
    else:
        shd = 0.0005

    orient_adv[spearman_np <= 0.1] = 0
    orient_dec[spearman_np >= -0.1] = 0
    orient_adv[p_np > shd] = 0
    orient_dec[p_np > shd] = 0

    orient_adv_gpu = torch.tensor((orient_adv != 0), dtype=torch.bool, requires_grad=False, device=device)
    orient_dec_gpu = torch.tensor((orient_dec != 0), dtype=torch.bool, requires_grad=False, device=device)
    # orient_adv_gpu = torch.tensor(orient_adv, dtype=torch.float32, requires_grad=False, device=device)
    # orient_dec_gpu = torch.tensor(orient_dec, dtype=torch.float32, requires_grad=False, device=device)

    return orient_adv_gpu, orient_dec_gpu


def train(data, args, device, **kw):
    def h_dag_com(A, orient_adv_gpu, orient_dec_gpu, **kw):

        B = torch.matrix_power(orient_adv_gpu * A * A / (d * d) + torch.eye(d, device=device, requires_grad=False), d)

        C_ = (orient_dec_gpu * A * A / (d * d)) @ B
        C = torch.matrix_power(C_ @ C_ + torch.eye(d, device=device, requires_grad=False), int(d / 2))

        return torch.trace(B).sum() - d, torch.trace(C).sum() - d  # l_dag_adv, l_dag_com

    batch_num = 32
    max_comp = 50
    printT("data.shape", data.shape)
    n = data.shape[0]
    d = data.shape[1]
    tf_len = len(kw['TF_ids_list'])

    setup_seed()
    data_np_ori = data.values
    dt_min = np.min(data_np_ori, axis=0)
    dt_max = np.max(data_np_ori, axis=0)
    data_np = (data_np_ori - dt_min) / (dt_max - dt_min) + 1e-8
    orient_adv_gpu, orient_dec_gpu = get_subnets(data_np, device, tf_len)

    p_A_x_dist = p_A_x_func(dim_feat=d, tf_len=tf_len)
    p_A_x_dist.to(device, non_blocking=True)
    # p_A_x_dist.half()
    p_A_x_dist.train()

    stat_params = {'p_A_x': p_A_x_dist, }
    params = [pre for v in stat_params.values() for pre in list(v.parameters())]

    loss_MSE = torch.nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(params, lr=1e-2, weight_decay=1e-3)  # lr 学习率 wd 权重衰减

    n_iter_batch, idx = int(n / batch_num), list(range(n))

    loss_list = defaultdict(list)
    best_l_epoch = np.inf

    epoch = 0
    pre_comp = 0

    lambda_com, pho_com, l_dag_com_last = 0, 1e-16, np.inf
    lambda_adv, pho_adv, l_dag_adv_last = 0, 1e-16, np.inf

    while True:
        if pre_comp >= max_comp and epoch >= 500:
            break
        epoch += 1
        np.random.shuffle(idx)

        for batch in range(n_iter_batch):
            id_batch = np.random.choice(idx, batch_num, replace=False)
            # data_batch = torch.tensor(data_np[id_batch], requires_grad=False,device=device).half()
            data_batch = torch.tensor(data_np[id_batch], dtype=torch.float32, requires_grad=False, device=device)
            # data_batch = torch.from_numpy(data_np[id_batch]).float().to(device)
            data_pred = p_A_x_dist(data_batch)
            A = p_A_x_dist.get_A()

            l_A = loss_MSE(data_batch, data_pred)
            l_dag_adv, l_dag_com = h_dag_com(A, orient_adv_gpu, orient_dec_gpu)

            loss = torch.sum(l_A) + 0.01 * torch.sum(A ** 2) \
                   + lambda_adv * l_dag_adv + 0.5 * pho_adv * l_dag_adv * l_dag_adv \
                   + lambda_com * l_dag_com + 0.5 * pho_com * l_dag_com * l_dag_com

            with torch.no_grad():
                loss_list['l'].append(float(loss))
                loss_list['l_A'].append(float(torch.sum(l_A)))
                loss_list['l_dag_adv'].append(float(l_dag_adv))
                loss_list['l_dag_com'].append(float(l_dag_com))

            loss.backward()  # 反向传播计算每个参数的梯度
            # torch.nn.utils.clip_grad_norm_(params, max_norm=3, norm_type=2)
            optimizer.step()  # 梯度下降参数更新
            optimizer.zero_grad()  # 梯度归零

        # print("epoch: %s/%s  batch: %s/%s  loss=%s" % (epoch, args.epochs, batch, n_iter_batch, loss))
        # print("epoch: %s/%s  batch: %s/%s  "
        #       "l_A=%s l_dag_adv=%s l_dag_dec=%s" % (epoch, args.epochs, batch, n_iter_batch,
        #                                             torch.mean(l_A).cpu().detach().numpy(),
        #                                             l_dag_adv.cpu().detach().numpy(),
        #                                             l_dag_dec.cpu().detach().numpy()))

        l_pre_epoch = np.mean(loss_list['l'][-n_iter_batch:])
        l_A_pre_epoch = np.mean(loss_list['l_A'][-n_iter_batch:])
        l_dag_adv_pre_epoch = np.mean(loss_list['l_dag_adv'][-n_iter_batch:])
        l_dag_com_pre_epoch = np.mean(loss_list['l_dag_com'][-n_iter_batch:])

        lambda_adv += pho_adv * l_dag_adv_pre_epoch
        lambda_com += pho_com * l_dag_com_pre_epoch

        if l_dag_adv_pre_epoch >= 0.8 * l_dag_adv_last and pho_adv < 1e+16:
            pho_adv *= 10
        if l_dag_com_pre_epoch >= 0.8 * l_dag_com_last and pho_com < 1e+16:
            pho_com *= 10

        l_dag_adv_last = l_dag_adv_pre_epoch
        l_dag_com_last = l_dag_com_pre_epoch

        if l_A_pre_epoch <= best_l_epoch:
            pre_comp = 0
            best_l_epoch = l_A_pre_epoch
            save_models(stat_params, args.cache_dir)
            printT("update best model:", epoch, best_l_epoch)
        else:
            pre_comp = pre_comp + 1

        printT("epoch: %4s  l=%-10f l_A=%-10f l_dag_adv=%-10f l_dag_com=%-10f "
               "pho_adv=%-.2e pho_com=%-.2e lambda_adv=%-.2e lambda_com=%-.2e"
               % (epoch,
                  l_pre_epoch, l_A_pre_epoch,
                  l_dag_adv_pre_epoch, l_dag_com_pre_epoch,
                  pho_adv, pho_com,
                  lambda_adv, lambda_com,
                  ))

    restore_models(stat_params, args.cache_dir)
    p_A_x_dist.eval()
    A = p_A_x_dist.get_A()

    # G = np.random.rand(len(data_np.columns), len(data_np.columns))
    G = A.cpu().detach().numpy() * (1 - np.eye(d))
    G_pd = pd.DataFrame(G, index=data.columns, columns=data.columns)
    return G_pd, loss_list
