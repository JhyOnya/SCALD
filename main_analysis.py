import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

import pandas as pd
import torch
from torch.distributions import normal
from torch import optim
import importlib

from pynvml import *
from utils_jhy import *

import datasets
import evaluation as eva

from scipy import stats


# set random seeds:
def setup_seed(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_cuda():  # 占满了
    nvmlInit()
    deviceCount = nvmlDeviceGetCount()
    free_list = []

    for i in range(deviceCount):
        handle = nvmlDeviceGetHandleByIndex(i)
        info = nvmlDeviceGetMemoryInfo(handle)
        free_list.append(info.free)
    print("   cuda remain", free_list)
    print("   cuda choose", free_list.index(max(free_list)) + 1)
    return free_list.index(max(free_list))


def TCGA_ls_main(args, res_ls):
    print("Init")
    if torch.cuda.is_available():
        device = torch.device("cuda", get_cuda())
    else:
        print("   cuda is not available")
        device = torch.device("cpu")
    setup_seed()

    print("Read data and networks", args.data, args.cache_dir)
    dataset = getattr(datasets, "TCGA_RPPA")(args.data)
    data_ls = dataset.get_msg()

    networks_ls, spears_ls = [], []
    for i, data_np in enumerate(data_ls):
        print("Stage", i + 1, " sample_shape", data_np.shape)

        pre_network = pd.read_csv("{}S{}_adj_pred.csv".format(args.cache_dir, i + 1), index_col=0)

        networks_ls.append(pre_network)

        spearman_np, p_np = stats.spearmanr(data_np)
        np.fill_diagonal(spearman_np, 0)
        np.fill_diagonal(p_np, 1)
        spears_ls.append(spearman_np)

    analysis_method = importlib.import_module("analysis." + args.method_analysis)
    analysis_method.run(networks_ls, spears_ls, args)


def GSE4183_main(args, res_ls):
    print("Init")
    if torch.cuda.is_available():
        device = torch.device("cuda", get_cuda())
    else:
        print("   cuda is not available")
        device = torch.device("cpu")
    setup_seed()

    print("Read data and networks", args.data, args.cache_dir)
    dataset = eval("datasets.GSE4183")(args.data)
    data_ls, TF_ids_list = dataset.get_msg()

    networks_ls, spears_ls = [], []
    for i, data_np in enumerate(data_ls):
        data_stage = ['colon_normal', 'colon_IBD', 'colon_adenoma', 'colon_CRC', ]
        print(data_stage[i], " sample_shape", data_np.shape)

        pre_network = pd.read_csv("{}S{}_adj_pred.csv".format(args.cache_dir, i + 1), index_col=0)

        networks_ls.append(pre_network)

        spearman_np, p_np = stats.spearmanr(data_np)
        np.fill_diagonal(spearman_np, 0)
        np.fill_diagonal(p_np, 1)
        spears_ls.append(spearman_np)

    analysis_method = importlib.import_module("analysis." + args.method_analysis)
    analysis_method.run(networks_ls, spears_ls, args)


def breast_cancer_main(args, res):
    return


def TCGA_main(args, pred_adj_pd_ori):
    print('Evaluate')
    dataset = getattr(datasets, "TCGA_RPPA")(args.data)
    data, gt_edges_pd = dataset.get_msg()

    # analysis_method = importlib.import_module("analysis." + args.method_analysis)
    # analysis_method.run(pred_adj_pd_ori, gt_edges_pd,args)

    scores = eva.evaluates_tcga(gt_edges_pd, pred_adj_pd_ori)
    draw(X=scores["Rec"], Y=scores["Prec"], x_label="Rec", y_label="Prec", title="PR", dir=args.cache_dir)
    draw(X=scores["FPR"], Y=scores["TPR"], x_label="FPR", y_label="TPR", title="ROC", dir=args.cache_dir)
    if "p_aupr" not in scores:
        return {
            "EPR": scores["EPR"],
            "AUPR": scores["AUPR"],
            "AUROC": scores["AUROC"],
            # "T_EPR": scores_T["EPR"],
            # "T_AUPR": scores_T["AUPR"],
            # "T_AUROC": scores_T["AUROC"],
        }
    else:
        return {
            "EPR": scores["EPR"],
            "AUPR": scores["AUPR"],
            "AUROC": scores["AUROC"],
            "conf_score": scores["conf_score"],
            "p_aupr": scores["p_aupr"],
            "p_auroc": scores["p_auroc"],
            # "T_EPR": scores_T["EPR"],
            # "T_AUPR": scores_T["AUPR"],
            # "T_AUROC": scores_T["AUROC"],
            # "T_conf_score": scores_T["conf_score"],
            # "T_p_aupr": scores_T["p_aupr"],
            # "T_p_auroc": scores_T["p_auroc"],
        }


def Dream5_main(args, pred_adj_pd_ori):
    print('Evaluate')
    dataset = getattr(datasets, args.data)()
    data, gt_edges_pd, gene_ids_pd, TF_ids_list, pdf_auroc, pdf_aupr = dataset.get_msg()
    scores = eva.evaluates_dream5(gt_edges_pd, pred_adj_pd_ori, pdf_auroc, pdf_aupr,
                                  TF_ids_list=TF_ids_list, gene_ids=gene_ids_pd)
    draw(X=scores["Rec"], Y=scores["Prec"], x_label="Rec", y_label="Prec", title="PR", dir=args.cache_dir)
    draw(X=scores["FPR"], Y=scores["TPR"], x_label="FPR", y_label="TPR", title="ROC", dir=args.cache_dir)
    if "p_aupr" not in scores:
        return {
            "EPR": scores["EPR"],
            "AUPR": scores["AUPR"],
            "AUROC": scores["AUROC"],
            # "T_EPR": scores_T["EPR"],
            # "T_AUPR": scores_T["AUPR"],
            # "T_AUROC": scores_T["AUROC"],
        }
    else:
        return {
            "EPR": scores["EPR"],
            "AUPR": scores["AUPR"],
            "AUROC": scores["AUROC"],
            "conf_score": scores["conf_score"],
            "p_aupr": scores["p_aupr"],
            "p_auroc": scores["p_auroc"],
            # "T_EPR": scores_T["EPR"],
            # "T_AUPR": scores_T["AUPR"],
            # "T_AUROC": scores_T["AUROC"],
            # "T_conf_score": scores_T["conf_score"],
            # "T_p_aupr": scores_T["p_aupr"],
            # "T_p_auroc": scores_T["p_auroc"],
        }


def seq_main(args, pred_adj_pd_ori):
    print('Evaluate')
    dataset = getattr(datasets, "seq")(args.data)
    data, gt_edges_pd, gene_ids_pd, TF_ids_list, pdf_auroc, pdf_aupr = dataset.get_msg()
    scores = eva.evaluates_seq(gt_edges_pd, pred_adj_pd_ori, pdf_auroc, pdf_aupr,
                               TF_ids_list=TF_ids_list, gene_ids=gene_ids_pd)
    draw(X=scores["Rec"], Y=scores["Prec"], x_label="Rec", y_label="Prec", title="PR", dir=args.cache_dir)
    draw(X=scores["FPR"], Y=scores["TPR"], x_label="FPR", y_label="TPR", title="ROC", dir=args.cache_dir)
    if "p_aupr" not in scores:
        return {
            "EPR": scores["EPR"],
            "AUPR": scores["AUPR"],
            "AUROC": scores["AUROC"],
            # "T_EPR": scores_T["EPR"],
            # "T_AUPR": scores_T["AUPR"],
            # "T_AUROC": scores_T["AUROC"],
        }
    else:
        return {
            "EPR": scores["EPR"],
            "AUPR": scores["AUPR"],
            "AUROC": scores["AUROC"],
            "conf_score": scores["conf_score"],
            "p_aupr": scores["p_aupr"],
            "p_auroc": scores["p_auroc"],
            # "T_EPR": scores_T["EPR"],
            # "T_AUPR": scores_T["AUPR"],
            # "T_AUROC": scores_T["AUROC"],
            # "T_conf_score": scores_T["conf_score"],
            # "T_p_aupr": scores_T["p_aupr"],
            # "T_p_auroc": scores_T["p_auroc"],
        }


def sachs_main(args, pred_adj_pd_ori):
    print('Evaluate')
    dataset = getattr(datasets, args.data)()
    data, gt_edges_pd, gene_ids_pd, TF_ids_list, pdf_auroc, pdf_aupr = dataset.get_msg()
    scores = eva.evaluates_seq(gt_edges_pd, pred_adj_pd_ori, pdf_auroc, pdf_aupr,
                               TF_ids_list=TF_ids_list, gene_ids=gene_ids_pd)
    draw(X=scores["Rec"], Y=scores["Prec"], x_label="Rec", y_label="Prec", title="PR", dir=args.cache_dir)
    draw(X=scores["FPR"], Y=scores["TPR"], x_label="FPR", y_label="TPR", title="ROC", dir=args.cache_dir)

    pred_edges_pd_ori = graph2edges(pred_adj_pd_ori)
    pred_edges_pd_sorted = sort_edges(pred_edges_pd_ori, sort="descending")
    pred_edges_pd_sorted["pred_weight"]=1

    G_gt = edges2graph(gt_edges_pd, pred_adj_pd_ori.index)
    G_pd = edges2graph(pred_edges_pd_sorted[:20], pred_adj_pd_ori.index)

    cor_pd = G_pd * G_gt
    rev_pd = G_pd * G_gt.T
    ans_gt_pd = np.linalg.matrix_power(G_gt + np.eye(G_pd.shape[0]), G_pd.shape[0])
    ind_pd = (G_pd - cor_pd - rev_pd) * ans_gt_pd
    ind_pd[ind_pd > 0] = 1

    une_pd = G_pd - cor_pd - rev_pd - ind_pd
    une_pd[une_pd > 0] = 1
    une_pd[une_pd < 0] = 0

    num_all = G_pd.sum().sum()
    num_correct = cor_pd.sum().sum()
    num_reverse = rev_pd.sum().sum()
    num_indirect = ind_pd.sum().sum()
    num_unexplain = une_pd.sum().sum()

    print("num_all", num_all, )
    print("num_correct", num_correct, )
    print("num_reverse", num_reverse, )
    print("num_indirect", num_indirect, )
    print("num_unexplain", num_unexplain, )

    if "p_aupr" not in scores:
        return {
            "EPR": scores["EPR"],
            "AUPR": scores["AUPR"],
            "AUROC": scores["AUROC"],
            # "T_EPR": scores_T["EPR"],
            # "T_AUPR": scores_T["AUPR"],
            # "T_AUROC": scores_T["AUROC"],
            "num_all": num_all,
            "num_correct": num_correct,
            "num_reverse": num_reverse,
            "num_indirect": num_indirect,
            "num_unexplain": num_unexplain,
        }
    else:
        return {
            "EPR": scores["EPR"],
            "AUPR": scores["AUPR"],
            "AUROC": scores["AUROC"],
            "conf_score": scores["conf_score"],
            "p_aupr": scores["p_aupr"],
            "p_auroc": scores["p_auroc"],
            # "T_EPR": scores_T["EPR"],
            # "T_AUPR": scores_T["AUPR"],
            # "T_AUROC": scores_T["AUROC"],
            # "T_conf_score": scores_T["conf_score"],
            # "T_p_aupr": scores_T["p_aupr"],
            # "T_p_auroc": scores_T["p_auroc"],
            "num_all": num_all,
            "num_correct": num_correct,
            "num_reverse": num_reverse,
            "num_indirect": num_indirect,
            "num_unexplain": num_unexplain,
        }


def Synthetic_main(args, pred_adj_pd_ori):
    print('Evaluate')
    dataset = getattr(datasets, "synthetic")(args.data)
    data, gt_edges_pd, gene_ids_pd, TF_ids_list, pdf_auroc, pdf_aupr = dataset.get_msg()
    scores = eva.evaluates_synthetic(gt_edges_pd, pred_adj_pd_ori, pdf_auroc, pdf_aupr,
                                     TF_ids_list=TF_ids_list, gene_ids=gene_ids_pd)
    draw(X=scores["Rec"], Y=scores["Prec"], x_label="Rec", y_label="Prec", title="PR", dir=args.cache_dir)
    draw(X=scores["FPR"], Y=scores["TPR"], x_label="FPR", y_label="TPR", title="ROC", dir=args.cache_dir)
    if "p_aupr" not in scores:
        return {
            "EPR": scores["EPR"],
            "AUPR": scores["AUPR"],
            "AUROC": scores["AUROC"],
            # "T_EPR": scores_T["EPR"],
            # "T_AUPR": scores_T["AUPR"],
            # "T_AUROC": scores_T["AUROC"],
        }
    else:
        return {
            "EPR": scores["EPR"],
            "AUPR": scores["AUPR"],
            "AUROC": scores["AUROC"],
            "conf_score": scores["conf_score"],
            "p_aupr": scores["p_aupr"],
            "p_auroc": scores["p_auroc"],
        }


def main(args, pred_adj_pd_ori):
    return eval(args.data_type + "_main")(args, pred_adj_pd_ori)
