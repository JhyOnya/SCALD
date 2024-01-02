import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

import torch
from torch.distributions import normal
from torch import optim
import importlib

from pynvml import *
from utils_jhy import *

import evaluation as eva
import datasets


# set random seeds:
def setup_seed(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # random.seed(seed)
    torch.backends.cudnn.deterministic = True


def common_process(args, data, TF_ids_list, max_comp=100,
                   gene_ids_pd=None, gt_edges_pd=None, pdf_auroc=None, pdf_aupr=None, **kw):
    method = importlib.import_module(args.method)
    # method=eval("networks." + args.method)

    print("Init")
    if torch.cuda.is_available() and not args.notcuda:
        device = torch.device("cuda", 1)
    else:
        print("   cuda is not available")
        device = torch.device("cpu")
    # torch.set_default_dtype(torch.float32)
    setup_seed()

    # gt_edges_pd= gt_edges_pd[gt_edges_pd['gt_weight']==1]
    # gt_adj_pd = edges2graph(gt_edges_pd, gene_ids_pd)

    print("Train")
    pred_adj_pd_ori, loss_list = method.train(data, args, device, max_comp=max_comp,
                                              TF_ids_list=TF_ids_list, gene_ids_pd=gene_ids_pd,
                                              gt_edges_pd=gt_edges_pd, pdf_auroc=pdf_auroc, pdf_aupr=pdf_aupr)

    # pred_adj_pd_ori[pred_adj_pd_ori < args.threshold] = 0  # remove weight<args.threshold

    # pred_edges_pd_ori[['from', 'to']] = pred_edges_pd_ori[['to', 'from']]
    # pred_edges_pd_T = pred_edges_pd_ori[pred_edges_pd_ori['from'].isin(TF_ids_list)]  # remove TG -> xx
    # pred_edges_pd_T_sorted = sort_edges(pred_edges_pd_T, sort="descending")

    return pred_adj_pd_ori, loss_list


def save_res(loss_list, pred_adj_pd_ori, pred_edges_pd_sorted, dir):
    if loss_list is not None:
        draw(X=range(len(loss_list["l_A"])), Y=loss_list["l_A"], title="l_A", dir=dir)
        draw(X=range(len(loss_list["l_dag_adv"])), Y=loss_list["l_dag_adv"], title="l_dag_adv", dir=dir)
        draw(X=range(len(loss_list["l_dag_dec"])), Y=loss_list["l_dag_dec"], title="l_dag_dec", dir=dir)

    pred_adj_pd_ori.to_csv(dir + "adj_pred.csv")
    pred_edges_pd_sorted.to_csv(dir + "pred_edges_pd.csv", index=False)


def main(args):
    def TCGA_ls_main(args):
        print("Read data", args.data)
        dataset = eval("datasets.TCGA_RPPA")(args.data)
        # dataset = getattr(datasets, "TCGA_RPPA")(args.data)

        data_ls = dataset.get_msg()
        for i, pre in enumerate(data_ls):
            print("Stage", i + 1, " sample_shape", pre.shape)

        print("Train")
        pred_adj_pd_ori_ls = []
        max_batch = args.batch
        for i, data in enumerate(data_ls):
            print("")
            print("Stage", i + 1, data.shape)
            if data.shape[0] < args.batch:
                args.batch = data.shape[0]
            else:
                args.batch = max_batch
            TF_ids_list = data.columns.to_list()
            pred_adj_pd_ori, loss_list = common_process(args, data, TF_ids_list=TF_ids_list)
            pred_adj_pd_ori_ls.append(pred_adj_pd_ori)
            pred_edges_pd_ori = graph2edges(pred_adj_pd_ori)
            pred_edges_pd = pred_edges_pd_ori[pred_edges_pd_ori['from'].isin(TF_ids_list)]  # remove TG -> xx
            pred_edges_pd_sorted = sort_edges(pred_edges_pd, sort="descending")

            print('Save')
            save_res(loss_list, pred_adj_pd_ori, pred_edges_pd_sorted, args.cache_dir + "S" + str(i + 1) + "_")

        # scores = eva.evaluates_tcga(gt_edges_pd, pred_adj_pd_ori, pdf_auroc, pdf_aupr,
        #                        TF_ids_list=TF_ids_list, gene_ids=gene_ids_pd)
        return pred_adj_pd_ori_ls

    def GSE4183_main(args):
        print("Read data", args.data)
        dataset = eval("datasets.GSE4183")(args.data)
        # dataset = getattr(datasets, "TCGA_RPPA")(args.data)

        data_ls, TF_ids_list = dataset.get_msg()
        for i, pre in enumerate(data_ls):
            print("Stage", i + 1, " sample_shape", pre.shape)

        print("Train")
        data_stage = ['colon_normal', 'colon_IBD', 'colon_adenoma', 'colon_CRC', ]
        pred_adj_pd_ori_ls = []
        for i, data in enumerate(data_ls):
            print("")
            print(data_stage[i], data.shape)
            if data.shape[0] < args.batch:
                args.batch = data.shape[0]
            pred_adj_pd_ori, loss_list = common_process(args, data, TF_ids_list=TF_ids_list, max_comp=10000)
            pred_adj_pd_ori_ls.append(pred_adj_pd_ori)
            pred_edges_pd_ori = graph2edges(pred_adj_pd_ori)
            pred_edges_pd = pred_edges_pd_ori[pred_edges_pd_ori['from'].isin(TF_ids_list)]  # remove TG -> xx
            pred_edges_pd_sorted = sort_edges(pred_edges_pd, sort="descending")

            print('Save')
            save_res(loss_list, pred_adj_pd_ori, pred_edges_pd_sorted, args.cache_dir + "S" + str(i + 1) + "_")

        # scores = eva.evaluates_tcga(gt_edges_pd, pred_adj_pd_ori, pdf_auroc, pdf_aupr,
        #                        TF_ids_list=TF_ids_list, gene_ids=gene_ids_pd)
        return pred_adj_pd_ori_ls

    def TCGA_main(args):
        print("Read data", args.data)
        dataset = getattr(datasets, "TCGA_RPPA")(args.data)
        data, gt_edges_pd = dataset.get_msg()
        TF_ids_list = data.columns.to_list()

        print("Train")
        pred_adj_pd_ori, loss_list = common_process(args, data, TF_ids_list=TF_ids_list)
        pred_edges_pd_ori = graph2edges(pred_adj_pd_ori)
        pred_edges_pd = pred_edges_pd_ori[pred_edges_pd_ori['from'].isin(TF_ids_list)]  # remove TG -> xx
        pred_edges_pd_sorted = sort_edges(pred_edges_pd, sort="descending")

        # print('  pred_edges_pd.T')
        # scores_T = eva.evaluates(gt_edges_pd, pred_edges_pd_T_sorted, pdf_auroc, pdf_aupr,
        #                          TF_ids_list=TF_ids_list, gene_ids=gene_ids_pd)

        print('Save')
        save_res(loss_list, pred_adj_pd_ori, pred_edges_pd_sorted, args.cache_dir)

        # edges_compare = pd.merge(pred_edges_pd_sorted, gt_edges_pd, how="outer")
        # edges_compare.to_csv(args.cache_dir + "edges_compare_outer.csv", index=False)
        # edges_compare = pd.merge(pred_edges_pd_sorted, gt_edges_pd, how="inner")
        # edges_compare.to_csv(args.cache_dir + "edges_compare_inner.csv", index=False)
        return pred_adj_pd_ori

    def sachs_main(args):
        print("Read data", args.data)
        dataset = getattr(datasets, args.data)()
        data, gt_edges_pd, gene_ids_pd, TF_ids_list, pdf_auroc, pdf_aupr = dataset.get_msg()

        print("Train")
        pred_adj_pd_ori, loss_list = common_process(args, data, TF_ids_list=TF_ids_list)
        pred_edges_pd_ori = graph2edges(pred_adj_pd_ori)
        pred_edges_pd = pred_edges_pd_ori[pred_edges_pd_ori['from'].isin(TF_ids_list)]  # remove TG -> xx
        pred_edges_pd_sorted = sort_edges(pred_edges_pd, sort="descending")

        pred_adj_pd_ori_half = pred_adj_pd_ori.copy()
        pred_adj_pd_ori_half[pred_adj_pd_ori_half < pred_adj_pd_ori_half.T] = 0
        pred_edges_pd_ori_half = graph2edges(pred_adj_pd_ori_half)
        pred_edges_pd_half = pred_edges_pd_ori_half[pred_edges_pd_ori_half['from'].isin(TF_ids_list)]  # remove TG -> xx
        pred_edges_pd_sorted_half = sort_edges(pred_edges_pd_half, sort="descending")

        # print('  pred_edges_pd.T')
        # scores_T = eva.evaluates(gt_edges_pd, pred_edges_pd_T_sorted, pdf_auroc, pdf_aupr,
        #                          TF_ids_list=TF_ids_list, gene_ids=gene_ids_pd)

        print('Save')
        save_res(loss_list, pred_adj_pd_ori, pred_edges_pd_sorted, args.cache_dir)

        edges_compare = pd.merge(pred_edges_pd_sorted, gt_edges_pd, how="outer")
        edges_compare.to_csv(args.cache_dir + "edges_compare_outer.csv", index=False)
        edges_compare = pd.merge(pred_edges_pd_sorted, gt_edges_pd, how="inner")
        edges_compare.to_csv(args.cache_dir + "edges_compare_inner.csv", index=False)

        edges_compare_half = pd.merge(pred_edges_pd_sorted_half, gt_edges_pd, how="outer")
        edges_compare_half.to_csv(args.cache_dir + "edges_compare_half_outer.csv", index=False)
        edges_compare_half = pd.merge(pred_edges_pd_sorted_half, gt_edges_pd, how="inner")
        edges_compare_half.to_csv(args.cache_dir + "edges_compare_half_inner.csv", index=False)

        # pred_edges_pd_T_sorted.to_csv(args.cache_dir + "pred_edges_pd_T.csv", index=False)
        # edges_compare_T = pd.merge(pred_edges_pd_T_sorted, gt_edges_pd, how="outer")
        # edges_compare_T.to_csv(args.cache_dir + "edges_compare_T_outer.csv", index=False)
        # edges_compare_T = pd.merge(pred_edges_pd_T_sorted, gt_edges_pd, how="inner")
        # edges_compare_T.to_csv(args.cache_dir + "edges_compare_T_inner.csv", index=False)

        # edges2graph(gt_edges_pd, gene_ids_pd).to_csv(args.cache_dir + "adj_gt.csv")

        # pred_edges_pd_sorted \
        #     .to_csv(args.cache_dir + "pred_edges_pd.csv", index=False)
        # pd.merge(pred_edges_pd_sorted, gt_edges_pd, how="inner") \
        #     .to_csv(args.cache_dir + "edges_compare_inner.csv", index=False)
        # pd.merge(pred_edges_pd_sorted, gt_edges_pd, how="outer") \
        #     .to_csv(args.cache_dir + "edges_compare_outer.csv", index=False)
        #
        # pred_edges_pd_T_sorted \
        #     .to_csv(args.cache_dir + "pred_edges_pd_T.csv", index=False)
        # pd.merge(pred_edges_pd_T_sorted, gt_edges_pd, how="inner") \
        #     .to_csv(args.cache_dir + "edges_compare_T_inner.csv", index=False)
        # pd.merge(pred_edges_pd_T_sorted, gt_edgx`es_pd, how="outer") \
        #     .to_csv(args.cache_dir + "edges_compare_T_outer.csv", index=False)
        return pred_adj_pd_ori

    def Dream5_main(args):
        print("Read data", args.data)
        dataset = getattr(datasets, args.data)()
        data, gt_edges_pd, gene_ids_pd, TF_ids_list, pdf_auroc, pdf_aupr = dataset.get_msg()

        print("Train")
        pred_adj_pd_ori, loss_list = common_process(args, data, TF_ids_list=TF_ids_list)
        pred_edges_pd_ori = graph2edges(pred_adj_pd_ori)
        pred_edges_pd = pred_edges_pd_ori[pred_edges_pd_ori['from'].isin(TF_ids_list)]  # remove TG -> xx
        pred_edges_pd_sorted = sort_edges(pred_edges_pd, sort="descending")

        # print('  pred_edges_pd.T')
        # scores_T = eva.evaluates(gt_edges_pd, pred_edges_pd_T_sorted, pdf_auroc, pdf_aupr,
        #                          TF_ids_list=TF_ids_list, gene_ids=gene_ids_pd)

        print('Save')
        save_res(loss_list, pred_adj_pd_ori, pred_edges_pd_sorted, args.cache_dir)

        edges_compare = pd.merge(pred_edges_pd_sorted, gt_edges_pd, how="outer")
        edges_compare.to_csv(args.cache_dir + "edges_compare_outer.csv", index=False)
        edges_compare = pd.merge(pred_edges_pd_sorted, gt_edges_pd, how="inner")
        edges_compare.to_csv(args.cache_dir + "edges_compare_inner.csv", index=False)

        # pred_edges_pd_T_sorted.to_csv(args.cache_dir + "pred_edges_pd_T.csv", index=False)
        # edges_compare_T = pd.merge(pred_edges_pd_T_sorted, gt_edges_pd, how="outer")
        # edges_compare_T.to_csv(args.cache_dir + "edges_compare_T_outer.csv", index=False)
        # edges_compare_T = pd.merge(pred_edges_pd_T_sorted, gt_edges_pd, how="inner")
        # edges_compare_T.to_csv(args.cache_dir + "edges_compare_T_inner.csv", index=False)

        # edges2graph(gt_edges_pd, gene_ids_pd).to_csv(args.cache_dir + "adj_gt.csv")

        # pred_edges_pd_sorted \
        #     .to_csv(args.cache_dir + "pred_edges_pd.csv", index=False)
        # pd.merge(pred_edges_pd_sorted, gt_edges_pd, how="inner") \
        #     .to_csv(args.cache_dir + "edges_compare_inner.csv", index=False)
        # pd.merge(pred_edges_pd_sorted, gt_edges_pd, how="outer") \
        #     .to_csv(args.cache_dir + "edges_compare_outer.csv", index=False)
        #
        # pred_edges_pd_T_sorted \
        #     .to_csv(args.cache_dir + "pred_edges_pd_T.csv", index=False)
        # pd.merge(pred_edges_pd_T_sorted, gt_edges_pd, how="inner") \
        #     .to_csv(args.cache_dir + "edges_compare_T_inner.csv", index=False)
        # pd.merge(pred_edges_pd_T_sorted, gt_edgx`es_pd, how="outer") \
        #     .to_csv(args.cache_dir + "edges_compare_T_outer.csv", index=False)
        return pred_adj_pd_ori

    def seq_main(args):
        print("Read data", args.data)
        dataset = getattr(datasets, "seq")(args.data)
        data, gt_edges_pd, gene_ids_pd, TF_ids_list, pdf_auroc, pdf_aupr = dataset.get_msg()

        print("Train")
        pred_adj_pd_ori, loss_list = common_process(args, data, TF_ids_list=TF_ids_list)
        pred_edges_pd_ori = graph2edges(pred_adj_pd_ori)
        pred_edges_pd = pred_edges_pd_ori[pred_edges_pd_ori['from'].isin(TF_ids_list)]  # remove TG -> xx
        pred_edges_pd_sorted = sort_edges(pred_edges_pd, sort="descending")

        # print('  pred_edges_pd.T')
        # scores_T = eva.evaluates(gt_edges_pd, pred_edges_pd_T_sorted, pdf_auroc, pdf_aupr,
        #                          TF_ids_list=TF_ids_list, gene_ids=gene_ids_pd)

        print('Save')
        save_res(loss_list, pred_adj_pd_ori, pred_edges_pd_sorted, args.cache_dir)

        edges_compare = pd.merge(pred_edges_pd_sorted, gt_edges_pd, how="outer")
        edges_compare.to_csv(args.cache_dir + "edges_compare_outer.csv", index=False)
        edges_compare = pd.merge(pred_edges_pd_sorted, gt_edges_pd, how="inner")
        edges_compare.to_csv(args.cache_dir + "edges_compare_inner.csv", index=False)

        return pred_adj_pd_ori

    def Synthetic_main(args):
        print("Read data", args.data)
        dataset = getattr(datasets, "synthetic")(args.data)
        data, gt_edges_pd, gene_ids_pd, TF_ids_list, pdf_auroc, pdf_aupr = dataset.get_msg()

        print("Train")
        pred_adj_pd_ori, loss_list = common_process(args, data, TF_ids_list=TF_ids_list)
        pred_edges_pd_ori = graph2edges(pred_adj_pd_ori)
        pred_edges_pd = pred_edges_pd_ori[pred_edges_pd_ori['from'].isin(TF_ids_list)]  # remove TG -> xx
        pred_edges_pd_sorted = sort_edges(pred_edges_pd, sort="descending")

        # print('  pred_edges_pd.T')
        # scores_T = eva.evaluates(gt_edges_pd, pred_edges_pd_T_sorted, pdf_auroc, pdf_aupr,
        #                          TF_ids_list=TF_ids_list, gene_ids=gene_ids_pd)

        print('Save')
        save_res(loss_list, pred_adj_pd_ori, pred_edges_pd_sorted, args.cache_dir)

        edges_compare = pd.merge(pred_edges_pd_sorted, gt_edges_pd, how="outer")
        edges_compare.to_csv(args.cache_dir + "edges_compare_outer.csv", index=False)
        edges_compare = pd.merge(pred_edges_pd_sorted, gt_edges_pd, how="inner")
        edges_compare.to_csv(args.cache_dir + "edges_compare_inner.csv", index=False)

        return pred_adj_pd_ori

    def breast_cancer_main(args):
        print("Read data", args.data)
        data_class = eval("datasets.breast_cancer")(args.data)
        dataset_ori, genes_ = data_class.get_msg()
        genes_ori = genes_.index.tolist()

        # meanPre = dataset_ori.mean(0)  # mean
        # varPre = dataset_ori.var(0)
        # remainindex = varPre[varPre > 0.1].index
        # remain_ = list(set(remainindex) & set(genes_ori))
        # data = dataset_ori[remain_]
        # print((set(dataset_ori.columns) & set(genes_ori)) -set(remain_) )
        # print(dataset_ori["NLGN4Y"])

        data = dataset_ori[list(set(dataset_ori.columns) & set(genes_ori))].dropna(axis=1)

        print("shape", data.shape)

        print("Train")
        if data.shape[0] < args.batch:
            args.batch = data.shape[0]
        pred_adj_pd_ori, loss_list = common_process(args, data, TF_ids_list=data.columns, max_comp=10000)
        pred_edges_pd_ori = graph2edges(pred_adj_pd_ori)
        pred_edges_pd_sorted = sort_edges(pred_edges_pd_ori, sort="descending")

        print('Save')
        save_res(loss_list, pred_adj_pd_ori, pred_edges_pd_sorted, args.cache_dir)

        # scores = eva.evaluates_tcga(gt_edges_pd, pred_adj_pd_ori, pdf_auroc, pdf_aupr,
        #                        TF_ids_list=TF_ids_list, gene_ids=gene_ids_pd)
        return pred_adj_pd_ori

    return eval(args.data_type + "_main")(args)
    # if args.data.startswith("TCGA"):
    #     return tcga_main_ls(args)
    # elif args.data.startswith("Dream5"):
    #     return dream5_main(args)
    # elif args.data.startswith("Synthetic"):
    #     return synthetic_main(args)
    # elif args.data.startswith("mDC") or args.data.startswith("hESC"):
    #     return seq_main(args)
    # else:
    #     return None
