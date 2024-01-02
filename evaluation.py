import numpy as np
import torch
from scipy.stats import sem
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
from utils_jhy import *
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import scipy.io as scio
import time


# def plt_ROC_curve( ground_truth_known, pred_known, path):
#     fpr, tpr, thresholds = metrics.roc_curve(ground_truth_known, pred_known)
#     roc_auc = metrics.auc(fpr, tpr)
#     display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
#                                       estimator_name='example estimator')
#     display.plot()
#     plt.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)
#     # plt.show()
#     plt.savefig(path)
#     return

# def eva_method_AUC( ground_truth_known, pred_known):
#     # TODO
#     fpr, tpr, thresholds = metrics.roc_curve(ground_truth_known, pred_known)
#     roc_auc = metrics.auc(fpr, tpr)
#     return roc_auc
#
# def eva_method_F1( ground_truth_known, pred_known):
#     # TODO
#     return 0

# def draw( ground_truth_known, pred_known, cache_dir):
#     plt_ROC_curve(ground_truth_known, pred_known, cache_dir + "ROC_curve.pdf")

# def tp(gt_edges_pd, pred_edges_pd):
#     gt_edges_pd_1 = gt_edges_pd[gt_edges_pd['gt_weight'] == 1]
#     tp_k_pd = pd.merge(gt_edges_pd_1, pred_edges_pd, on=['from', 'to'], how='inner')
#     return tp_k_pd.shape[0]

# def eva_method_AUROC(ground_truth_known, pred_known):
#     # TODO
#     return 0
#
#
# def eva_method_AUPR(ground_truth_known, pred_known):
#     # TODO
#     return 0

#
# def eva_method_CS(ground_truth_known, pred_known):
#     # TODO
#     return 0


def eva_method_EPR(ground_truth_known, pred_known):
    def EP(gt, pred):
        k = gt.shape[0]
        pred = pred.iloc[:k, :]
        # gs_set = set(zip(gt['from'], gt['to']))
        # pred_set = set(zip(pred['from'], pred['to']))
        # edges_pd = pd.merge(pred, gt, how="inner")
        l = pd.merge(pred, gt, how="inner").shape[0]
        return l / k

    gs_pos = ground_truth_known[ground_truth_known['gt_weight'] == 1]
    ep = EP(gs_pos, pred_known)

    random_ep = []
    for i in range(100):
        random_net = shuffle(pred_known)
        random_ep.append(EP(gs_pos, random_net))
    return ep / np.mean(random_ep)


def eva_method_AUROC_AUPR(edges_pd):
    results = dict()

    P = (edges_pd['gt_weight'] == 1).sum()
    N = (edges_pd['gt_weight'] == 0).sum()

    K = np.arange(1, (edges_pd.shape[0] + 1))
    tp_k = np.cumsum(edges_pd['gt_weight'].values)
    fp_k = np.cumsum(1 - edges_pd['gt_weight'].values)

    # PR
    results["Prec"] = tp_k / K  # 曲线竖轴
    results["Rec"] = tp_k / P  # 曲线横轴
    # results["AUPR"] = np.trapz(results["Prec"], results["Rec"])
    results["AUPR"] = np.trapz(results["Prec"], results["Rec"]) / (1 - 1 / P)  # normalized by max possible value

    # ROC
    results["TPR"] = tp_k / P  # 曲线竖轴
    results["FPR"] = fp_k / N  # 曲线横轴
    results["AUROC"] = np.trapz(results["TPR"], results["FPR"])

    # fpr, tpr, thresholds = roc_curve(edges_pd['gt_weight'], edges_pd['pred_weight'])
    # print(auc(fpr, tpr))
    # precision, recall, thresholds = precision_recall_curve(edges_pd['gt_weight'], edges_pd['pred_weight'])
    # print(auc(recall, precision))

    return results


def probability(X, Y, x):
    X = X.squeeze(0)
    Y = Y.squeeze(0)
    tmp = (X >= x)
    P = np.sum(tmp * Y) / len(X)
    return P


def evaluates_tcga(gt_edges_pd, pred_adj_pd_ori, pdf_auroc=None, pdf_aupr=None, TF_ids_list=None, gene_ids=None):
    res_score = dict()
    pred_edges_pd_ori = graph2edges(fill_zeros_neg(pred_adj_pd_ori))
    if TF_ids_list is None:
        TF_ids_list = gt_edges_pd.index.tolist()
    pred_edges_pd = pred_edges_pd_ori[pred_edges_pd_ori['from'].isin(TF_ids_list)]  # remove TG -> xx
    pred_edges_pd_sorted = sort_edges(pred_edges_pd, sort="descending")

    if (gt_edges_pd["gt_weight"] == 0).any():  # gt是否存在0
        edges_pd = pd.merge(pred_edges_pd_sorted, gt_edges_pd, how="inner")
    else:  # 只提供了有边的信息
        edges_pd = pd.merge(pred_edges_pd_sorted, gt_edges_pd, how="outer")
        edges_pd = edges_pd.fillna(0)

    # res_au_rdm = eva_method_AUROC_AUPR(edges_pd.sample(frac=1).reset_index(drop=True))
    res_au = eva_method_AUROC_AUPR(edges_pd)
    res_score["EPR"] = eva_method_EPR(gt_edges_pd, pred_edges_pd_sorted)

    print("          epr:  %15f" % res_score["EPR"])
    print("         AUPR:  %15f" % res_au["AUPR"])
    print("        AUROC:  %15f" % res_au["AUROC"])

    if pdf_aupr is None:
        return dict(res_au, **res_score)

    # conf_score
    # res_score['p_aupr'] = 1
    res_score['p_aupr'] = probability(pdf_aupr['X'], pdf_aupr['Y'], res_au['AUPR'])
    res_score['aupr_score'] = 100 if (res_score['p_aupr'] == 0 or -np.log10(res_score['p_aupr']) > 100) else -np.log10(
        res_score['p_aupr'])

    # res_score['p_auroc'] = 1
    res_score['p_auroc'] = probability(pdf_auroc['X'], pdf_auroc['Y'], res_au['AUROC'])
    res_score['auroc_score'] = 100 if (
            res_score['p_auroc'] == 0 or -np.log10(res_score['p_auroc']) > 100) else -np.log10(
        res_score['p_auroc'])

    # conf_score = (aupr_score + auroc_score) / 2 #  TODO 冯轲版
    res_score['conf_score'] = (res_score['aupr_score'] + res_score['auroc_score']) / 2

    # print("EPR:", res_score["EPR"])
    print("   conf_score:  %15f" % res_score["conf_score"])
    print("       p_aupr:  %15f" % res_score["p_aupr"])
    print("      p_auroc:  %15f" % res_score["p_auroc"])
    print("   aupr_score:  %15f" % res_score['aupr_score'])
    print("  auroc_score:  %15f" % res_score['auroc_score'])

    return dict(res_au, **res_score)


def evaluates_seq(gt_edges_pd, pred_adj_pd_ori, pdf_auroc, pdf_aupr, TF_ids_list, gene_ids):
    res_score = dict()
    # pred_edges_pd_ori = graph2edges(fill_zeros_neg(pred_adj_pd_ori))
    pred_edges_pd_ori = graph2edges(pred_adj_pd_ori)
    pred_edges_pd = pred_edges_pd_ori[pred_edges_pd_ori['from'].isin(TF_ids_list)]  # remove TG -> xx
    pred_edges_pd_sorted = sort_edges(pred_edges_pd, sort="descending")

    if (gt_edges_pd["gt_weight"] == 0).any():  # gt是否存在0
        edges_pd = pd.merge(pred_edges_pd_sorted, gt_edges_pd, how="inner")
    else:  # 只提供了有边的信息
        edges_pd = pd.merge(pred_edges_pd_sorted, gt_edges_pd, how="outer")
        edges_pd = edges_pd.fillna(0)

    # res_au_rdm = eva_method_AUROC_AUPR(edges_pd.sample(frac=1).reset_index(drop=True))
    res_au = eva_method_AUROC_AUPR(edges_pd)
    res_score["EPR"] = eva_method_EPR(gt_edges_pd, pred_edges_pd_sorted)

    print("          epr:  %15f" % res_score["EPR"])
    print("         AUPR:  %15f" % res_au["AUPR"])
    print("        AUROC:  %15f" % res_au["AUROC"])

    if pdf_aupr is None:
        return dict(res_au, **res_score)

    # conf_score
    # res_score['p_aupr'] = 1
    res_score['p_aupr'] = probability(pdf_aupr['X'], pdf_aupr['Y'], res_au['AUPR'])
    res_score['aupr_score'] = 100 if (res_score['p_aupr'] == 0 or -np.log10(res_score['p_aupr']) > 100) else -np.log10(
        res_score['p_aupr'])

    # res_score['p_auroc'] = 1
    res_score['p_auroc'] = probability(pdf_auroc['X'], pdf_auroc['Y'], res_au['AUROC'])
    res_score['auroc_score'] = 100 if (
            res_score['p_auroc'] == 0 or -np.log10(res_score['p_auroc']) > 100) else -np.log10(
        res_score['p_auroc'])

    # conf_score = (aupr_score + auroc_score) / 2 #  TODO 冯轲版
    res_score['conf_score'] = (res_score['aupr_score'] + res_score['auroc_score']) / 2

    # print("EPR:", res_score["EPR"])
    print("   conf_score:  %15f" % res_score["conf_score"])
    print("       p_aupr:  %15f" % res_score["p_aupr"])
    print("      p_auroc:  %15f" % res_score["p_auroc"])
    print("   aupr_score:  %15f" % res_score['aupr_score'])
    print("  auroc_score:  %15f" % res_score['auroc_score'])

    return dict(res_au, **res_score)


def evaluates_synthetic(gt_edges_pd, pred_adj_pd_ori, pdf_auroc, pdf_aupr, TF_ids_list, gene_ids):
    res_score = dict()
    # pred_edges_pd_ori = graph2edges(fill_zeros_neg(pred_adj_pd_ori))
    pred_edges_pd_ori = graph2edges(pred_adj_pd_ori)
    pred_edges_pd = pred_edges_pd_ori[pred_edges_pd_ori['from'].isin(TF_ids_list)]  # remove TG -> xx
    pred_edges_pd_sorted = sort_edges(pred_edges_pd, sort="descending")

    if (gt_edges_pd["gt_weight"] == 0).any():  # gt是否存在0
        edges_pd = pd.merge(pred_edges_pd_sorted, gt_edges_pd, how="inner")
    else:  # 只提供了有边的信息
        edges_pd = pd.merge(pred_edges_pd_sorted, gt_edges_pd, how="outer")
        edges_pd = edges_pd.fillna(0)

    # res_au_rdm = eva_method_AUROC_AUPR(edges_pd.sample(frac=1).reset_index(drop=True))
    res_au = eva_method_AUROC_AUPR(edges_pd)
    res_score["EPR"] = eva_method_EPR(gt_edges_pd, pred_edges_pd_sorted)

    print("          epr:  %15f" % res_score["EPR"])
    print("         AUPR:  %15f" % res_au["AUPR"])
    print("        AUROC:  %15f" % res_au["AUROC"])

    if pdf_aupr is None:
        return dict(res_au, **res_score)

    # conf_score
    # res_score['p_aupr'] = 1
    res_score['p_aupr'] = probability(pdf_aupr['X'], pdf_aupr['Y'], res_au['AUPR'])
    res_score['aupr_score'] = 100 if (res_score['p_aupr'] == 0 or -np.log10(res_score['p_aupr']) > 100) else -np.log10(
        res_score['p_aupr'])

    # res_score['p_auroc'] = 1
    res_score['p_auroc'] = probability(pdf_auroc['X'], pdf_auroc['Y'], res_au['AUROC'])
    res_score['auroc_score'] = 100 if (
            res_score['p_auroc'] == 0 or -np.log10(res_score['p_auroc']) > 100) else -np.log10(
        res_score['p_auroc'])

    # conf_score = (aupr_score + auroc_score) / 2 #  TODO 冯轲版
    res_score['conf_score'] = (res_score['aupr_score'] + res_score['auroc_score']) / 2

    # print("EPR:", res_score["EPR"])
    print("   conf_score:  %15f" % res_score["conf_score"])
    print("       p_aupr:  %15f" % res_score["p_aupr"])
    print("      p_auroc:  %15f" % res_score["p_auroc"])
    print("   aupr_score:  %15f" % res_score['aupr_score'])
    print("  auroc_score:  %15f" % res_score['auroc_score'])

    return dict(res_au, **res_score)

def evaluates_dream5(gt_edges_pd, pred_adj_pd_ori, pdf_auroc, pdf_aupr, TF_ids_list, gene_ids):
    res_score = dict()
    pred_edges_pd_ori = graph2edges(fill_zeros_neg(pred_adj_pd_ori))
    pred_edges_pd = pred_edges_pd_ori[pred_edges_pd_ori['from'].isin(TF_ids_list)]  # remove TG -> xx
    pred_edges_pd_sorted = sort_edges(pred_edges_pd, sort="descending")

    if (gt_edges_pd["gt_weight"] == 0).any():  # gt是否存在0
        edges_pd = pd.merge(pred_edges_pd_sorted, gt_edges_pd, how="inner")
    else:  # 只提供了有边的信息
        edges_pd = pd.merge(pred_edges_pd_sorted, gt_edges_pd, how="outer")
        edges_pd = edges_pd.fillna(0)

    # res_au_rdm = eva_method_AUROC_AUPR(edges_pd.sample(frac=1).reset_index(drop=True))
    res_au = eva_method_AUROC_AUPR(edges_pd)
    res_score["EPR"] = eva_method_EPR(gt_edges_pd, pred_edges_pd_sorted)

    print("          epr:  %15f" % res_score["EPR"])
    print("         AUPR:  %15f" % res_au["AUPR"])
    print("        AUROC:  %15f" % res_au["AUROC"])

    if pdf_aupr is None:
        return dict(res_au, **res_score)

    # conf_score
    # res_score['p_aupr'] = 1
    res_score['p_aupr'] = probability(pdf_aupr['X'], pdf_aupr['Y'], res_au['AUPR'])
    res_score['aupr_score'] = 100 if (res_score['p_aupr'] == 0 or -np.log10(res_score['p_aupr']) > 100) else -np.log10(
        res_score['p_aupr'])

    # res_score['p_auroc'] = 1
    res_score['p_auroc'] = probability(pdf_auroc['X'], pdf_auroc['Y'], res_au['AUROC'])
    res_score['auroc_score'] = 100 if (
            res_score['p_auroc'] == 0 or -np.log10(res_score['p_auroc']) > 100) else -np.log10(
        res_score['p_auroc'])

    # conf_score = (aupr_score + auroc_score) / 2 #  TODO 冯轲版
    res_score['conf_score'] = (res_score['aupr_score'] + res_score['auroc_score']) / 2

    # print("EPR:", res_score["EPR"])
    print("   conf_score:  %15f" % res_score["conf_score"])
    print("       p_aupr:  %15f" % res_score["p_aupr"])
    print("      p_auroc:  %15f" % res_score["p_auroc"])
    print("   aupr_score:  %15f" % res_score['aupr_score'])
    print("  auroc_score:  %15f" % res_score['auroc_score'])

    return dict(res_au, **res_score)

    # # for rate in range(100):
    # #     k = int(((rate + 1) / 100) * pred_edges_pd_sorted.shape[0])
    # # pred_edges_pd_sorted[(pred_edges_pd_sorted['from']=="G29") & (pred_edges_pd_sorted['to']=="G1305")]
    # for k in range(pred_edges_pd_sorted.shape[0]):
    #     if pred_edges_pd_sorted[k, 'gt_weight'] == 1:
    #         if k == 0:
    #             results["TPR"].append(1)
    #             results["FPR"].append(0)
    #             results["A"].append(1 / P)
    #         else:
    #             results["TPR"].append(results["TPR"][-1] + 1)
    #             results["FPR"].append(results["FPR"][-1])
    #             results["A"].append(results["A"][-1] + 1 / P - results["FPR"][-1] * np.log(k + 1 / k) / P)
    #     else:
    #         results["TPR"].append(results["TPR"][-1])
    #         results["FPR"].append(results["FPR"][-1] + 1)
    #         results["A"].append(results["A"][-1])
    #
    # tp_k = tp(gt_edges_pd, pred_edges_pd_sorted_k)
    # fp_k = k - tp_k
    #
    # # PR
    # prec_k = tp_k / k
    # rec_k = tp_k / p
    #
    # # ROC
    # tpr_k = tp_k / p
    # fpr_k = fp_k / n
    #
    # results["k"].append(k)
    # results["AUROC"].append(eva_method_AUROC(gt_edges_pd, pred_edges_pd_sorted_k))
    # results["AUPR"].append(eva_method_AUPR(gt_edges_pd, pred_edges_pd_sorted_k))
    # results["Confidence Score"].append(eva_method_CS(gt_edges_pd, pred_edges_pd_sorted_k))
    # results["EPR"].append(eva_method_EPR(gt_edges_pd, pred_edges_pd_sorted_k))

    # if cache_dir is not None:
    #     draw(ground_truth_known, pred_known, cache_dir)

    # if p_auroc == 0:
    #     auroc_score = 100
    # else:
    #     auroc_score = 100 if -np.log10(p_auroc) > 100 else -np.log10(p_auroc)


if __name__ == "__main__":
    import scipy.io as sio
    import random

    # train_paths = ["../datasets/dream5/training data/network1-in silico/net1_",
    #                "../datasets/dream5/training data/network2-s.aureus/net2_",
    #                "../datasets/dream5/training data/network3-e.coil/net3_",
    #                "../datasets/dream5/training data/network4-s.cerevisiae/net4_", ]
    # test_paths = ["../datasets/dream5/test data/DREAM5_NetworkInference_GoldStandard_Network1 - in silico.tsv",
    #               "../datasets/dream5/test data/DREAM5_NetworkInference_GoldStandard_Network2 - S. aureus.tsv",
    #               "../datasets/dream5/test data/DREAM5_NetworkInference_GoldStandard_Network3 - E. coli.tsv",
    #               "../datasets/dream5/test data/DREAM5_NetworkInference_GoldStandard_Network4 - S. cerevisiae.tsv", ]

    path_adj_prediction = "/home/jianghongyang/Code/Python/2022_CD/cache/e_coil/mine.method282_01/adj_pred.csv"
    path_prediction = "/home/jianghongyang/Code/Python/2022_CD/cache/e_coil/mine.method297_01/pred_edges_pd.csv"
    prob_dens_path = "../datasets/dream5/Evaluation_scripts/INPUT/probability_densities/"
    path_gold_standard = "../datasets/dream5/test data/DREAM5_NetworkInference_GoldStandard_Network3 - E. coli.tsv"
    path_pdf_aupr = "../datasets/dream5/Evaluation_scripts/INPUT/probability_densities/Network3_AUPR.mat"
    path_pdf_auroc = "../datasets/dream5/Evaluation_scripts/INPUT/probability_densities/Network3_AUROC.mat"
    gene_ids = pd.read_csv("../datasets/dream5/training data/network3-e.coil/net3_gene_ids.tsv", sep="\t")
    TF_ids = pd.read_csv("../datasets/dream5/training data/network3-e.coil/net3_transcription_factors.tsv", sep="\t",
                         header=None)

    # path_adj_prediction = "/home/jianghongyang/Code/Python/2022_CD/cache/s_cerevisiae/mine.method282_01/adj_pred.csv"
    # path_prediction = "/home/jianghongyang/Code/Python/2022_CD/cache/s_cerevisiae/mine.method297_01/pred_edges_pd.csv"
    # prob_dens_path = "../datasets/dream5/Evaluation_scripts/INPUT/probability_densities/"
    # path_gold_standard = "../datasets/dream5/test data/DREAM5_NetworkInference_GoldStandard_Network4 - S. cerevisiae.tsv"
    # path_pdf_aupr = "../datasets/dream5/Evaluation_scripts/INPUT/probability_densities/Network4_AUPR.mat"
    # path_pdf_auroc = "../datasets/dream5/Evaluation_scripts/INPUT/probability_densities/Network4_AUROC.mat"
    # gene_ids = pd.read_csv("../datasets/dream5/training data/network4-s.cerevisiae/net4_gene_ids.tsv", sep="\t")
    # TF_ids = pd.read_csv("../datasets/dream5/training data/network4-s.cerevisiae/net4_transcription_factors.tsv",
    #                      sep="\t",
    #                      header=None)

    TF_ids_list = TF_ids[0].to_list()


    def setup_seed(seed=1):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        # random.seed(seed)
        torch.backends.cudnn.deterministic = True


    setup_seed()

    # adj_prediction = pd.read_csv(path_adj_prediction, index_col=0)
    # adj_prediction[adj_prediction - adj_prediction.T < 0] = 0
    # graph = adj_prediction.values
    # edges_a, edges_b = np.where(graph != 0)
    # edges_list = [graph[f, t] for (f, t) in zip(edges_a, edges_b)]
    # prediction = pd.DataFrame({"from": adj_prediction.index[edges_a],
    #                            "to": adj_prediction.index[edges_b],
    #                            "pred_weight": edges_list
    #                            })
    # prediction = prediction.sort_values('pred_weight', ascending=False)

    prediction = pd.read_csv(path_prediction)

    gold_standard = pd.read_csv(path_gold_standard, sep="\t", header=None)
    pdf_aupr = sio.loadmat(path_pdf_aupr)
    pdf_auroc = sio.loadmat(path_pdf_auroc)

    # print(gold_standard)
    # print(prediction)
    prediction = prediction[["from", "to", "pred_weight"]]
    gold_standard.columns = ["from", "to", 'gt_weight']
    # print(gold_standard)
    # print(prediction)

    print("yuan")
    evaluates(gold_standard, prediction, pdf_auroc, pdf_aupr, TF_ids_list, gene_ids)

    print("random")
    prediction.loc[prediction['pred_weight'] < 0.1, 'pred_weight'] = np.nan
    # prediction_p["pred_weight"].apply(lambda x: prediction_p["pred_weight"].fillna(random.random()))
    rand_res = pd.DataFrame(0.1 * np.random.random(prediction['pred_weight'].shape))
    prediction["pred_weight"].fillna(rand_res[0], inplace=True)
    prediction = prediction.sort_values('pred_weight', ascending=False)
    evaluates(gold_standard, prediction, pdf_auroc, pdf_aupr, TF_ids_list, gene_ids)

    print("cut")
    prediction_shd = prediction[prediction['pred_weight'] > 0]
    evaluates(gold_standard, prediction_shd, pdf_auroc, pdf_aupr, TF_ids_list, gene_ids)
