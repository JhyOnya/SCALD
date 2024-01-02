# encoding: utf-8

import numpy as np
import pandas as pd
import igraph as ig
import networkx as nx
from sklearn.model_selection import train_test_split
import scipy.sparse as sp

import scipy.stats as st
import scipy.io as scio
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle
import os

from utils_jhy import *


# import eval


def get_dir_list():
    net_list = ["in_silico", "e_coli", "s_cere", "hESC", "mDC"]
    input_base_dir = "/home/fengke/pycode/my_exp/paper/exp/input/"
    input_dir_list = []
    for net in net_list:
        if net in ["in_silico", "e_coli", "s_cere"]:
            input_dir_list.append(input_base_dir + net)
        else:
            for type in ["Cell_type_specific", "Non_specific", "STRING"]:
                for num in ["500", "1000"]:
                    input_dir_list.append(input_base_dir + net + "/" + type + "/" + num)
    return input_dir_list


def generate_au_dist():
    # input_dir_list = get_dir_list()
    # dir_list = [i for i in input_dir_list if ('in_silico' not in i) and ('e_coli' not in i) and ('s_cere' not in i)]
    # for dir in dir_list:
    #     print(dir)
    #     plt.cla()
    #     plt.rcParams['figure.dpi'] = 300
    #     eval_res = dir + "/random_eval_res.tsv"
    #     res = pd.read_csv(eval_res, sep="\t", header=0)
    #     sns.set()
    #     auroc = res['auroc']
    #     aupr = res['aupr']
    #     auroc_fig = sns.histplot(auroc, binwidth=0.00001, kde=True, log_scale=(False, 10)).set_title(dir[43:])
    #     auroc_fig = auroc_fig.get_figure()
    #     auroc_fig.savefig(dir + '/auroc.png', dpi=300)
    #     X_plot = np.linspace(0, 1, 100000)
    #     scipy_kde = st.gaussian_kde(auroc)
    #     auroc_dense = scipy_kde(X_plot)
    #     plt.cla()
    #     # ------------------------------------
    #     aupr_fig = sns.histplot(aupr, binwidth=0.00001, kde=True, log_scale=(False, 10)).set_title(dir[43:])
    #     aupr_fig = aupr_fig.get_figure()
    #     aupr_fig.savefig(dir + '/aupr.png', dpi=300)
    #     scipy_kde = st.gaussian_kde(aupr)
    #     aupr_dense = scipy_kde(X_plot)
    #     scio.savemat(dir + "/AUROC.mat", {"X": X_plot, "Y": auroc_dense})
    #     scio.savemat(dir + "/AUPR.mat", {"X": X_plot, "Y": aupr_dense})
    X_plot = np.linspace(0, 1, 100000)
    auroc_dense = np.linspace(0, 1, 100000)
    aupr_dense = np.linspace(0, 1, 100000)
    return {"X": X_plot, "Y": auroc_dense}, {"X": X_plot, "Y": aupr_dense}


def Synthetic_CLD(object):
    def __init__(self, size=100):
        self.size = 100

    def get_msg(self):
        # read files
        chip_features = pd.read_csv(self.chip_features_path, sep='\t', )
        TF_ids = pd.read_csv(self.transcription_factors_path, sep='\t', header=None)
        data = pd.read_csv(self.expression_data_path, sep='\t', )
        gene_ids_pd = pd.read_csv(self.gene_ids_path, sep='\t', )
        gt_edges_pd = pd.read_csv(self.test_path, sep='\t', header=None, names=["from", "to", "gt_weight"])
        TF_ids_list = TF_ids[0].to_list()
        try:
            pdf_auroc = scio.loadmat(self.prob_dens_AUROC_path)
            pdf_aupr = scio.loadmat(self.prob_dens_AUPR_path)
        except:
            # pdf_auroc, pdf_aupr = generate_au_dist()
            pdf_auroc, pdf_aupr = None, None
        return data, gt_edges_pd, gene_ids_pd, TF_ids_list, pdf_auroc, pdf_aupr


# class Synthetic(object):
#     def gen_coef(self):
#         return np.random.uniform(1, 3)
#
#     def get_msg(self):
#         np.random.seed(0)
#         sample_size, loc, scale = 2000, 0.0, 1.0
#         T1 = np.random.normal(loc=loc, scale=scale, size=sample_size)
#         T2 = np.random.normal(loc=loc, scale=scale, size=sample_size)
#         C = np.random.normal(loc=loc, scale=scale, size=sample_size)
#         F = C * se
#
#     "\"Synthetic_CLD\"", lf.gen_coef() + np.random.normal(loc=loc, scale=scale, size=sample_size)
#     H = C * self.gen_coef() + np.random.normal(loc=loc, scale=scale, size=sample_size)
#     B = F * self.gen_coef() + T1 * self.gen_coef() + np.random.normal(loc=loc, scale=scale, size=sample_size)
#     D = H * self.gen_coef() + T2 * self.gen_coef() + np.random.normal(loc=loc, scale=scale, size=sample_size)
#     A = D * self.gen_coef() + T1 * self.gen_coef() + np.random.normal(loc=loc, scale=scale, size=sample_size)
#     E = B * self.gen_coef() + T2 * self.gen_coef() + np.random.normal(loc=loc, scale=scale, size=sample_size)
#     data = np.array([A, B, C, D, E, F, H]).T
#     data = pd.DataFrame(data, columns=["G1", "G2", "G3", "G4", "G5", "G6", "G7"])
#
#     gt_edges = [["G3", "G6", 1], ["G6", "G3", 0],
#                 ["G3", "G7", 1], ["G7", "G3", 0],
#                 ["G6", "G2", 1], ["G2", "G6", 0],
#                 ["G7", "G4", 1], ["G4", "G7", 0],
#                 ["G4", "G1", 1], ["G1", "G4", 0],
#                 ["G2", "G5", 1], ["G5", "G2", 0], ]
#     gt_edges_pd = pd.DataFrame(gt_edges, columns=["from", "to", "gt_weight"])
#
#     gene_ids = pd.DataFrame([['G1', 'str1'],
#                              ['G2', 'str2'],
#                              ['G3', 'str3'],
#                              ['G4', 'str4'],
#                              ['G5', 'str5'],
#                              ['G6', 'str6'],
#                              ['G7', 'str7'], ], columns=['#ID', 'Name'])
#     TF_ids_list = ['G1', 'G2', 'G3', 'G4', 'G5']
#     # TF_ids_list = None
#     gene_ids, pdf_auroc, pdf_aupr = None, None, None
#
#     # gene_ids["#ID"] = gene_ids["#ID"].str[1:].astype(int)
#     # gt_edges_pd["from"] = gt_edges_pd["from"].str[1:].astype(int)
#     # gt_edges_pd["to"] = gt_edges_pd["to"].str[1:].astype(int)
#     # return data, gt_edges_pd, \
#     #        gene_ids, TF_ids_list, \
#     #        {'X': np.array([[0, 0, 0]]), 'Y': np.array([[0, 0, 0]])}, \
#     #        {'X': np.array([[0, 0, 0]]), 'Y': np.array([[0, 0, 0]])}
#     return data, gt_edges_pd, gene_ids, TF_ids_list, pdf_auroc, pdf_aupr


class Dream5(object):
    def __init__(self, id=0):
        train_paths = ["./datasets/dream5/training data/network1-in silico/net1_",
                       "./datasets/dream5/training data/network2-s.aureus/net2_",
                       "./datasets/dream5/training data/network3-e.coil/net3_",
                       "./datasets/dream5/training data/network4-s.cerevisiae/net4_", ]
        test_paths = ["./datasets/dream5/test data/DREAM5_NetworkInference_GoldStandard_Network1 - in silico.tsv",
                      "./datasets/dream5/test data/DREAM5_NetworkInference_GoldStandard_Network2 - S. aureus.tsv",
                      "./datasets/dream5/test data/DREAM5_NetworkInference_GoldStandard_Network3 - E. coli.tsv",
                      "./datasets/dream5/test data/DREAM5_NetworkInference_GoldStandard_Network4 - S. cerevisiae.tsv", ]
        prob_dens_path = "./datasets/dream5/Evaluation_scripts/INPUT/probability_densities/"

        self.id = id
        self.train_path = train_paths[self.id]
        self.test_path = test_paths[self.id]
        self.chip_features_path = self.train_path + "chip_features.tsv"
        self.expression_data_path = self.train_path + "expression_data.tsv"
        self.gene_ids_path = self.train_path + "gene_ids.tsv"
        self.transcription_factors_path = self.train_path + "transcription_factors.tsv"
        self.prob_dens_AUPR_path = prob_dens_path + "Network%s_AUPR.mat" % (self.id + 1)
        self.prob_dens_AUROC_path = prob_dens_path + "Network%s_AUROC.mat" % (self.id + 1)

    def get_msg(self):
        # read files
        chip_features = pd.read_csv(self.chip_features_path, sep='\t', )
        TF_ids = pd.read_csv(self.transcription_factors_path, sep='\t', header=None)
        data = pd.read_csv(self.expression_data_path, sep='\t', )
        gene_ids_pd = pd.read_csv(self.gene_ids_path, sep='\t', )
        gt_edges_pd = pd.read_csv(self.test_path, sep='\t', header=None, names=["from", "to", "gt_weight"])
        # pred_edges_pd_ori = graph2edges(pd.DataFrame(np.zeros((TF_ids.shape[0],))))
        TF_ids_list = TF_ids[0].to_list()
        try:
            pdf_auroc = scio.loadmat(self.prob_dens_AUROC_path)
            pdf_aupr = scio.loadmat(self.prob_dens_AUPR_path)
        except:
            # pdf_auroc, pdf_aupr = generate_au_dist()
            pdf_auroc, pdf_aupr = None, None

        # gene_ids["#ID"] = gene_ids["#ID"].str[1:].astype(int)
        # gt_edges_pd["from"] = gt_edges_pd["from"].str[1:].astype(int)
        # gt_edges_pd["to"] = gt_edges_pd["to"].str[1:].astype(int)

        # # get adj
        # # edgelist = g.get_edgelist()
        # # g=ig.Graph.Read_Edgelist(GoldStandard_Network, directed=False)
        # g = nx.DiGraph()
        # g.add_nodes_from(list(gene_ids["#ID"].values))  # 节点序号从1开始编号
        # # g.add_nodes_from(list(transcription_factors.values.reshape(-1)))
        # # edgeList=[(f,e) for [f,e,_] in edges]
        # # g.add_edges_from(edgeList)
        # for i in range(len(ground_truth_edges)):
        #     if ground_truth_edges[i][2] == 0:
        #         g.add_edge(ground_truth_edges[i][0], ground_truth_edges[i][1], weight=-2)
        #     else:
        #         g.add_edge(ground_truth_edges[i][0], ground_truth_edges[i][1], weight=ground_truth_edges[i][2])
        # ground_truth_adj = nx.adjacency_matrix(g).A
        # ground_truth_adj = np.where(ground_truth_adj == 0, -1, ground_truth_adj)
        # ground_truth_adj = np.where(ground_truth_adj == -2, 0, ground_truth_adj)  # unknown=-1  unlinked=0  linked=1

        # return chip_features, expression_data, gene_ids, transcription_factors, A, ground_truth_adj
        return data, gt_edges_pd, gene_ids_pd, TF_ids_list, pdf_auroc, pdf_aupr


class Dream5_in_silico(Dream5):
    def __init__(self, id=0):
        super(Dream5_in_silico, self).__init__(id)


class Dream5_s_aureus(Dream5):
    def __init__(self, id=1):
        super(Dream5_s_aureus, self).__init__(id)


class Dream5_e_coil(Dream5):
    def __init__(self, id=2):
        super(Dream5_e_coil, self).__init__(id)


class Dream5_s_cerevisiae(Dream5):
    def __init__(self, id=3):
        super(Dream5_s_cerevisiae, self).__init__(id)


class sachs(object):
    def __init__(self):
        self.data_path = "./datasets/sachs/%s"
        self.test_path = "./datasets/sachs/gt.csv"
        # self.expression_data_path = path + "expression_data.tsv"
        # self.tf_path = path + "transcription_factors.tsv"
        # self.gene_ids = path + "gene_ids.tsv"
        self.id_dict = {
            "praf": "Raf",
            "pmek": "Mek",
            "plcg": "Plcγ",
            "PIP2": "PIP2",
            "PIP3": "PIP3",
            "p44/42": "Erk",
            "pakts473": "Akt",
            "PKA": "PKA",
            "PKC": "PKC",
            "P38": "P38",
            "pjnk": "Jnk",
        }

    def get_msg(self):
        # read csves
        data_name_ls = ["1. cd3cd28.csv",
                        "2. cd3cd28icam2.csv",
                        "3. cd3cd28+aktinhib.csv",
                        "4. cd3cd28+g0076.csv",
                        "5. cd3cd28+psitect.csv",
                        "6. cd3cd28+u0126.csv",
                        "7. cd3cd28+ly.csv",
                        "8. pma.csv",
                        "9. b2camp.csv",
                        # "10. cd3cd28icam2+aktinhib.csv",
                        # "11. cd3cd28icam2+g0076.csv",
                        # "12. cd3cd28icam2+psit.csv",
                        # "13. cd3cd28icam2+u0126.csv",
                        # "14. cd3cd28icam2+ly.csv",
                        ]
        data_ls = []
        data = pd.DataFrame()
        for pre in data_name_ls:
            data_pre = pd.read_csv(self.data_path % pre, sep=',', )
            data_ls.append(data_pre)
            data = data.append(data_pre)
        s = data.columns.to_series()
        data.columns = s.map(self.id_dict).fillna(s)

        gt_edges_pd = pd.read_csv(self.test_path, sep='\t', header=None, names=["from", "to", "gt_weight"])
        print("data.shape", data.shape)
        print("gt.shape", gt_edges_pd.shape)
        TF_ids_list = data.columns.to_list()
        return data, gt_edges_pd, None, TF_ids_list, None, None


class seq(object):
    def __init__(self, data_name="mDC-Non_specific-500"):
        if data_name.endswith("sub"):
            path = "./datasets/BEELINE/%s_" % (data_name)

            self.expression_data_path = path + "expression_data.tsv"
            self.test_path = path + "gold_standard.tsv"
            self.tf_path = path + "transcription_factors.tsv"
            self.gene_ids = path + "gene_ids.tsv"
        else:
            pth = data_name.split("-")
            path = "./datasets/BEELINE/%s/%s/%s/" % (pth[0], pth[1], pth[2])

            self.expression_data_path = path + "expression_data.tsv"
            self.test_path = path + "gold_standard.tsv"
            self.tf_path = path + "transcription_factors.tsv"
            self.gene_ids = path + "gene_ids.tsv"

    def get_msg(self):
        # read files
        TF_ids = pd.read_csv(self.tf_path, sep='\t', header=None)
        data = pd.read_csv(self.expression_data_path, sep='\t', )
        gt_edges_pd = pd.read_csv(self.test_path, sep='\t', header=None, names=["from", "to", "gt_weight"])
        print("gt.shape", gt_edges_pd.shape)
        gt_edges_pd["gt_weight"] = 1
        TF_ids_list = TF_ids[0].to_list()
        gene_ids_pd = pd.read_csv(self.gene_ids, sep='\t', )
        pdf_auroc, pdf_aupr = None, None
        return data, gt_edges_pd, gene_ids_pd, TF_ids_list, pdf_auroc, pdf_aupr


class synthetic(object):
    def __init__(self, data_name="Synthetic_CLD1"):
        pth = data_name.split("-")
        path = "./datasets/Synthetic_CLD/%s" % data_name

        self.expression_data_path = path + "_data.csv"
        self.test_path = path + "_gt.csv"

    def get_msg(self):
        # read files
        data = pd.read_csv(self.expression_data_path, sep='\t', )
        gt_edges_pd = pd.read_csv(self.test_path, sep='\t', header=None, names=["from", "to", "gt_weight"])
        gt_edges_pd["gt_weight"] = 1
        TF_ids_list = data.columns.to_list()
        gene_ids_pd, pdf_auroc, pdf_aupr = None, None, None
        return data, gt_edges_pd, gene_ids_pd, TF_ids_list, pdf_auroc, pdf_aupr


class breast_cancer(object):

    def __init__(self, data_name="breast_cancer"):
        self.data_name = data_name
        self.train_path = r"./datasets/breast_cancer/breast_cancer_count_mtx_1k_norm.txt"
        self.gene_path = r"./datasets/breast_cancer/brca_degenes.txt"

    def get_msg(self):
        data_ = pd.read_csv(self.train_path, sep='\t', index_col=0).T
        genes_ = pd.read_csv(self.gene_path, sep='\t', index_col=0)

        return data_, genes_


class GSE4183(object):

    def __init__(self, data_name="GSE4183"):
        self.data_name = data_name
        self.train_path = r"./datasets/GSE4183/GSE4183_data.txt"
        self.id_path = r"./datasets/GSE4183/GPL570_id.txt"
        self.colon_id_path = r"./datasets/GSE4183/colon_id.txt"
        self.tf_path = r"./datasets/GSE4183/Homo_sapiens_TF.txt"

    def get_msg(self):

        id_ = pd.read_csv(self.id_path, sep='\t', index_col=0, header=0)
        id_c = id_[["Gene Symbol"]]  # 54675
        id_c = id_c.dropna()  # 45782

        data_ = pd.read_csv(self.train_path, sep='\t', header=0, index_col=0).T  # 53 x 54708
        data_value = data_.iloc[:, 33:].astype("float")  # 53 x 54675
        data_value = data_value.loc[:, id_c.index]  # 53 x 45782
        data_value.columns = id_c["Gene Symbol"][data_value.columns]
        # 数据均值排序去重
        data_value.loc['mean'] = data_value.apply(lambda x: x.mean(), axis=0)
        data_value = data_value.sort_values(by='mean', axis=1, ascending=False)
        data_value.drop("mean", inplace=True)
        data_value = data_value.loc[:, ~data_value.columns.duplicated(keep='first')]  # 53 x 23520


        categories = ['colon_normal', 'colon_IBD', 'colon_adenoma', 'colon_CRC']
        datasets = {prefix: data_value[data_value.index.str.startswith(prefix)] for prefix in categories}


        from scipy import stats

        t_stat, p_value = stats.ttest_ind(datasets['colon_normal'], datasets['colon_CRC'])
        genes_ls = data_value.columns[np.where(p_value < 0.001)]  # 536


        tf_data_ = pd.read_csv(self.tf_path, sep='\t', header=0, index_col=0)  # 1665
        TF_ids_list_all = list(tf_data_.loc[:, "Symbol"])  # 1665
        TF_ids_list = list(set(TF_ids_list_all) & set(genes_ls))  # 30

        data_group_ls = [datasets[pre].loc[:, TF_ids_list + list(set(genes_ls) - set(TF_ids_list_all))]
                         for pre in categories if pre != 'colon_adenoma'] # [8, 15, 15] x 536

        return data_group_ls, TF_ids_list

class TCGA_RPPA(object):
    def __init__(self, data_name="TCGA-THCA2"):
        self.data_name = data_name
        self.train_path = r"./datasets/TCGA_RPPA/{}.csv".format(data_name)
        if data_name == "TCGA-THCA1" or data_name == "TCGA-THCA2":
            self.data_name = data_name[:-1]
        self.stage_path = r"./datasets/TCGA_RPPA/clinical.tsv"
        # self.gt_path = r"./datasets/TCGA_RPPA/clinical.tsv"

    def get_msg(self):
        # read files
        gt_link = pd.read_csv("./datasets/TCGA_RPPA/9606.protein.links.v11.5.txt", sep=" ")
        gt_info = pd.read_csv("./datasets/TCGA_RPPA/9606.protein.info.v11.5.txt", sep="\t")
        gt_aliases = pd.read_csv("./datasets/TCGA_RPPA/9606.protein.aliases.v11.5.txt", sep="\t")
        gt_anti = pd.read_csv("./datasets/TCGA_RPPA/TCGA_antibodies_descriptions.gencode.v36.tsv",
                              sep="\t")  # https://gdc.cancer.gov/about-data/gdc-data-processing/gdc-reference-files

        data_ = pd.read_csv(self.train_path, sep=',', index_col=4)
        data = data_.drop(["AGID", "lab_id", "catalog_number", "set_id"], axis=1).T

        data = data.dropna(axis=1, how="all")
        data = data.fillna(data.mean())
        sample_name_list = [pre[:-4] for pre in data.index]
        data.index = sample_name_list

        stages_ = pd.read_csv(self.stage_path, sep='\t', index_col=1)
        stages_ = stages_[stages_["project_id"] == self.data_name]
        stages_['ajcc_pathologic_stage'] = stages_['ajcc_pathologic_stage'].str.replace('A', "")
        stages_['ajcc_pathologic_stage'] = stages_['ajcc_pathologic_stage'].str.replace('B', "")
        stages_['ajcc_pathologic_stage'] = stages_['ajcc_pathologic_stage'].str.replace('C', "")

        sample_name_list = list(set(sample_name_list).intersection(set(stages_.index.to_list())))

        stages = stages_.loc[sample_name_list]["ajcc_pathologic_stage"]
        stages = stages.loc[~stages.index.duplicated()]
        data_ls = []
        # data_ls.append(data.loc[stages[stages == "Stage 0"].index])
        data_ls.append(data.loc[stages[stages == "Stage I"].index])
        data_ls.append(data.loc[stages[stages == "Stage II"].index])
        data_ls.append(data.loc[stages[stages == "Stage III"].index])
        data_ls.append(data.loc[stages[stages == "Stage IV"].index])

        return data_ls


class TCGA_RPPA_old(object):
    def __init__(self, data_name="TCGA-THCA2"):
        self.data_name = data_name
        self.train_path = r"./datasets/TCGA_RPPA/{}.csv".format(data_name)
        if data_name == "TCGA-THCA1" or data_name == "TCGA-THCA2":
            self.data_name = data_name[:-1]

    def get_msg(self):
        data_ = pd.read_csv(self.train_path, sep=',', index_col=4)
        data = data_.drop(["AGID", "lab_id", "catalog_number", "set_id"], axis=1).T
        data = data.dropna(axis=1, how="all")
        data = data.fillna(data.mean())

        # read files
        gt_link = pd.read_csv("./datasets/TCGA_RPPA/9606.protein.links.v11.5.txt", sep=" ")
        gt_info = pd.read_csv("./datasets/TCGA_RPPA/9606.protein.info.v11.5.txt", sep="\t")
        # gt_aliases = pd.read_csv("./datasets/TCGA_RPPA/9606.protein.aliases.v11.5.txt", sep="\t")
        gt_anti = pd.read_csv("./datasets/TCGA_RPPA/TCGA_antibodies_descriptions.gencode.v36.tsv",
                              sep="\t")  # https://gdc.cancer.gov/about-data/gdc-data-processing/gdc-reference-files

        null_protein_anti = []
        id_dict = {}
        for pre_protein in data.columns:
            if gt_anti['peptide_target'].isin([pre_protein]).any():
                gene_name = gt_anti[gt_anti['peptide_target'].isin([pre_protein])]['gene_name'].values[0]
                # gene_name = gene_name.split("/")[0]
                if gt_info['preferred_name'].isin([gene_name]).any():
                    id_pre = gt_info[gt_info['preferred_name'].isin([gene_name])]['#string_protein_id'].values[0]
                    if id_pre in id_dict.keys():
                        if len(id_dict[id_pre]) > len(pre_protein):
                            id_dict[id_pre] = pre_protein
                    else:
                        id_dict[id_pre] = pre_protein
                else:
                    null_protein_anti.append(pre_protein)
            else:
                null_protein_anti.append(pre_protein)
        # print(len(data.columns), len(null_protein_anti), null_protein_anti)
        # print(id_dict)

        gt_link = gt_link[(gt_link.protein1.isin(id_dict.keys()) & gt_link.protein2.isin(id_dict.keys()))]
        gt_link.protein1 = gt_link.protein1.map(id_dict)
        gt_link.protein2 = gt_link.protein2.map(id_dict)
        gt_link.reset_index()
        gt_link.columns = ["from", "to", "gt_weight"]
        gt_link["gt_weight"] = 1

        sample_name_list = [pre[:-4] for pre in data.index]
        data.index = sample_name_list
        data = data[id_dict.values()]

        return data, gt_link


class bnlearn_data(object):
    def __init__(self, data_name=""):
        import bnlearn as bn

        path = "./datasets/{}/".format(data_name)
        if not os.path.exists(path):
            os.makedirs(path)
        self.train_path = path + "data.csv"
        self.test_path = path + "gt.csv"


    def get_msg(self):
        # read files
        data = pd.read_csv(self.train_path, sep='\t', index_col=0)
        gt_adj_pd = pd.read_csv(self.test_path, sep='\t', index_col=0, dtype="str")
        gt_adj_pd[gt_adj_pd == "True"] = 1
        gt_adj_pd[gt_adj_pd == "False"] = 0

        column = ["from", "to", "gt_weight"]
        edges_a, edges_b = np.where(gt_adj_pd == 1)
        edges_list = [gt_adj_pd.values[f, t] for (f, t) in zip(edges_a, edges_b)]
        gt_edges_pd = pd.DataFrame({column[0]: gt_adj_pd.index[edges_a],
                                    column[1]: gt_adj_pd.index[edges_b],
                                    column[2]: edges_list
                                    })
        return data, gt_edges_pd, None, None, None, None


class asia(bnlearn_data):
    def __init__(self):
        super(asia, self).__init__("asia")


if __name__ == "__main__":

    data_ls = [
        "GSE4183",
    ]

    for pre in data_ls:
        print(pre)
        dataset = GSE4183(pre)
        data, gt_edges_pd, gene_ids_pd, TF_ids_list, pdf_auroc, pdf_aupr = dataset.get_msg()
        for i, pre in enumerate(data):
            print("  Stage", i + 1, " sample_shape", pre.shape)

