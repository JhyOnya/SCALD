import datetime
import os, shutil

# 初始化
import pandas as pd
import numpy as np

import matplotlib.pylab as plt
import time

log = None
tofile = True

start = datetime.datetime.now()
zeroToday = start - datetime.timedelta(hours=start.hour, minutes=start.minute, seconds=start.second,
                                       microseconds=start.microsecond)
now_time = datetime.timedelta(hours=start.hour, minutes=start.minute, seconds=start.second,
                              microseconds=start.microsecond)


def fill_zeros_neg(pred_adj_pd_ori):
    random_np = -np.random.random(size=pred_adj_pd_ori.shape)
    pred_adj_pd_zero = (pred_adj_pd_ori == 0)
    pred_adj_pd_rdm = random_np * pred_adj_pd_zero
    return pred_adj_pd_ori + pred_adj_pd_rdm


def graph2edges(graph_pd, column=["from", "to", "pred_weight"]):
    graph = graph_pd.values
    edges_a, edges_b = np.where(graph != 0)
    edges_list = [graph[f, t] for (f, t) in zip(edges_a, edges_b)]
    edges_pd = pd.DataFrame({column[0]: graph_pd.index[edges_a],
                             column[1]: graph_pd.index[edges_b],
                             column[2]: edges_list
                             })
    return edges_pd


def edges2graph(edges_pd, features_ls=None):
    G_np = np.zeros((len(features_ls), len(features_ls)))

    index_f = features_ls.get_indexer(list(edges_pd.iloc[:, 0]))
    index_t = features_ls.get_indexer(list(edges_pd.iloc[:, 1]))
    G_np[index_f, index_t] = list(edges_pd.iloc[:, 2])
    return pd.DataFrame(G_np,  # dtype=int,
                        index=features_ls,
                        columns=features_ls)


def draw(X, Y, title, dir, x_label=None, y_label=None):
    plt.plot(X, Y, label=title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    # plt.show()
    plt.savefig(dir + "pic_%s.pdf" % title)
    plt.clf()


def sort_edges(edges_pd, sort="descending"):
    if sort == "descending":
        return edges_pd.sort_values('pred_weight', ascending=False)
    elif sort == "random":
        return edges_pd.reindex(np.random.permutation(edges_pd.index))
    else:
        print("wrong sort method, do not change. sort='descending' or 'random'")
        return edges_pd


def tab_printer(args):
    from texttable import Texttable
    args = vars(args)
    keys = args.keys()
    # keys = sorted(args.keys())
    t = Texttable()
    t.set_cols_dtype(['t', 't'])
    p_v = [["start time", start]]
    p_v.extend([[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    t.add_rows(p_v)
    print(t.draw())
    print()
    return t.draw()


def initFile(args):
    args.cache_dir = "./cache/%s/" % args.data
    # 是否测试
    if not args.isnottest:
        args.cache_dir = args.cache_dir + "_/"
        global tofile
        tofile = False

    if args.log != "":
        args.log = "_" + args.log

    if not os.path.exists(args.cache_dir):  # create directory
        os.makedirs(args.cache_dir)

    his = [listx for listx in os.listdir(args.cache_dir)
           if listx != "_" and os.path.isdir(args.cache_dir + listx)]
    pre_log_num = str(his).count(args.method + args.log) + 1
    args.log = (args.method + args.log + "_" + "%02d" % pre_log_num).replace(" ", "_")
    args.cache_dir = args.cache_dir + args.log + "/"

    if not os.path.exists(args.cache_dir):  # create directory
        os.makedirs(args.cache_dir)

    return


def getFile(args):
    args.cache_dir = "./cache/%s/" % args.data
    if args.log != "":
        args.log = "_" + args.log

    if not os.path.exists(args.cache_dir):  # create directory
        os.makedirs(args.cache_dir)

    his = [listx for listx in os.listdir(args.cache_dir)
           if listx != "_" and os.path.isdir(args.cache_dir + listx)]
    pre_log_num = str(his).count(args.method + args.log)
    args.cache_dir = args.cache_dir + args.log + "/"

    return pd.read_csv(args.cache_dir + "adj_pred.csv", index_col=0)
