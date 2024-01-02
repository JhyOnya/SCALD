# -*- coding:utf-8 -*-


from utils_jhy import *
from argparse import ArgumentParser
import main
import main_analysis


def set_parser():
    parser = ArgumentParser()
    # onya
    parser.add_argument("-notcuda", action='store_true', default=False)
    parser.add_argument('-r', "--isnottest", action='store_true', default=False)  # False True if-test
    parser.add_argument('-log', type=str, default="")
    parser.add_argument('-cache_dir', type=str, default="./cache/")

    parser.add_argument('-threshold', type=float, default=0)

    parser.add_argument('-data', type=str, default="sachs", )

    parser.add_argument('-method', type=str, default="auto")
    parser.add_argument('-batch', type=int, default=64)  # 每次训练样本数
    parser.add_argument('-lr', type=float, default=1e-3)  # 312优化掉
    parser.add_argument('-decay', type=float, default=1e-4)  # 312优化掉

    args = parser.parse_args()
    if args.data.startswith("TCGA"):
        args.data_type = "TCGA_ls"
    elif args.data.startswith("Dream5_e_coil"):
        args.data_type = "Dream5"
        args.method = "method_dream5_ecoil" if args.method == "auto" else args.method
    elif args.data.startswith("Dream5_s_cerevisiae"):
        args.data_type = "Dream5"
        args.method = "method_dream5_scerevisiae" if args.method == "auto" else args.method
    elif args.data.startswith("Synthetic"):
        args.data_type = "Synthetic"
    elif args.data.startswith("mDC") or args.data.startswith("hESC"):
        args.method = "method_seq" if args.method == "auto" else args.method
        args.data_type = "seq"
    elif args.data.startswith("GSE4183"):
        args.method = "method_GSE4183" if args.method == "auto" else args.method
        args.data_type = "GSE4183"

    elif args.data.startswith("sachs"):
        args.method = "method_sachs" if args.method == "auto" else args.method
        args.data_type = "sachs"
    else:
        print("undefined type of", args.data)
    return args


if __name__ == "__main__":
    args = set_parser()

    initFile(args)
    pred_adj_pd_ori=main.main(args)
    msg_dict = main_analysis.main(args, pred_adj_pd_ori)
    print(msg_dict)

