import argparse
import sys

argv = sys.argv
dataset = argv[1]


def acm_params():
    parser = argparse.ArgumentParser()
    parser.add_argument()


def set_params():
    if dataset == "acm":
        args = acm_params()
    # elif dataset == "dblp":
    #     args = dblp_params()
    # elif dataset == "aminer":
    #     args = aminer_params()
    # elif dataset == "freebase":
    #     args = freebase_params()
    return args
