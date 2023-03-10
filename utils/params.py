import argparse

dataset = "amz"


def amazon_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="amz")
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--test_size', type=int, default=0.3)
    parser.add_argument('--num_view', type=int, default=3)
    parser.add_argument('--num_pos', type=int, default=7)
    parser.add_argument('--num_neigh', type=int, default=10)
    args, _ = parser.parse_known_args()
    return args


def yelp_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="yelp")
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--test_size', type=int, default=0.3)
    parser.add_argument('--num_view', type=int, default=3)
    parser.add_argument('--num_pos', type=int, default=5)
    parser.add_argument('--num_neigh', type=int, default=10)
    args, _ = parser.parse_known_args()
    return args


def set_params():
    if dataset == "amz":
        args = amazon_params()
    elif dataset == "yelp":
        args = yelp_params()
    else:
        raise ValueError("unsupported dataset")
    return args
