import argparse

dataset = "Amazon"


def amazon_params():
    parser = argparse.ArgumentParser()
    parser.add_argument()
    args, _ = parser.parse_known_args()
    return args


def yelp_params():
    parser = argparse.ArgumentParser()
    parser.add_argument()
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
