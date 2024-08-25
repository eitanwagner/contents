
import argparse
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--template', type=str, default=" <NP> is a thing")
    parser.add_argument('--set_num', type=int, default=-1)

    parser.add_argument('--load', action="store_true", help="whether to load data")
    parser.add_argument('--only_make', action="store_true", help="whether to only generate data")
    parser.add_argument('--base_path', type=str, help="path to project (name included)")
    parser.add_argument('--cache_dir', type=str, default=None, help="cache directory")
    parser.add_argument('--seed', type=int, default=None)

    args = parser.parse_args()
    return args
