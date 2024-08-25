
import argparse
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--template', type=str, default=" <NP> is a thing")
    parser.add_argument('--set_num', type=int, default=-1)
    parser.add_argument('--model_type', type=str, default="flan", help="flan, llama, roberta, electra, xlm-roberta")
    parser.add_argument('--model_size', type=str, default="base", help="base, large, small, xl, xxl, 7b, 11b, 13b")
    parser.add_argument('--dataset', type=str, default="wikitext-2", help="wikitext-2, new_news, new_news2, total_noise, extra_noise, NPs, noise")

    parser.add_argument('--load', action="store_true", help="whether to load data")
    parser.add_argument('--only_make', action="store_true", help="whether to only generate data")
    parser.add_argument('--entropies', action="store_true", help="whether to test entropies")

    parser.add_argument('--base_path', type=str, help="path to project (name included)")
    parser.add_argument('--cache_dir', type=str, default=None, help="cache directory")
    parser.add_argument('--seed', type=int, default=None)

    args = parser.parse_args()
    return args
