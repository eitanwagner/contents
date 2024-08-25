
from transformers import T5Tokenizer, T5ForConditionalGeneration, BatchEncoding
from transformers import AutoTokenizer, RobertaForMaskedLM, DebertaForMaskedLM, DebertaV2ForMaskedLM
from transformers import XLMRobertaForMaskedLM, XLMRobertaXLForMaskedLM, AutoModelForMaskedLM, ElectraForMaskedLM
import torch
import torch.nn.functional as F
import numpy as np
from scipy.special import logsumexp
from scipy.spatial import distance
import json
import tqdm
import sys
import pandas as pd

dev = "cuda" if torch.cuda.is_available() else torch.device("cpu")

from utils import parse_args
args = parse_args()


# ***************************

def H(x):
    """ Entropy from logits"""
    b = F.softmax(x, dim=-1) * F.log_softmax(x, dim=-1)
    b = -1.0 * b.sum()
    return b

# ***********************
# make synthetic datasets

def make_extra_noise(length=50, num_samples=1000):
    """
    completely random strings (like passwords)
    :param length:
    :param num_samples:
    :return:
    """
    import string
    import random
    print(f"Making extra noise. {num_samples} texts.")
    if args.load:
        print('Loading data')
        with open(args.base_path + 'data/_extra_noise.json', 'r') as f:
            texts = json.load(f)[:num_samples]
    else:
        characters = string.ascii_letters + string.digits + string.punctuation
        texts = [''.join(random.choice(characters) for i in range(length)) for _ in range(num_samples)]
        with open(args.base_path + 'data/_extra_noise.json', 'w') as f:
            json.dump(texts, f)
    return texts

def make_total_noise(tokenizer, length=20, num_samples=1000):
    """
    completely random strings (like passwords)
    :param length:
    :param num_samples:
    :return:
    """
    print(f"Making total noise. {num_samples} texts.")
    if args.load:
        print('Loading data')
        with open(args.base_path + 'data/_total_noise.json', 'r') as f:
            texts = json.load(f)[:num_samples]
    else:
        tokens = np.random.randint(tokenizer.vocab_size, size=(num_samples, length))
        texts = tokenizer.batch_decode(tokens, skip_special_tokens=True)
        with open(args.base_path + 'data/_total_noise.json', 'w') as f:
            json.dump(texts, f)
    return texts

def make_synthetic(template=" <NP> is a thing", tokenizer=None, noise=False, num_samples=10000, uses_eos=True):
    """
    Make synthetic data
    :return: list of texts, location of first inserted token
    """
    from datasets import load_dataset
    import spacy

    if args.only_make:
        num_samples = 50000

    template = args.template

    print(f"Making synthetic data. {num_samples} texts.")
    print(f"Template: {template}")
    print(f"Noise: {noise}")
    inserted_id = tokenizer(" <NP>")['input_ids'][1]
    # inserted_id = 28696
    # obtain list of two-word noun-phrases
    if args.load:
        if "common" in sys.argv:
            print('Loading data - common NPs')
            with open(args.base_path + 'data/common_nps2.json', 'r') as f:
                pairs = json.load(f)[:num_samples]
        else:
            print(f'Loading data - {tokenizer.name_or_path.split("/")[-1].split("-")[0]}')
            with open(args.base_path + f'data/{tokenizer.name_or_path.split("/")[-1].split("-")[0]}'
                      f'{"-ul2" if "ul2" in tokenizer.name_or_path else ""}_nps2{"-noise" if noise else ""}.json', 'r') as f:
                pairs = json.load(f)[:num_samples]
    else:
        seed = 1
        print(f"Seed: {seed}")
        dataset = load_dataset("AlekseyKorshuk/fiction-books")
        texts = dataset['train']['text']
        nlp = spacy.load('en_core_web_md')
        np_list = []
        w_list = []
        for text in tqdm.tqdm(texts[:1500]):
            # Assuming you have a large body of text in 'text' variable.
            if len(text) > 9e5:
                continue
            doc = nlp(text)
            #ignore also compound? and proper nouns?
            if not noise and "llama" not in sys.argv:
                np_list = np_list + [chunk.text for chunk in doc.noun_chunks if
                                     len(chunk) == 2 and chunk[0].dep_ not in ["det", "poss"]
                                     and len(tokenizer(chunk.text)['input_ids']) == (4 if "flan" not in sys.argv else 3)
                                     and len(tokenizer(" " + chunk.text)['input_ids']) == (4 if "flan" not in sys.argv else 3)
                                     and chunk.root.tag_ not in ["NNS", "NNPS", "SPACE"]
                                     and not chunk[0].is_stop and not chunk[1].is_stop
                                     and not chunk[0].is_punct and not chunk[1].is_punct]
            elif not noise and "llama" in sys.argv:
                np_list = np_list + [chunk.text for chunk in doc.noun_chunks if
                                     len(chunk) == 2 and chunk[0].dep_ not in ["det", "poss"]
                                     and len(tokenizer(chunk.text)['input_ids']) == 3
                                     and len(tokenizer("a " + chunk.text)['input_ids']) == 4
                                     and chunk.root.tag_ not in ["NNS", "NNPS", "SPACE"]
                                     and not chunk[0].is_stop and not chunk[1].is_stop
                                     and not chunk[0].is_punct and not chunk[1].is_punct]
            elif "llama" in sys.argv:
                w_list = w_list + [t.lemma_ for t in doc if
                                   not t.is_stop and not t.is_punct and not t.is_digit and not t.is_space
                                   and not t.tag_ in ["NNP"] and len(tokenizer(t.lemma_)['input_ids']) == 2
                                   and len(tokenizer("a " + t.lemma_)['input_ids']) == 3]
            else:
                w_list = w_list + [t.lemma_ for t in doc if
                                   not t.is_stop and not t.is_punct and not t.is_digit and not t.is_space
                                   and not t.tag_ in ["NNP"] and len(tokenizer(t.lemma_)['input_ids']) == (3 if "flan" not in sys.argv else 2)
                                   and len(tokenizer(" " + t.lemma_)['input_ids']) == (3 if "flan" not in sys.argv else 2)]
        if not noise:
            np.random.seed(seed)
            nps = list(set(np_list))
            nps.sort()
            np.random.shuffle(nps)
            pairs = nps[:int(num_samples * 1.3)]
        else:
            np.random.seed(seed)
            w_list = list(set(w_list))
            w_list.sort()
            _pairs = np.random.choice(w_list, size=(num_samples, 2))
            pairs = [" ".join([p1, p2]) for p1, p2 in _pairs]
        pairs = [p for p in pairs if p.isascii()]
        with open(args.base_path + f'data/{tokenizer.name_or_path.split("/")[-1].split("-")[0]}'
                  f'{"-ul2" if "ul2" in tokenizer.name_or_path else ""}_nps2{"-noise" if noise else ""}.json', 'w') as f:
            json.dump(pairs, f)

    return [template.replace("<NP>", p).strip() for p in pairs], tokenizer(template)['input_ids'].index(inserted_id)

# ***************

def on_data_calibration(dataset_name='wikitext-2', size="small", version="", return_tokenizer=False, tokenizer=None):
    """
    Test joint probability calibration on a text dataset
    :param dataset:
    :param size:
    :return:
    """
    set_num = 0
    set_num_s = ""
    if args.set_num >= 0:
        set_num = args.set_num
        set_num_s = f"s{set_num}"

    if not return_tokenizer:
        index = None
        deps = "deps" in sys.argv
        return_probs = "return_probs" in sys.argv
        deps2 = "deps2" in sys.argv
        entropies = "entropies" in sys.argv
        texts = []
        print("Using loss (counting also the special tokens)")
        if dataset_name == 'wikitext-2':
            from datasets import load_dataset
            dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split="train")
            texts = dataset['text']
        elif dataset_name == 'new_news':
            with open(args.base_path + 'data/news-2.7.2023.json', 'r') as file:
                texts = [t['text'] for t in json.load(file) if t['title'].find("Subscribe") == -1]
            with open(args.base_path + 'data/news-6.7.2023.json', 'r') as file:
                texts = texts + [t['text'] for t in json.load(file) if t['title'].find("Subscribe") == -1]
        elif dataset_name == 'new_news2':
            with open(args.base_path + 'data/news-4.9.2023.json', 'r') as file:
                texts = [t['text'] for t in json.load(file) if t['title'].find("Subscribe") == -1]
            with open(args.base_path + 'data/news-18.9.2023.json', 'r') as file:
                texts = texts + [t['text'] for t in json.load(file) if t['title'].find("Subscribe") == -1]
        elif dataset_name == "NPs":
            texts, index = make_synthetic(tokenizer=tokenizer, uses_eos="llama" not in sys.argv)
        elif dataset_name == "noise":
            texts, index = make_synthetic(tokenizer=tokenizer, noise=True, uses_eos="llama" not in sys.argv)
        elif dataset_name == "extra_noise":
            texts, index = make_extra_noise(), None
        elif dataset_name == "total_noise":
            texts, index = make_total_noise(tokenizer=tokenizer), None
        if "only_make" in sys.argv:
            return

        if "ranks" in sys.argv:
            texts = texts[:25]

    # load model and tokenizer
    if "flan" in sys.argv:
        # flan models but used like the llama case
        _size = size
        if size == "3b":
            _size = "xl"
        if size == "11b":
            _size = "xxl"
        name = f"google/flan-t5-{_size}"
        tokenizer = AutoTokenizer.from_pretrained(name, legacy=False)
        if return_tokenizer:
            return tokenizer
        model = T5ForConditionalGeneration.from_pretrained(name, device_map="auto", load_in_4bit=dev == "cuda")
        # model = T5ForConditionalGeneration.from_pretrained(name)
    elif "flan-ul2" in sys.argv:
        # ul2 models but not in the t5 setting
        name = f"google/flan-ul2"
        from transformers import BitsAndBytesConfig, AutoModelForSeq2SeqLM
        double_quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(name, legacy=False)
        if return_tokenizer:
            return tokenizer
        model = AutoModelForSeq2SeqLM.from_pretrained(name, device_map="auto",
                                                      quantization_config=double_quant_config)
    elif "llama" in sys.argv:
        chat = "chat" in sys.argv
        size = "13b"
        if "7b" in sys.argv:
            size = "7b"
        if "70b" in sys.argv:
            size = "70b"
        from transformers import LlamaForCausalLM, LlamaTokenizer
        # name = "meta-llama/Llama-2-13b-hf"
        name = f"meta-llama/Llama-2-{size}{'-chat' if chat else ''}-hf"
        print(name)
        from transformers import BitsAndBytesConfig, AutoModelForSeq2SeqLM
        double_quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(name)
        tokenizer.pad_token = tokenizer.eos_token
        if return_tokenizer:
            return tokenizer
        if "testing" not in sys.argv:
            model = LlamaForCausalLM.from_pretrained(name, device_map="auto",
                                                     quantization_config=double_quant_config)
        else:
            model = None

    elif "roberta" in sys.argv:
        name = f"roberta-{size}" if size != "small" else "distilroberta-base"
        tokenizer = AutoTokenizer.from_pretrained(name)
        if return_tokenizer:
            return tokenizer
        model = RobertaForMaskedLM.from_pretrained(name)
    elif "xlm-roberta" in sys.argv:
        if size in ["base", "large"]:
            name = f"xlm-roberta-{size}"
            tokenizer = AutoTokenizer.from_pretrained(name)
            if return_tokenizer:
                return tokenizer
            model = XLMRobertaForMaskedLM.from_pretrained(name)
        elif size in ["xl", "xxl"]:
            name = f"facebook/xlm-roberta-{size}"
            tokenizer = AutoTokenizer.from_pretrained(name)
            if return_tokenizer:
                return tokenizer
            model = AutoModelForMaskedLM.from_pretrained(name, device_map='auto', load_in_4bit=True)
    elif "electra" in sys.argv:
        name = f"google/electra-{size}-generator"
        tokenizer = AutoTokenizer.from_pretrained(name)
        if return_tokenizer:
            return tokenizer
        model = ElectraForMaskedLM.from_pretrained(name)
    elif "deberta" in sys.argv:
        name = f"microsoft/deberta-{size}"
        tokenizer = AutoTokenizer.from_pretrained(name)
        if return_tokenizer:
            return tokenizer
        model = DebertaForMaskedLM.from_pretrained(name)
    elif "deberta-v2" in sys.argv:
        name = f"microsoft/deberta-v2-{size}"
        # model = DebertaV2ForMaskedLM.from_pretrained(name, device_map="auto", quantization_config=double_quant_config)
        tokenizer = AutoTokenizer.from_pretrained(name)
        if return_tokenizer:
            return tokenizer
        model = DebertaV2ForMaskedLM.from_pretrained(name)
    if "testing" not in sys.argv:
        model.eval()
    if "flan-ul2" not in sys.argv and "ul2" not in sys.argv and "11b" not in sys.argv and "llama" not in sys.argv:
        if "t5" in sys.argv:
            model.to(dev)

    if entropies:
        return llama_entropies(texts, model, tokenizer)

    if "llama" in sys.argv or "flan-ul2" in sys.argv or "flan" in sys.argv:
        if return_probs:
            probs, ids = llama_pair_probabilities(texts, model, tokenizer, fixed_pos=index, return_probs=True, set_num=set_num)
            with open(args.base_path + f'results/{dataset_name}{set_num_s}_scores_{name.split("/")[-1]}_probs.npy',
                      'wb') as outfile:
                np.save(outfile, probs)
            with open(args.base_path + f'results/{dataset_name}{set_num_s}_scores_{name.split("/")[-1]}_ids.npy',
                      'wb') as outfile:
                np.save(outfile, ids)
            return
        else:
            pp = llama_pair_probabilities(texts, model, tokenizer, fixed_pos=index, set_num=set_num)
    else:
        if return_probs:
            probs, ids = mlm_pair_probabilities(texts, model, tokenizer, temperature=1., deps=deps, deps2=deps2, fixed_pos=index, return_probs=True)
            with open(args.base_path + f'results/{dataset_name}_scores_{name.split("/")[-1]}_probs.npy', 'wb') as outfile:
                np.save(outfile, probs)
            with open(args.base_path + f'results/{dataset_name}_scores_{name.split("/")[-1]}_ids.npy', 'wb') as outfile:
                np.save(outfile, ids)
            return
        else:
            print("len(texts)")
            print(len(texts))
            print(model)
            pp = mlm_pair_probabilities(texts, model, tokenizer, temperature=1., deps=deps, deps2=deps2, fixed_pos=index)
            print("\n\npp")
            print(pp)
            print("\n\n")
    try:
        pp.sort(reverse=True)
    except:
        print("Did not sort")
    l1_2 = "l1_2" in sys.argv
    l2_1 = "l2_1" in sys.argv

    na = "_na" if "non_ascii" in sys.argv else ""
    ranks = "_ranks" if "ranks" in sys.argv else ""
    with open(args.base_path + f'results/{dataset_name}{set_num_s}_scores_{name.split("/")[-1]}'
              f'{"_deps" if deps else ""}{"_deps2" if deps2 else ""}{"_l1_2" if l1_2 else ""}{"_l2_1" if l2_1 else ""}'
              f'{na}{ranks}.json', 'w') as outfile:
        json.dump(pp, outfile)
    on_data_evaluation(pp, max_ent=np.log(len(tokenizer.vocab)))


def on_data_evaluation(pp, max_ent=11., cutoffs=None, high_H=False):
    # using quantile cutoffs
    all_probs_12, all_probs_21, _, Hs = list(zip(*pp))[1:5]
    # w1_only, w2_only, w1 | w2, w2 | w1
    Hs1 = np.array([h[0] for h in Hs])
    Hs2 = np.array([h[1] for h in Hs])
    Hs1_2 = np.array([h[2] for h in Hs])
    Hs2_1 = np.array([h[3] for h in Hs])
    Hs12 = np.array([h[0] + h[3] for h in Hs])
    Hs21 = np.array([h[1] + h[2] for h in Hs])
    H_diffs = Hs12 - Hs21

    allHs = np.hstack([Hs1, Hs2, Hs1_2, Hs2_1])
    avgHs = np.mean(allHs)
    print("Average H: ")
    print(avgHs * 2.)
    if cutoffs is not None:
        # Hs_cutoff1 = cutoffs[0]
        # Hs_cutoff2 = cutoffs[1]
        Hs_cutoff1, Hs_cutoff2 = np.quantile(allHs, cutoffs)
    else:
        Hs_cutoff1 = max_ent / 3
        Hs_cutoff2 = 2 * max_ent / 3
    print("Cutoffs:")
    print(Hs_cutoff1, Hs_cutoff2)

    # avgHs = np.mean(Hs12 + Hs21) / 2

    print("Num Samples:")
    print(len(all_probs_12))

    js_all = distance.jensenshannon(np.exp(all_probs_12), np.exp(all_probs_21))
    print("JS all:")
    print(js_all)

    print("AVG p12:")
    print(sum(np.array(all_probs_12)) / len(all_probs_12))
    print("AVG p21:")
    print(sum(np.array(all_probs_21)) / len(all_probs_12))
    print("AVG p12 - p21:")
    print(sum(np.array(all_probs_12) - np.array(all_probs_21)) / len(all_probs_12))
    print("Random AVG p12 - p21:")
    arr2 = np.array(all_probs_21)
    np.random.shuffle(arr2)
    print(sum(np.array(all_probs_12) - arr2) / len(all_probs_12))
    print("Count p12 > p21:")
    print(sum(np.array(all_probs_12) > np.array(all_probs_21)) / len(all_probs_12))
    print("Random Count p12 > p21:")
    print(sum(np.array(all_probs_12) > arr2) / len(all_probs_12))
    print("AVG abs(p12 - p21):")
    print(sum(abs(np.array(all_probs_12) - np.array(all_probs_21))) / len(all_probs_12))
    print("Var(p12 - p21):")
    print(np.var(np.array(all_probs_12) - np.array(all_probs_21)))

    print("AVG H12 - H21:")
    print(sum(Hs12 - Hs21) / len(Hs12))
    print("AVG abs(H12 - H21):")
    print(sum(abs(Hs12 - Hs21)) / len(Hs12))
    print("Var H12 - H21")
    print((Hs12 - Hs21).var())
    print("Var Hs")
    print(allHs.var())
    print("AVG H1 - H2:")
    print(sum(Hs1 - Hs2) / len(Hs12))
    print("AVG H1|2 - H2|1:")
    print(sum(Hs1_2 - Hs2_1) / len(Hs12))

    # if "t5" not in sys.argv and "llama" not in sys.argv and "flan-ul2" not in sys.argv:
    if "t5" not in sys.argv:
        # analysis by entropy
        pp_ent0 = [p for p in pp if p[4][0] >= p[4][1]]
        pp_ent1 = [p for p in pp if p[4][0] < p[4][1]]
        pp_ent0_0 = [p for p in pp_ent0 if p[4][3] >= p[4][2]]
        pp_ent0_1 = [p for p in pp_ent0 if p[4][3] < p[4][2]]
        pp_ent1_0 = [p for p in pp_ent1 if p[4][3] >= p[4][2]]
        pp_ent1_1 = [p for p in pp_ent1 if p[4][3] < p[4][2]]
        pp_ent0_c = pp_ent0_0 + pp_ent1_0
        pp_ent1_c = pp_ent0_1 + pp_ent1_1
        pp_ent12 = [p for p in pp if p[4][0] + p[4][3] >= p[4][1] + p[4][2]]
        pp_ent21 = [p for p in pp if p[4][0] + p[4][3] < p[4][1] + p[4][2]]
        if high_H:
            H_cutoff = np.quantile(np.abs(H_diffs), 0.8)
            pp_ent12_h = [p for p in pp if p[4][0] + p[4][3] >= p[4][1] + p[4][2] if abs(p[4][0] + p[4][3] - p[4][1] - p[4][2]) >= H_cutoff]
            pp_ent21_h = [p for p in pp if p[4][0] + p[4][3] < p[4][1] + p[4][2] if abs(p[4][0] + p[4][3] - p[4][1] - p[4][2]) >= H_cutoff]
            pp_ent12_l = [p for p in pp if p[4][0] + p[4][3] >= p[4][1] + p[4][2] if abs(p[4][0] + p[4][3] - p[4][1] - p[4][2]) < H_cutoff]
            pp_ent21_l = [p for p in pp if p[4][0] + p[4][3] < p[4][1] + p[4][2] if abs(p[4][0] + p[4][3] - p[4][1] - p[4][2]) < H_cutoff]

        def print_by_cond(pp0, pp1, name0, name1):
            print(f"AVG p12, p21 for cases {name0}, {name1}:")
            p12_ent0 = sum([p[1] for p in pp0]) / len(pp0)
            p21_ent0 = sum([p[2] for p in pp0]) / len(pp0)
            p12_ent1 = sum([p[1] for p in pp1]) / len(pp1)
            p21_ent1 = sum([p[2] for p in pp1]) / len(pp1)
            print(p12_ent0, p21_ent0, p12_ent1, p21_ent1)
            print(f"AVG p12 - p21 for cases {name0}, {name1}:")
            p_ent0 = sum([p[1] - p[2] for p in pp0]) / len(pp0)
            p_ent1 = sum([p[1] - p[2] for p in pp1]) / len(pp1)
            print(p_ent0, p_ent1)
            print(f"var(p12 - p21) for cases {name0}, {name1}:")
            var_ent0 = np.var([p[1] - p[2] for p in pp0])
            var_ent1 = np.var([p[1] - p[2] for p in pp1])
            print(var_ent0, var_ent1)
            count0 = len([p for p in pp0 if p[1] > p[2]]) / len(pp0)
            count1 = len([p for p in pp1 if p[1] > p[2]]) / len(pp1)
            print(f"Count (p12 > p21) for case {name0}:")
            print(count0)
            print(f"Count (p12 > p21) for case {name1}:")
            print(count1)

        print("\nlens for cases H_w1 >= H_w2, H_w1 < H_w2:")
        print(len(pp_ent0), len(pp_ent1))
        if len(pp_ent0) > 0 and len(pp_ent1) > 0:
            print_by_cond(pp_ent0, pp_ent1, name0="H_w1 >= H_w2", name1="H_w1 < H_w2")

        print("\nlens for cases H_w2|1 > H_w1|2, H_w2|1 < H_w1|2:")
        print(len(pp_ent0_c), len(pp_ent1_c))
        if len(pp_ent0_c) > 0 and len(pp_ent1_c) > 0:
            print_by_cond(pp_ent0_c, pp_ent1_c, name0="H_w2|1 >= H_w1|2", name1="H_w2|1 < H_w1|2",)

        print("\nlens for for cases H_w1 > H_w2, H_w1 < H_w2, for each one two cases, H_w2|1 > H_w1|2, H_w2|1 < H_w1|2:")
        print(len(pp_ent0_0), len(pp_ent0_1), len(pp_ent1_0), len(pp_ent1_1))
        if len(pp_ent0_0) > 0 and len(pp_ent0_1) > 0 and len(pp_ent1_0) > 0 and len(pp_ent1_1) > 0:
            print("AVG p12, p21 for cases H_w1 > H_w2, H_w1 < H_w2, two cases for each:")
            p12_ent0_0 = sum([p[1] for p in pp_ent0_0]) / len(pp_ent0_0)
            p21_ent0_0 = sum([p[2] for p in pp_ent0_0]) / len(pp_ent0_0)
            p12_ent1_0 = sum([p[1] for p in pp_ent1_0]) / len(pp_ent1_0)
            p21_ent1_0 = sum([p[2] for p in pp_ent1_0]) / len(pp_ent1_0)
            p12_ent0_1 = sum([p[1] for p in pp_ent0_1]) / len(pp_ent0_1)
            p21_ent0_1 = sum([p[2] for p in pp_ent0_1]) / len(pp_ent0_1)
            p12_ent1_1 = sum([p[1] for p in pp_ent1_1]) / len(pp_ent1_1)
            p21_ent1_1 = sum([p[2] for p in pp_ent1_1]) / len(pp_ent1_1)
            print(p12_ent0_0, p21_ent0_0, p12_ent0_1, p21_ent0_1, p12_ent1_0, p21_ent1_0, p12_ent1_1, p21_ent1_1)
            print("AVG p12 - p21 for cases H_w1 > H_w2, H_w1 < H_w2, two cases for each:")
            p_ent0_0 = sum([p[1] - p[2] for p in pp_ent0_0]) / len(pp_ent0_0)
            p_ent0_1 = sum([p[1] - p[2] for p in pp_ent0_1]) / len(pp_ent0_1)
            p_ent1_0 = sum([p[1] - p[2] for p in pp_ent1_0]) / len(pp_ent1_0)
            p_ent1_1 = sum([p[1] - p[2] for p in pp_ent1_1]) / len(pp_ent1_1)
            print(p_ent0_0, p_ent0_1, p_ent1_0, p_ent1_1)
            count0_0 = len([p for p in pp_ent0_0 if p[1] > p[2]]) / len(pp_ent0_0)
            count0_1 = len([p for p in pp_ent0_1 if p[1] > p[2]]) / len(pp_ent0_1)
            count1_0 = len([p for p in pp_ent1_0 if p[1] > p[2]]) / len(pp_ent1_0)
            count1_1 = len([p for p in pp_ent1_1 if p[1] > p[2]]) / len(pp_ent1_1)
            print("Counts (p12 > p21) for case H_w1 > H_w2 (two counts, by second entropy):")
            print(count0_0, count0_1)
            print("Count (p12 > p21) for case H_w1 < H_w2 (two counts, by second entropy):")
            print(count1_0, count1_1)

        print("\nlens for cases H_w12 >= H_w21, H_w12 < H_w21:")
        print(len(pp_ent12), len(pp_ent21))
        if len(pp_ent12) > 0 and len(pp_ent21) > 0:
            print_by_cond(pp_ent12, pp_ent21, name0="H_w12 >= H_w21", name1="H_w12 < H_w21")

        if high_H:
            print("\nlens for cases H_w12_h >= H_w21_h, H_w12_h < H_w21_h:")
            print(len(pp_ent12_h), len(pp_ent21_h))
            if len(pp_ent12_h) > 0 and len(pp_ent21_h) > 0:
                print_by_cond(pp_ent12_h, pp_ent21_h, name0="H_w12_h >= H_w21_h", name1="H_w12_h < H_w21_h")
            print("\nlens for cases H_w12_l >= H_w21_l, H_w12_l < H_w21_l:")
            print(len(pp_ent12_l), len(pp_ent21_l))
            if len(pp_ent12_l) > 0 and len(pp_ent21_l) > 0:
                print_by_cond(pp_ent12_l, pp_ent21_l, name0="H_w12_l >= H_w21_l", name1="H_w12_l < H_w21_l")

        pp_ent_l = [p for p in pp if (p[4][0] + p[4][3] + p[4][1] + p[4][2]) / 4 >= avgHs]
        pp_ent_s = [p for p in pp if (p[4][0] + p[4][3] + p[4][1] + p[4][2]) / 4 < avgHs]
        print("\nlens for cases H_w >= AvgH_w, H_w < AvgH_w:")
        print(len(pp_ent_l), len(pp_ent_s))
        if len(pp_ent_l) > 0 and len(pp_ent_s) > 0:
            print_by_cond(pp_ent_l, pp_ent_s, name0="H_w >= AvgH_w", name1="H_w < AvgH_w")

        pp_ent_1 = [p for p in pp if (p[4][0] + p[4][3] + p[4][1] + p[4][2]) / 4 <= Hs_cutoff1]
        pp_ent_2 = [p for p in pp if Hs_cutoff1 < (p[4][0] + p[4][3] + p[4][1] + p[4][2]) / 4 <= Hs_cutoff2]
        pp_ent_3 = [p for p in pp if Hs_cutoff2 < (p[4][0] + p[4][3] + p[4][1] + p[4][2]) / 4]
        print("\nlens for cases H_w <= Hs_cutoff1, Hs_cutoff1 < H_w <= Hs_cutoff2, Hs_cutoff2 < H_w:")
        print(len(pp_ent_1), len(pp_ent_2), len(pp_ent_3))
        if len(pp_ent_1) > 0 and len(pp_ent_2) > 0 and len(pp_ent_3) > 0:
            print("AVG p12, p21 for H_w <= Hs_cutoff1, Hs_cutoff1 < H_w <= Hs_cutoff2, Hs_cutoff2 < H_w:")
            p12_ent_1 = sum([p[1] for p in pp_ent_1]) / len(pp_ent_1)
            p21_ent_1 = sum([p[2] for p in pp_ent_1]) / len(pp_ent_1)
            p12_ent_2 = sum([p[1] for p in pp_ent_2]) / len(pp_ent_2)
            p21_ent_2 = sum([p[2] for p in pp_ent_2]) / len(pp_ent_2)
            p12_ent_3 = sum([p[1] for p in pp_ent_3]) / len(pp_ent_3)
            p21_ent_3 = sum([p[2] for p in pp_ent_3]) / len(pp_ent_3)
            print(p12_ent_1, p21_ent_1, p12_ent_2, p21_ent_2, p12_ent_3, p21_ent_3)
            print("AVG p12 - p21 for H_w <= Hs_cutoff1, Hs_cutoff1 < H_w <= Hs_cutoff2, Hs_cutoff2 < H_w:")
            p_ent_1 = sum([p[1] - p[2] for p in pp_ent_1]) / len(pp_ent_1)
            p_ent_2 = sum([p[1] - p[2] for p in pp_ent_2]) / len(pp_ent_2)
            p_ent_3 = sum([p[1] - p[2] for p in pp_ent_3]) / len(pp_ent_3)
            print(p_ent_1, p_ent_2, p_ent_3)
            count_1 = len([p for p in pp_ent_1 if p[1] > p[2]]) / len(pp_ent_1)
            count_2 = len([p for p in pp_ent_2 if p[1] > p[2]]) / len(pp_ent_2)
            count_3 = len([p for p in pp_ent_3 if p[1] > p[2]]) / len(pp_ent_3)
            print("Count (p12 > p21) for case H_w <= Hs_cutoff1:")
            print(count_1)
            print("Count (p12 > p21) for case Hs_cutoff1 < H_w <= Hs_cutoff2:")
            print(count_2)
            print("Count (p12 > p21) for case Hs_cutoff2 < H_w:")
            print(count_3)


# *******************
# for evaluation

def load_probs(dataset_name="wikitext", name="roberta-base", set_num=""):
    set_num_s = ""
    if set_num != "":
        set_num_s = f"s{set_num}"
    with open(args.base_path + f'results/{dataset_name}{set_num_s}_scores_{name.split("/")[-1]}_.json', 'r') as infile:
        pp = json.load(infile)
    return pp

def load_full_probs(dataset_name="wikitext", name="roberta-base", set_num=""):
    set_num_s = ""
    if set_num != "":
        set_num_s = f"s{set_num}"
    with open(args.base_path + f'results/{dataset_name}{set_num_s}_scores_{name.split("/")[-1]}_probs.npy', 'rb') as infile:
        probs = np.load(infile)
    with open(args.base_path + f'results/{dataset_name}{set_num_s}_scores_{name.split("/")[-1]}_ids.npy', 'rb') as infile:
        ids = np.load(infile)
    return probs, ids

def to_bin(p):
    from bisect import bisect
    intervals = np.arange(0, 10, 0.25)
    return intervals[bisect(intervals, p)-1]

def scatter_entropy_consistency(pp, probs=None, ids=None, name="", separate=False, box=False, average=False):
    # make pairs of entropy, consistency
    import matplotlib.pyplot as plt
    entropies = []
    consistency = []
    consistency2 = []
    # for _, p12, p21, _, (H1, H2, H1_2, H2_1), _ in pp:
    for p in pp:
        _, p12, p21, _, Hs = p[:5]
        if separate:
            consistency.append(abs(p12))
            consistency2.append(abs(p21))
        else:
            consistency.append(abs(p12 - p21))
        entropies.append(sum(Hs) / 4)
    if not separate or box:
        # [to_bin(e) for e in entropies]
        df = pd.DataFrame({"H": [to_bin(e) for e in entropies], "Consistency": consistency})
        df.boxplot(column=['Consistency'], by="H", sym="")
        plt.locator_params(axis='x', nbins=10)
    else:
        # df.plot.scatter("H", "Consistency", s=2)
        plt.scatter(entropies, consistency, c="b", s=2)
        plt.xlabel('H')
        plt.ylabel('Consistency')
    if probs is None:
        plt.title(name)
    else:
        plt.title(name + ("_c" if separate else "") + f"; log(V)={np.log(probs.shape[-1]):.2f}")
    plt.show()
    if separate:
        if box:
            df = pd.DataFrame({"H": [to_bin(e) for e in entropies], "Consistency": consistency2})
            df.boxplot(column=['Consistency'], by="H", sym="")
            plt.locator_params(axis='x', nbins=10)
        else:
            plt.ylim([0, 25])
            plt.scatter(entropies, consistency2, c="r", s=2)
        if probs is None:
            plt.title(name)
        else:
            plt.title(name + ("_c2" if separate else "") + f"; log(V)={np.log(probs.shape[-1]):.2f}")
    plt.show()

    if probs is not None:
        avg_entropies = []
        avg_consistency = []
        from scipy.special import log_softmax, softmax
        from scipy.stats import entropy
        for i, (p, id) in enumerate(zip(probs[:], ids[:])):
            # mlm and llama
            entropies = []
            consistency = []
            for t in np.linspace(1, 5, 30):
                # ps = log_softmax(p[:4] / t, axis=-1)
                ps = log_softmax(p / t, axis=-1)
                chosen_ps = ps[np.arange(ps.shape[0]), id]
                set1 = [0, 1, 6, 7] if ps.shape[0] == 8 else [0, 3]
                set2 = [2, 3, 4, 5] if ps.shape[0] == 8 else [1, 2]
                consistency.append(abs(chosen_ps[set1].sum() - chosen_ps[set2].sum()))
                # consistency.append(abs(chosen_ps[:ps.shape[0] // 2].sum() - chosen_ps[ps.shape[0] // 2:].sum()))
                # consistency.append(abs(ps[0, id[0]] + ps[3, id[3]] - ps[2, id[2]] - ps[1, id[1]]))
                # consistency.append(abs(chosen_ps[0] + chosen_ps[3] - chosen_ps[1] - chosen_ps[2]))
                # H = -np.sum((softmax(p[:4] / t, axis=-1) * log_softmax(p[:4] / t, axis=-1)), axis=-1)
                H = -np.sum((softmax(p / t, axis=-1) * log_softmax(p / t, axis=-1)), axis=-1)
                entropies.append(np.mean(H))

            if not average:
                N = len(probs)
                cmap = plt.cm.get_cmap("hsv", N + 1)
                # df = pd.DataFrame({"H": entropies[:], "Consistency": consistency[:]})
                # df.plot.scatter("H", "Consistency", c=c)
                plt.scatter(entropies, consistency, c=np.array(cmap(i)).reshape(1, -1), s=2)
            else:
                avg_entropies.append(entropies)
                avg_consistency.append(consistency)
        if average:
            entropies = np.mean(avg_entropies, axis=0)
            consistency = np.mean(avg_consistency, axis=0)
            plt.scatter(entropies, consistency, s=4)
        plt.title(name + f"; log(V)={np.log(probs.shape[-1]):.2f}" + " - Temperature")
        plt.xlabel('H')
        plt.ylabel('Consistency')
        plt.show()

def plot_by_name(dataset="new_news", model_name="Llama-2-13b-hf", set_num="", separate=False, box=False, with_probs=True, average=False):
    pp = load_probs(dataset, model_name, set_num=set_num)
    probs, ids = None, None
    if with_probs:
        probs, ids = load_full_probs(dataset, model_name, set_num=set_num)
    scatter_entropy_consistency(pp, probs, ids, name=", ".join([dataset, model_name]), separate=separate, box=box,
                                average=average)
    return pp, probs, ids

def evaluate_by_name(dataset="new_news", model_name="Llama-2-13b-hf", set_num="", cutoffs=None, high_H=False):
    pp = load_probs(dataset, model_name, set_num=set_num)
    on_data_evaluation(pp, cutoffs=cutoffs, high_H=high_H)
    return


def llama_entropies(texts, model, tokenizer):
    chat = "chat" in sys.argv
    llama = "llama" in sys.argv
    extra_ids = tokenizer.convert_tokens_to_ids(["%", "@"])  # extra_ids[0] is the one to predict
    colon_id = tokenizer.convert_tokens_to_ids(":")
    system_prompt = "You will be given a passage with one masked token that you should fill in. We denote this token by %. The passage might also contain corrupted tokens denoted by @. You are not expected to fill in corrupted tokens - fill only the masked one. Your answer should include the filled-in token only with no extra explanations or context."
    less_docs = "70b" in sys.argv
    import spacy
    nlp = spacy.load("en_core_web_md")
    all_entropies = []
    for text in tqdm.tqdm(texts[:250] if less_docs else texts):
        entropies = []
        if len(text) < 100 or not text.isascii():
            continue
        if chat:
            text = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\nPassage: {text} [/INST] "
        else:
            # text = system_prompt + "\nPassage: " + text + "\nAnswer:"
            text = system_prompt + "\nPassage: " + text + "\nAnswer"
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs2 = BatchEncoding(inputs)

        doc = nlp(text)
        t2st = {i: doc.char_span(s, e, alignment_mode="contract") for i, (s, e) in enumerate(inputs.encodings[0].offsets)}
        def is_stop(i):
            # check if words are problematic
            if t2st[i] is None:  # covers token that are not full words
                return True
            if t2st[i][0].is_stop or t2st[i][0].is_digit \
                    or t2st[i][0].is_punct:
                return True

        for i in range(1, len(inputs.input_ids[0]) - 10):
            if i < len(system_prompt.split()) + 10 or is_stop(i):
                continue
            if not llama:
                inputs2['input_ids'] = torch.clone(inputs.input_ids)
                labels = torch.tensor([colon_id, inputs2['input_ids'][0, i].item(), tokenizer.eos_token_id]).unsqueeze(0)
                inputs2['input_ids'][0, i] = extra_ids[0]
                for j in range(i+1, len(inputs2['input_ids'][0])):
                    if not is_stop(j):
                        inputs2['input_ids'][0, j] = extra_ids[1]
                # inputs2['input_ids'][0, i+1:] = extra_ids[1]
                # print(len(inputs2['input_ids'][0, i + 1:]))
            else:
                _labels = torch.tensor([inputs['input_ids'][0, i].item(), tokenizer.eos_token_id]).unsqueeze(0)
                inputs2['input_ids'] = torch.cat([torch.clone(inputs.input_ids)[:, :-1], _labels], dim=1)
                labels = torch.clone(inputs2['input_ids'])
                inputs2['input_ids'][0, i] = extra_ids[0]
                for j in range(i+1, len(inputs2[0]) - 10):
                    if not is_stop(j):
                        inputs2['input_ids'][0, j] = extra_ids[1]
                # inputs2['input_ids'][0, i + 1:-10] = extra_ids[1]
                print(len(inputs2['input_ids'][0, i + 1:-10]))
                labels[0, :-2] = -100
            with torch.no_grad():
                out = model(inputs2['input_ids'].to(dev), labels=labels.to(dev))
            if not llama:
                entropies.append(float(H(out.logits[0, -2]).cpu().numpy()) + float(H(out.logits[0, -1]).cpu().numpy()))
            else:
                entropies.append(float(H(out.logits[0, -3]).cpu().numpy()) + float(H(out.logits[0, -2]).cpu().numpy()))
        all_entropies.append(entropies)
    return all_entropies


# ***********************
# calculate probabilities

def llama_pair_probabilities(texts, model, tokenizer, fixed_pos=None, return_probs=False, set_num=0):
    """
    Measures probabilities for adjacent token pairs in the given texts
    :param texts: list of strings
    :param model:
    :param tokenizer:
    :return:
    """
    skip_word = "skip_word" in sys.argv
    deps = "deps" in sys.argv
    deps3 = "deps3" in sys.argv
    chat = "chat" in sys.argv
    llama = "llama" in sys.argv
    if deps3:
        deps = True
    import spacy
    nlp = spacy.load("en_core_web_md")
    # dep_tags = ["amod"]
    dep_tags = ["amod", "compound", "nummod", "poss"]
    print(f"Using only {dep_tags}")
    print("For deps3 avoiding all backward tags")
    print("Avoiding spacy stopwords, punctuation, digits")
    print("Removing last space from input")
    print("using -3 for entropy")
    print("For T5, putting colon in the labels")
    print("Using entropy sum")
    print("Fixed a lot in the flan case")

    if "llama" not in sys.argv:
        extra_ids = tokenizer.convert_tokens_to_ids(["%", "@"])  # extra_ids[0] is the one to predict
        if "other_prompt2" in sys.argv:
            extra_ids = tokenizer.convert_tokens_to_ids(["^", "$"])  # extra_ids[0] is the one to predict
    else:
        extra_ids = tokenizer("%")["input_ids"][1], tokenizer("@")["input_ids"][1]  # extra_ids[0] is the one to predict
        if "other_prompt2" in sys.argv:
            extra_ids = tokenizer("^")["input_ids"][1], tokenizer("$")["input_ids"][1]  # extra_ids[0] is the one to predict
    print("Extra_ids:")
    print(extra_ids)
    l = []
    probs = []
    ids = []
    colon_id = tokenizer.convert_tokens_to_ids(":")
    system_prompt = "You will be given a passage with one masked token that you should fill in. We denote this token by %. The passage might also contain corrupted tokens denoted by @. You are not expected to fill in corrupted tokens - fill only the masked one. Your answer should include the filled-in token only with no extra explanations or context."
    if "other_prompt" in sys.argv:
        system_prompt = "You will be given a passage masked tokens. The masks tokens are marked with % and, possibly, also @. Fill only the token that is masked with % -- do not fill in ta token masked by @. Your answer should include the filled-in token only. Do not add explanations or context."
    elif "other_prompt2" in sys.argv:
        system_prompt = "You will be given a passage with one masked token that you should fill in. We denote this token by ^. The passage might also contain corrupted tokens denoted by $. You are not expected to fill in corrupted tokens - fill only the masked one. Your answer should include the filled-in token only with no extra explanations or context."
    less_docs = "70b" in sys.argv and fixed_pos is None
    na = "non_ascii" in sys.argv
    # first = True
    if return_probs:
        r_texts = texts[:20]
    else:
        idx = set_num
        r_texts = texts[(200 if na else 500) * idx:(200 if na else 500) * (idx+1)] if less_docs else texts
    for ti, text in tqdm.tqdm(enumerate(r_texts)):
        o_text = text
        if chat:
            text_len = len(tokenizer(text, return_tensors="pt")["input_ids"][0]) - 1
            # text = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\nPassage: {text} [/INST]"
            text = f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\nPassage: {text} [/INST] "
            tail_len = len(tokenizer(" [/INST] ", return_tensors="pt")["input_ids"][0]) - 1
        else:
            text_len = len(tokenizer(text, return_tensors="pt")["input_ids"][0]) - (1 if "flan" not in sys.argv else 0)
            # text = system_prompt + "\nPassage: " + text + "\nAnswer:"
            # text = system_prompt + "\nPassage: " + text + ("\nAnswer: " if llama else "\nAnswer")  # is this also for llama??
            text = system_prompt + "\nPassage: " + text + ("." if fixed_pos is not None else "") + "\nAnswer: "  # is this also for llama??
            tail_len = len(tokenizer(".\nAnswer: ", return_tensors="pt")["input_ids"][0]) - 1

        # if fixed_pos is None or "flan" not in sys.argv:
        if True:
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True,
                               max_length=512 if fixed_pos is not None else None)
            inputs2 = BatchEncoding(inputs)
            doc = nlp(text)
            t2st = {i: doc.char_span(s, e, alignment_mode="contract") for i, (s, e) in
                    enumerate(inputs.encodings[0].offsets)}

            def is_stop(i, j):
                # check if words are problematic
                if t2st[i] is None or t2st[j] is None:  # covers token that are not full words
                    return True
                if t2st[i][0].is_stop or t2st[j][0].is_stop or t2st[i][0].is_digit or t2st[j][0].is_digit \
                        or t2st[i][0].is_punct or t2st[j][0].is_punct:
                    return True

            def is_skip(i, j, all_tags=False):
                return t2st[i][0].head.i == t2st[j][0].i and t2st[j][0].i > t2st[i][0].i and \
                       t2st[i][0].n_rights + t2st[i][0].n_lefts == 0 and (all_tags or t2st[i][0].dep_ in dep_tags)
        if fixed_pos is None:
            if "non_ascii" in sys.argv:
                if len(text) < 100 or text.isascii() or len(text) > 10000:
                    continue
            else:
                if ("13b" in sys.argv and ti == 868) or ("70b" in sys.argv and set_num == 1 and ti == 368):
                    continue
                if len(text) < 100 or not text.isascii():
                    continue

            new_fixed_pos = len(inputs["input_ids"][0]) - tail_len - text_len + 1
            token_range = range(new_fixed_pos, new_fixed_pos + text_len - 3)
        else:
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs2 = BatchEncoding(inputs)
            # inputs3 = BatchEncoding(inputs)
            new_fixed_pos = len(inputs["input_ids"][0]) - tail_len - text_len + fixed_pos
            token_range = range(new_fixed_pos, new_fixed_pos+1)

        for i in token_range:
            dep = None
            j = i + 1
            t_ii = 0
            if fixed_pos is None:
                if skip_word:
                    j = i + 2
                if i < len(system_prompt.split()) or is_stop(i, j):
                    continue
                if deps3 and is_skip(i, j, all_tags=True):
                    continue
                elif deps and not deps3 and not is_skip(i, j):
                    continue
                t_ii = t2st[i][0].i
            if not is_stop(i, j) and is_skip(i, j, all_tags=True):
                dep = t2st[i][0].dep_
            # TODO: batchify

            # mask only first
            if not llama:
                if fixed_pos is not None:
                    if "other_prompt2" in sys.argv:
                        n_text = text.replace(o_text, " ".join(["^"] + o_text.split()[1:]))
                    else:
                        n_text = text.replace(o_text, " ".join(["%"] + o_text.split()[1:]))
                    inputs2 = tokenizer(n_text, return_tensors="pt", padding=True, truncation=True)
                    labels = tokenizer(f" {o_text.split()[0]}", return_tensors="pt", padding=True, truncation=True)["input_ids"]
                else:
                    if "other_prompt2" in sys.argv:
                        n_text = text[:t2st[i].start_char] + "^" + text[t2st[i].end_char:]
                    else:
                        n_text = text[:t2st[i].start_char] + "%" + text[t2st[i].end_char:]
                    inputs2 = tokenizer(n_text, return_tensors="pt", padding=True, truncation=True)
                    labels = tokenizer(t2st[i].text, return_tensors="pt", padding=True, truncation=True)["input_ids"]
            else:
                _labels = torch.tensor([inputs['input_ids'][0, i].item(), tokenizer.eos_token_id]).unsqueeze(0)
                inputs2['input_ids'] = torch.cat([torch.clone(inputs.input_ids)[:, :-1], _labels], dim=1)
                inputs2['input_ids'][0, i] = extra_ids[0]
                labels = torch.clone(inputs2['input_ids'])
                labels[0, :-2] = -100
            with torch.no_grad():
                out = model(inputs2['input_ids'].to(dev), labels=labels.to(dev))

            # mask only second
            if not llama:
                if fixed_pos is not None:
                    s_text = o_text.split()
                    if "other_prompt2" in sys.argv:
                        s_text[1] = "^"
                    else:
                        s_text[1] = "%"
                    n_text = text.replace(o_text, " ".join(s_text))
                    inputs2 = tokenizer(n_text, return_tensors="pt", padding=True, truncation=True)
                    labels = tokenizer(f" {o_text.split()[1]}", return_tensors="pt", padding=True, truncation=True)["input_ids"]
                else:
                    if "other_prompt2" in sys.argv:
                        n_text = text[:t2st[j].start_char] + "$" + text[t2st[j].end_char:]
                    else:
                        n_text = text[:t2st[j].start_char] + "@" + text[t2st[j].end_char:]
                    inputs2 = tokenizer(n_text, return_tensors="pt", padding=True, truncation=True)
                    labels = tokenizer(t2st[j].text, return_tensors="pt", padding=True, truncation=True)["input_ids"]
            else:
                _labels = torch.tensor([inputs['input_ids'][0, j].item(), tokenizer.eos_token_id]).unsqueeze(0)
                inputs2['input_ids'] = torch.cat([torch.clone(inputs.input_ids)[:, :-1], _labels], dim=1)
                labels = torch.clone(inputs2['input_ids'])
                inputs2['input_ids'][0, j] = extra_ids[0]
                labels[0, :-2] = -100
            with torch.no_grad():
                out2 = model(inputs2['input_ids'].to(dev), labels=labels.to(dev))

            # mask both, predict a
            if not llama:
                if fixed_pos is not None:
                    s_text = o_text.split()
                    if "other_prompt2" in sys.argv:
                        s_text[0] = "^"
                        s_text[1] = "$"
                    else:
                        s_text[0] = "%"
                        s_text[1] = "@"
                    n_text = text.replace(o_text, " ".join(s_text))
                    inputs2 = tokenizer(n_text, return_tensors="pt", padding=True, truncation=True)
                    labels = tokenizer(f" {o_text.split()[0]}", return_tensors="pt", padding=True, truncation=True)["input_ids"]
                else:
                    if "other_prompt2" in sys.argv:
                        n_text = text[:t2st[i].start_char] + "^ $" + text[t2st[j].end_char:]
                    else:
                        n_text = text[:t2st[i].start_char] + "% @" + text[t2st[j].end_char:]
                    inputs2 = tokenizer(n_text, return_tensors="pt", padding=True, truncation=True)
                    labels = tokenizer(t2st[i].text, return_tensors="pt", padding=True, truncation=True)["input_ids"]
            else:
                _labels = torch.tensor([inputs['input_ids'][0, i].item(), tokenizer.eos_token_id]).unsqueeze(0)
                inputs2['input_ids'] = torch.cat([torch.clone(inputs.input_ids)[:, :-1], _labels], dim=1)
                labels = torch.clone(inputs2['input_ids'])
                inputs2['input_ids'][0, i] = extra_ids[0]
                inputs2['input_ids'][0, j] = extra_ids[1]
                labels[0, :-2] = -100
            with torch.no_grad():
                out3_a = model(inputs2['input_ids'].to(dev), labels=labels.to(dev))
            # mask both, predict b
            if not llama:
                if fixed_pos is not None:
                    s_text = o_text.split()
                    if "other_prompt2" in sys.argv:
                        s_text[0] = "^"
                        s_text[1] = "$"
                    else:
                        s_text[0] = "@"
                        s_text[1] = "%"
                    n_text = text.replace(o_text, " ".join(s_text))
                    inputs2 = tokenizer(n_text, return_tensors="pt", padding=True, truncation=True)
                    labels = tokenizer(f" {o_text.split()[0]}", return_tensors="pt", padding=True, truncation=True)["input_ids"]
                else:
                    if "other_prompt2" in sys.argv:
                        n_text = text[:t2st[i].start_char] + "^ $" + text[t2st[j].end_char:]
                    else:
                        n_text = text[:t2st[i].start_char] + "% @" + text[t2st[j].end_char:]
                    inputs2 = tokenizer(n_text, return_tensors="pt", padding=True, truncation=True)
                    labels = tokenizer(t2st[j].text, return_tensors="pt", padding=True, truncation=True)["input_ids"]
            else:
                _labels = torch.tensor([inputs['input_ids'][0, j].item(), tokenizer.eos_token_id]).unsqueeze(0)
                inputs2['input_ids'] = torch.cat([torch.clone(inputs.input_ids)[:, :-1], _labels], dim=1)
                labels = torch.clone(inputs2['input_ids'])
                inputs2['input_ids'][0, i] = extra_ids[1]
                inputs2['input_ids'][0, j] = extra_ids[0]
                labels[0, :-2] = -100
            with torch.no_grad():
                out3_b = model(inputs2['input_ids'].to(dev), labels=labels.to(dev))

            # entropies of: w1 only, w2 only, w1|w2, w2|w1
            eos_id = tokenizer.eos_token_id
            eos_probs = []
            ranks = []
            if not llama:
                eos_probs = [float(out3_a.logits[0, -1].log_softmax(dim=-1).cpu().numpy()[eos_id]),
                             float(out.logits[0, -1].log_softmax(dim=-1).cpu().numpy()[eos_id])]
                if "ranks" in sys.argv:
                    _ranks = (-out3_a.logits[0, -2]).cpu().numpy().argsort().argsort()
                    _ranks1 = (-out.logits[0, -2]).cpu().numpy().argsort().argsort()
                    _ranks2 = (-out3_a.logits[0, -1]).cpu().numpy().argsort().argsort()
                    ranks = [int(_ranks2[eos_id]), int(_ranks[inputs['input_ids'][0, i].item()]), int(_ranks1[inputs['input_ids'][0, j].item()]), len(_ranks)]

                if return_probs:
                    probs.append([out3_a.logits[0, -2].log_softmax(dim=-1).cpu().numpy(), out3_a.logits[0, -1].log_softmax(dim=-1).cpu().numpy(),
                                 out3_b.logits[0, -2].log_softmax(dim=-1).cpu().numpy(), out3_b.logits[0, -1].log_softmax(dim=-1).cpu().numpy(),
                                 out.logits[0, -2].log_softmax(dim=-1).cpu().numpy(), out.logits[0, -1].log_softmax(dim=-1).cpu().numpy(),
                                 out2.logits[0, -2].log_softmax(dim=-1).cpu().numpy(), out2.logits[0, -1].log_softmax(dim=-1).cpu().numpy()])

                    ids.append([inputs['input_ids'][0, i].item(), inputs['input_ids'][0, j].item(),
                                 inputs['input_ids'][0, i].item(), inputs['input_ids'][0, j].item(),
                                 inputs['input_ids'][0, i].item(), inputs['input_ids'][0, j].item(),
                                 inputs['input_ids'][0, i].item(), inputs['input_ids'][0, j].item()])
                else:
                    entropies = [float(H(out3_a.logits[0, -2]).cpu().numpy()) + float(H(out3_a.logits[0, -1]).cpu().numpy()),
                                 float(H(out3_b.logits[0, -2]).cpu().numpy()) + float(H(out3_b.logits[0, -1]).cpu().numpy()),
                                 float(H(out.logits[0, -2]).cpu().numpy()) + float(H(out.logits[0, -1]).cpu().numpy()),
                                 float(H(out2.logits[0, -2]).cpu().numpy()) + float(H(out2.logits[0, -1]).cpu().numpy())]
            else:
                eos_probs = [float(out3_a.logits[0, -2].log_softmax(dim=-1).cpu().numpy()[eos_id]),
                             float(out.logits[0, -2].log_softmax(dim=-1).cpu().numpy()[eos_id])]
                if "ranks" in sys.argv:
                    _ranks = (-out3_a.logits[0, -3]).cpu().numpy().argsort().argsort()
                    _ranks1 = (-out.logits[0, -3]).cpu().numpy().argsort().argsort()
                    _ranks2 = (-out3_a.logits[0, -2]).cpu().numpy().argsort().argsort()
                    ranks = [int(_ranks2[eos_id]), int(_ranks[inputs['input_ids'][0, i].item()]), int(_ranks1[inputs['input_ids'][0, j].item()]), len(_ranks)]

                if return_probs:
                    probs.append([out3_a.logits[0, -3].log_softmax(dim=-1).cpu().numpy(), out3_a.logits[0, -2].log_softmax(dim=-1).cpu().numpy(),
                                 out3_b.logits[0, -3].log_softmax(dim=-1).cpu().numpy(), out3_b.logits[0, -2].log_softmax(dim=-1).cpu().numpy(),
                                 out.logits[0, -3].log_softmax(dim=-1).cpu().numpy(), out.logits[0, -2].log_softmax(dim=-1).cpu().numpy(),
                                 out2.logits[0, -3].log_softmax(dim=-1).cpu().numpy(), out2.logits[0, -2].log_softmax(dim=-1).cpu().numpy()])
                    ids.append([inputs['input_ids'][0, i].item(), inputs['input_ids'][0, j].item(),
                                 inputs['input_ids'][0, i].item(), inputs['input_ids'][0, j].item(),
                                 inputs['input_ids'][0, i].item(), inputs['input_ids'][0, j].item(),
                                 inputs['input_ids'][0, i].item(), inputs['input_ids'][0, j].item()])
                else:
                    entropies = [float(H(out3_a.logits[0, -3]).cpu().numpy()) + float(H(out3_a.logits[0, -2]).cpu().numpy()),
                                 float(H(out3_b.logits[0, -3]).cpu().numpy()) + float(H(out3_b.logits[0, -2]).cpu().numpy()),
                                 float(H(out.logits[0, -3]).cpu().numpy()) + float(H(out.logits[0, -2]).cpu().numpy()),
                                 float(H(out2.logits[0, -3]).cpu().numpy()) + float(H(out2.logits[0, -2]).cpu().numpy())]
            if not return_probs:
                _p12 = -out3_a.loss - out2.loss
                _p21 = -out3_b.loss - out.loss
                _pair_prob = _p12
                if fixed_pos is not None and "flan" in sys.argv:
                    id1, id2 = [0, 0]
                else:
                    id1, id2 = inputs['input_ids'][0, i].item(), inputs['input_ids'][0, j].item()
                if "ranks" not in sys.argv:
                    l.append((float(_pair_prob), float(_p12), float(_p21), [id1, id2], entropies, dep, eos_probs, [ti, t_ii]))
                else:
                    l.append((float(_pair_prob), float(_p12), float(_p21), [id1, id2], entropies, dep, eos_probs, [ti, t_ii], ranks))
    if return_probs:
        return np.array(probs), np.array(ids)
    return l


def mlm_pair_probabilities(texts, model, tokenizer, temperature=1., deps=False, deps2=False, fixed_pos=None,
                           return_probs=False):
    """
    Measures probabilities for adjacent token pairs in the given texts
    :param texts: list of strings
    :param model:
    :param tokenizer:
    :return:
    """
    deps3 = "deps3" in sys.argv
    if deps2 or deps3:
        deps = True
    # if deps:
    import spacy
    nlp = spacy.load("en_core_web_md")
    # POS_tags = ["ADJ", "ADV", "NOUN", "PRON", "PROPN", "VERB"]
    # print(f"Using only {POS_tags}")
    dep_tags = ["amod", "compound", "nummod", "poss"]
    print(f"Using only {dep_tags}")
    print("For deps3 avoiding all backward tags")
    print("Avoiding spacy stopwords, punctuation, digits")

    l = []
    probs = []
    ids = []
    model.to(dev)

    # for text in tqdm.tqdm(texts[:5000] if (not deps or deps3) else texts):
    if return_probs:
        r_texts = texts[:10]
    else:
        r_texts = texts if (not deps and not deps3) else texts[:5000] if deps3 else texts
    for ti, text in tqdm.tqdm(enumerate(r_texts)):
        if fixed_pos is None and "extra_noise" not in sys.argv and "total_noise" not in sys.argv:
            if "non_ascii" in sys.argv:
                if len(text) < 50 or text.isascii():
                    continue
            else:
                if len(text) < 50 or not text.isascii():
                    continue
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs2 = BatchEncoding(inputs)

        # if deps:
        if "extra_noise" not in sys.argv and "total_noise" not in sys.argv:
            doc = nlp(text)
            t2st = {i: doc.char_span(s, e, alignment_mode="contract") for i, (s, e) in enumerate(inputs.encodings[0].offsets)}
        def is_stop(i):
            # check if words are problematic
            if t2st[i] is None or t2st[i+1] is None:  # covers token that are not full words
                return True
            if t2st[i][0].is_stop or t2st[i+1][0].is_stop or t2st[i][0].is_digit or t2st[i+1][0].is_digit \
                    or t2st[i][0].is_punct or t2st[i+1][0].is_punct:
                return True
        def is_skip(i, all_tags=False):
            if not deps2:  # an arc going "backwards"
                return t2st[i][0].head.i == t2st[i+1][0].i and t2st[i+1][0].i > t2st[i][0].i and \
                       t2st[i][0].n_rights + t2st[i][0].n_lefts == 0 and (all_tags or t2st[i][0].dep_ in dep_tags)
            else:
                return t2st[i+1][0].head.i == t2st[i][0].i and t2st[i+1][0].i > t2st[i][0].i and t2st[i+1][0].n_rights + t2st[i+1][0].n_lefts == 0

        if fixed_pos is None:
            token_range = range(1, len(inputs.input_ids[0]) - 2)
        else:
            token_range = range(fixed_pos, fixed_pos+1)
        for i in token_range:
            dep = None
            t_ii = 0
            if "extra_noise" not in sys.argv and "total_noise" not in sys.argv:
                if fixed_pos is None:
                    if is_stop(i):
                        continue
                    if deps and deps3 and is_skip(i, all_tags=True):
                        continue
                    elif deps and not deps3 and not is_skip(i):
                        continue
                if is_skip(i, all_tags=True):
                    dep = t2st[i][0].dep_
                t_ii = t2st[i][0].i
            # mask first
            id1, id2 = inputs['input_ids'][0, i], inputs['input_ids'][0, i+1]
            inputs2['input_ids'] = torch.clone(inputs.input_ids)
            labels = torch.full_like(inputs2['input_ids'], -100)
            labels[0, i] = inputs2['input_ids'][0, i].item()
            inputs2['input_ids'][0, i] = tokenizer.mask_token_id
            with torch.no_grad():
                out = model(**inputs2.to(dev), labels=labels.to(dev))

            # mask second
            inputs2['input_ids'] = torch.clone(inputs.input_ids)
            labels = torch.full_like(inputs2['input_ids'], -100)
            labels[0, i+1] = inputs2['input_ids'][0, i+1].item()
            inputs2['input_ids'][0, i+1] = tokenizer.mask_token_id
            with torch.no_grad():
                out2 = model(**inputs2.to(dev), labels=labels.to(dev))

            # mask both
            inputs2['input_ids'] = torch.clone(inputs.input_ids)
            labels_a = torch.full_like(inputs2['input_ids'], -100)
            labels_b = torch.full_like(inputs2['input_ids'], -100)
            labels_j = torch.full_like(inputs2['input_ids'], -100)
            labels_a[0, i] = inputs2['input_ids'][0, i].item()
            labels_b[0, i+1] = inputs2['input_ids'][0, i+1].item()
            labels_j[0, i] = inputs2['input_ids'][0, i].item()
            labels_j[0, i+1] = inputs2['input_ids'][0, i+1].item()
            inputs2['input_ids'][0, i] = tokenizer.mask_token_id
            inputs2['input_ids'][0, i+1] = tokenizer.mask_token_id
            with torch.no_grad():
                out3_a = model(**inputs2.to(dev), labels=labels_a.to(dev))
                out3_b = model(**inputs2.to(dev), labels=labels_b.to(dev))
                # joint
                out4 = model(**inputs2.to(dev), labels=labels_j.to(dev))

            logits = out.logits / temperature
            logits2 = out2.logits / temperature
            logits4 = out4.logits / temperature
            if return_probs:
                probs.append(np.array([logits4[0, i].cpu().numpy(), logits4[0, i+1].cpu().numpy(),
                         logits[0, i].cpu().numpy(), logits2[0, i+1].cpu().numpy()]))
                ids.append([inputs['input_ids'][0, i], inputs['input_ids'][0, i+1],
                            inputs['input_ids'][0, i], inputs['input_ids'][0, i+1]])
            else:
                ranks =[]
                if "ranks" in sys.argv:
                    _ranks = (-out3_b.logits[0, i+1]).cpu().numpy().argsort().argsort()  # first token
                    _ranks1 = (-out.logits[0, i]).cpu().numpy().argsort().argsort()  # second token
                    ranks = [int(_ranks[inputs['input_ids'][0, i].item()]), int(_ranks1[inputs['input_ids'][0, i+1].item()]), len(_ranks)]

                # entropies of: w1 only, w2 only, w1|w2, w2|w1
                entropies = [float(H(logits4[0, i]).cpu().numpy()), float(H(logits4[0, i + 1]).cpu().numpy()),
                             float(H(logits[0, i]).cpu().numpy()), float(H(logits2[0, i + 1]).cpu().numpy())]
                # p12 = probs4[0, i][inputs['input_ids'][0, i]] + probs2[0, i+1][inputs['input_ids'][0, i+1]]
                _p12 = -out3_a.loss - out2.loss
                # p21 = probs4[0, i+1][inputs['input_ids'][0, i+1]] + probs[0, i][inputs['input_ids'][0, i]]
                _p21 = -out3_b.loss - out.loss
                # pair_prob = probs4[0, 1][inputs['input_ids'][0, i]] + probs4[0, 2][inputs['input_ids'][0, i+1]]
                _pair_prob = -out4.loss * 2
                if "ranks" not in sys.argv:
                    l.append((float(_pair_prob), float(_p12), float(_p21), [id1.item(), id2.item()], entropies, dep, [ti, t_ii]))
                else:
                    l.append((float(_pair_prob), float(_p12), float(_p21), [id1.item(), id2.item()], entropies, dep, [ti, t_ii], ranks))
    if return_probs:
        return np.array(probs), np.array(ids)
    return l

# ******************************
#

def to_bin(p):
    from bisect import bisect
    intervals = [0, 0.25, 0.5, 1, 2, 3, 5, 7.5, 10, 15, 20, 30, 40, 50, 75, 100, 125, 150]
    return intervals[bisect(intervals, p)-1]


def boxplot(name="", ylim=None, bins=None, dataset="wikitext-2", by_prob=False):
    # box plot of absolute log-distance
    import json

    with open(args.base_path + f'results/{dataset}_scores_{name}.json', 'r') as file:
        pp = json.load(file)
    import pandas as pd

    if by_prob:
        df = pd.DataFrame({"p12": [p[1] for p in pp], "p21": [p[2] for p in pp], f"p-{name}": [to_bin(-p[0]) for p in pp]})
    else:
        df = pd.DataFrame({"p12": [p[1] for p in pp], "p21": [p[2] for p in pp], "p": [p[0] for p in pp]})

    import matplotlib.pyplot as plt

    if bins is None:
        df[f'i10000-{dataset}-{name}'] = np.arange(len(pp)) // 10000
    else:
        df[f'i10000-{dataset}-{name}'] = (np.arange(len(pp)) * bins) // len(pp)
    df['ratio'] = (df['p21'] - df['p12']).abs()
    if by_prob:
        df.boxplot(column=['ratio'], by=f"p-{name}", sym="")
    else:
        df.boxplot(column=['ratio'], by=f'i10000-{dataset}-{name}', sym="")

    if ylim is not None:
        plt.ylim(*ylim)

    def rand(df):
        df['p21'] = df['p21'].sample(frac=1).to_numpy()
        return (df['p12'] - df['p21']).abs().mean()

    def mean(df):
        return df[f"p-{name}"].mean()

    if by_prob:
        df_g = df.groupby(f"p-{name}", group_keys=True)
    else:
        df_g = df.groupby(f'i10000-{dataset}-{name}', group_keys=True)
    df_means = df_g.apply(mean)
    dfr = df_g.apply(rand)
    plt.plot(range(1, len(dfr) + 1), dfr.to_numpy(), color='red')
    plt.show()
    print(df_means)

    # plot of js distances
    from scipy.spatial import distance
    def js(df):
        return distance.jensenshannon(np.exp(df['p12']), np.exp(df['p21']), base=2)
    df[f'js100-{dataset}-{name}'] = np.arange(len(pp)) // 100
    dfg = df.groupby(f'js100-{dataset}-{name}', group_keys=True).apply(js)
    dfg.plot()
    plt.show()


def test_top(dataset_name="new_news", size="small", version=""):
    print("testing...")
    import pandas as pd
    import json
    deps = "deps" in sys.argv
    l1_2 = "l1_2" in sys.argv
    l2_1 = "l2_1" in sys.argv
    deps3 = "deps3" in sys.argv

    if "t5" in sys.argv:
        name = f"t5-{size}"
        if version.startswith("v1_1") or version.startswith("flan"):
            _size = size
            if size == "3b":
                _size = "xl"
            if size == "11b":
                _size = "xxl"
            if version.startswith("v1_1"):
                name = f"google/t5-v1_1-{_size}"
            elif version.find("ul2") > -1:
                name = f"google/{version}"
            else:
                name = f"google/flan-t5-{_size}"
        if version.find("ul2") > -1:
            name = f"google/{version}"
            from transformers import BitsAndBytesConfig, AutoModelForSeq2SeqLM
            double_quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForSeq2SeqLM.from_pretrained(name, device_map="auto",
                                                          quantization_config=double_quant_config)
        elif size == "11b" or size == "xxl":
            torch.cuda.empty_cache()
            model = T5ForConditionalGeneration.from_pretrained(name, device_map="auto", load_in_4bit=True)
        else:
            model = T5ForConditionalGeneration.from_pretrained(name)
        tokenizer = AutoTokenizer.from_pretrained(name, legacy=False)

    model.eval()
    if "flan-ul2" not in sys.argv and "ul2" not in sys.argv and "11b" not in sys.argv:
        model.to(dev)
    extra_ids = tokenizer.get_sentinel_token_ids()
    extra_ids.sort(reverse=True)
    underscore = tokenizer.convert_tokens_to_ids('▁')
    underscore_str = '▁'

    with open(args.base_path + f'results/{dataset_name}_scores_{name.split("/")[-1]}'
              f'{"_deps" if deps else ""}{"_l1_2" if l1_2 else ""}{"_l2_1" if l2_1 else ""}.json', 'r') as infile:
        ppt5=json.load(infile)
    dict_t5_v11 = {"p_i":[_pp[0] for _pp in ppt5],
        "p12":[_pp[1] for _pp in ppt5],
        "p21":[_pp[2] for _pp in ppt5],
        "p12x":[_pp[6] for _pp in ppt5],
        "p21x":[_pp[7] for _pp in ppt5],
        "tokens": [tokenizer.convert_ids_to_tokens(_pp[3]) for _pp in ppt5],
        "ids": [_pp[5] for _pp in ppt5]
        }
    df_t5 = pd.DataFrame(dict_t5_v11)
    df_t5['diffs'] = df_t5['p12'] - df_t5['p21']
    df_t5['abs_diffs'] = (df_t5['p12'] - df_t5['p21']).abs()

    best_10 = df_t5.sort_values(by=['abs_diffs'])['ids'][:10]
    best_10_tokens = df_t5.sort_values(by=['abs_diffs'])['tokens'][:10]
    worst_10 = df_t5.sort_values(by=['abs_diffs'])['ids'][-10:]
    worst_10_tokens = df_t5.sort_values(by=['abs_diffs'])['tokens'][-10:]

    with open(args.base_path + f'data/news-2.7.2023.json', 'r') as file:
        texts = [t['text'] for t in json.load(file) if t['title'].find("Subscribe") == -1]
    with open(args.base_path + f'data/news-6.7.2023.json', 'r') as file:
        texts = texts + [t['text'] for t in json.load(file) if t['title'].find("Subscribe") == -1]

    print("top and worst 10 cases:")
    for set in [zip(best_10, best_10_tokens), zip(worst_10, worst_10_tokens)]:
        print("\n********")
        for (t, i), tokens in set:
            print(tokens)
            j = i + 2
            if l2_1:
                j = i + 3
            text = texts[t]
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            inputs2 = BatchEncoding(inputs)
            r_inputs2 = BatchEncoding(inputs)
            # mask both
            inputs2['input_ids'] = torch.clone(inputs.input_ids)
            r_inputs2['input_ids'] = torch.clone(inputs.input_ids)
            labels = torch.full_like(inputs2['input_ids'], -100)
            r_labels = torch.full_like(inputs2['input_ids'], -100)
            if not l1_2 and not l2_1:
                labels[0, :6] = torch.tensor([extra_ids[0], inputs2['input_ids'][0, i].item(), extra_ids[1],
                                              inputs2['input_ids'][0, j].item(), extra_ids[2], -100])
                r_labels[0, :6] = torch.tensor([extra_ids[0], inputs2['input_ids'][0, j].item(), extra_ids[1],
                                                inputs2['input_ids'][0, i].item(), extra_ids[2], -100])
                inputs2['input_ids'][0, i] = extra_ids[0]
                inputs2['input_ids'][0, j] = extra_ids[1]
                r_inputs2['input_ids'] = torch.clone(inputs2.input_ids)
                r_inputs2['input_ids'][0, i] = extra_ids[1]
                r_inputs2['input_ids'][0, j] = extra_ids[0]
            elif l1_2:

                labels[0, :7] = torch.tensor([extra_ids[0], inputs2['input_ids'][0, i].item(), extra_ids[1],
                                              inputs2['input_ids'][0, j].item(),
                                              inputs2['input_ids'][0, j + 1].item(),
                                              extra_ids[2], -100])
                r_labels[0, :7] = torch.tensor([extra_ids[0], inputs2['input_ids'][0, j].item(),
                                                inputs2['input_ids'][0, j + 1].item(), extra_ids[1],
                                                inputs2['input_ids'][0, i].item(),
                                                extra_ids[2], -100])
                inputs2['input_ids'][0, j + 1:-1] = inputs['input_ids'][0, j + 2:]
                inputs2['input_ids'][0, i] = extra_ids[0]
                inputs2['input_ids'][0, j] = extra_ids[1]
                r_inputs2['input_ids'] = torch.clone(inputs2.input_ids)
                r_inputs2['input_ids'][0, i] = extra_ids[1]
                r_inputs2['input_ids'][0, j] = extra_ids[0]
            elif l2_1:
                labels[0, :7] = torch.tensor([extra_ids[0], inputs2['input_ids'][0, i].item(),
                                              inputs2['input_ids'][0, i + 1].item(), extra_ids[1],
                                              inputs2['input_ids'][0, j].item(), extra_ids[2], -100])
                r_labels[0, :7] = torch.tensor([extra_ids[0], inputs2['input_ids'][0, j].item(), extra_ids[1],
                                                inputs2['input_ids'][0, i].item(),
                                                inputs2['input_ids'][0, i + 1].item(),
                                                extra_ids[2], -100])
                inputs2['input_ids'][0, i + 1:-1] = inputs['input_ids'][0, i + 2:]
                inputs2['input_ids'][0, i] = extra_ids[0]
                inputs2['input_ids'][0, j - 1] = extra_ids[1]
                r_inputs2['input_ids'] = torch.clone(inputs2.input_ids)
                r_inputs2['input_ids'][0, i] = extra_ids[1]
                r_inputs2['input_ids'][0, j - 1] = extra_ids[0]
            with torch.no_grad():
                out = model(**inputs2.to(dev), labels=labels.to(dev))
                r_out = model(**r_inputs2.to(dev), labels=r_labels.to(dev))
                out_x = model(**r_inputs2.to(dev), labels=labels.to(dev))
                r_out_x = model(**inputs2.to(dev), labels=r_labels.to(dev))

            logits = out.logits
            probs = logits.log_softmax(dim=-1).detach().cpu().numpy()
            r_logits = r_out.logits
            r_probs = r_logits.log_softmax(dim=-1).detach().cpu().numpy()
            logits_x = out_x.logits
            probs_x = logits_x.log_softmax(dim=-1).detach().cpu().numpy()
            r_logits_x = r_out_x.logits
            r_probs_x = r_logits_x.log_softmax(dim=-1).detach().cpu().numpy()

            print(tokenizer.decode(inputs2['input_ids'][0]))
            print(tokenizer.decode(labels[0][:5]))
            print(out.loss)
            print(probs[0, range(6), labels.numpy()[0][:6]])
            print(tokenizer.decode(r_inputs2['input_ids'][0]))
            print(tokenizer.decode(r_labels[0][:5]))
            print(r_out.loss)
            print(r_probs[0, range(6), r_labels.numpy()[0][:6]])

            print("reversing:")
            print(out_x.loss)
            print(probs_x[0, range(6), labels.numpy()[0][:6]])
            print(r_out_x.loss)
            print(r_probs_x[0, range(6), r_labels.numpy()[0][:6]])

            print("\n")
    print("\n\n")


# ****************************
# for rank evaluation

def load_pp(dataset, name):
    f_path = args.base_path + f'results/{dataset}_scores_{name}_ranks.json'
    print(f_path)
    with open(f_path, 'r') as file:
        pp = json.load(file)
    return pp

def load_ranks():
    d = {}
    for dataset in ["wikitext-2"]:
        for name in ["electra-base-generator_", "electra-large-generator_", "electra-small-generator_",
                     "flan-t5-small_", "flan-t5-base_", "flan-t5-large_", "flan-t5-xl_"
            , "flan-t5-xxl_", "Llama-2-7b-hf_", "Llama-2-7b-chat-hf_", "Llama-2-13b-hf_", "Llama-2-13b-chat-hf_",
                     "Llama-2-70b-hf_", "Llama-2-70b-chat-hf_", "roberta-base_", "roberta-large_"
            , "xlm-roberta-base_", "xlm-roberta-large_"]:
            # print(f"\n\n{dataset} {name}_ranks")
            try:
                pp = load_pp(dataset, name)
                print(len(pp))
                print(np.mean([len(p) for p in pp]))
                if len(pp[-1][-1]) == 4:
                    d[name] = {"eos_rank": [np.mean([p[-1][0] for p in pp])],
                               "token_rank1": [np.mean([p[-1][1] for p in pp])],
                               "token_rank2": [np.mean([p[-1][2] for p in pp])], "vocab": pp[-1][-1][-1]}
                else:
                    d[name] = {"token_rank1": [np.mean([p[-1][0] for p in pp])],
                               "token_rank2": [np.mean([p[-1][1] for p in pp])], "vocab": pp[-1][-1][-1]}
            except:
                print("doesn't exist")
                continue
    return d

# ****************************

def main():
    t_list = "t_list" in sys.argv
    single_token = "single_token" in sys.argv
    entropies = "entropies" in sys.argv
    dataset = args.dataset

    version = ""
    if "v1_1" in sys.argv:
        version = "v1_1"
    if "flan" in sys.argv:
        version = "flan"
    if "ul2" in sys.argv:
        version = "ul2"
    if "flan-ul2" in sys.argv:
        version = "flan-ul2"

    size = args.model_size

    with torch.no_grad():
        if not "synthetic" in sys.argv:
            if "test" not in sys.argv or "ttest" in sys.argv:
                on_data_calibration(dataset_name=dataset, size=size, version=version)
            if "test" in sys.argv or "ttest" in sys.argv:
                test_top(dataset_name=dataset, size=size, version=version)
        else:
            tokenizer = on_data_calibration(size=size, version=version, return_tokenizer=True)
            on_data_calibration(dataset_name=dataset, size=size, version=version, return_tokenizer=False, tokenizer=tokenizer)

    if entropies:
        with torch.no_grad():
            all_entropies = on_data_calibration(dataset_name=dataset, size=size, version=version)
        print("All entropies:")
        print(all_entropies)

    print("Done")


if __name__ == "__main__":

    print("\n\n\n**********************************************")
    print(sys.argv)
    sys.stdout.flush()
    main()



