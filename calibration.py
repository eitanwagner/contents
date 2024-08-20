
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

def _nonadjacent_mask_filling(model=None, tokenizer=None, w1=None, w2=None, batch_size=1):
    """

    :param model:
    :param tokenizer:
    :param w1: a list of length batch_size
    :param w2:
    :param batch_size:
    :return:
    """
    input_ids = tokenizer(["The <extra_id_0> is <extra_id_1>."] * batch_size, return_tensors="pt", padding=True).input_ids
    input_ids2 = tokenizer([f"The {_w1} is <extra_id_1>." for _w1 in w1], return_tensors="pt", padding=True).input_ids
    input_ids3 = tokenizer([f"The <extra_id_0> is {_w2}." for _w2 in w2], return_tensors="pt", padding=True).input_ids
    labels = tokenizer([f"<extra_id_0> {_w1} <extra_id_1> {_w2} <extra_id_2>" for _w1, _w2 in zip(w1, w2)], return_tensors="pt", padding=True).input_ids
    labels2 = tokenizer([f"<extra_id_1> {_w2} <extra_id_2>" for _w2 in w2], return_tensors="pt", padding=True).input_ids
    labels3 = tokenizer([f"<extra_id_0> {_w1} <extra_id_1>" for _w1 in w1], return_tensors="pt", padding=True).input_ids
    # the forward function automatically creates the correct decoder_input_ids
    with torch.no_grad():
        out = model(input_ids=input_ids, labels=labels)  # is this enough or should we look at the logits??
        out2 = model(input_ids=input_ids2, labels=labels2)  # is this enough or should we look at the logits??
        out3 = model(input_ids=input_ids3, labels=labels3)  # is this enough or should we look at the logits??
    # loss = out.loss
    # print(labels[0].numpy())
    probs = out.logits.log_softmax(dim=-1).numpy()
    probs2 = out2.logits.log_softmax(dim=-1).numpy()
    probs3 = out3.logits.log_softmax(dim=-1).numpy()

    # dog_id = tokenizer(w1).input_ids[0]
    w1_ids = [tokenizer(_w1).input_ids[:-1] for _w1 in w1]
    w2_ids = [tokenizer(_w2).input_ids[:-1] for _w2 in w2]

    scores = {"independent":
                  np.array([float(probs[i, np.arange(1, 1+ len(w1_ids[i])), w1_ids[i]].sum(dtype=float)
                                  + probs[i, np.arange(2 + len(w1_ids[i]), 2 + len(w1_ids[i]) + len(w2_ids[i])), w2_ids[
                      i]].sum(dtype=float))
                            for i in range(batch_size)]),
              "w1 first":
                  np.array([float(probs[i, np.arange(1, 1 + len(w1_ids[i])), w1_ids[i]].sum(dtype=float)
                                  + probs2[i, np.arange(1, 1 + len(w2_ids[i])), w2_ids[i]].sum(dtype=float))
                            for i in range(batch_size)]),
              # float(probs[np.arange(1, 1+len(w1_ids)), w1_ids].sum(dtype=float) + probs2[np.arange(1, 1+len(w2_ids)), w2_ids].sum(dtype=float)),
              "w2 first":
                  np.array([float(
                      probs[i, np.arange(2 + len(w1_ids[i]), 2 + len(w1_ids[i]) + len(w2_ids[i])), w2_ids[i]].sum(
                          dtype=float)
                      + probs3[i, np.arange(1, 1 + len(w1_ids[i])), w1_ids[i]].sum(dtype=float))
                            for i in range(batch_size)]),
              # float(probs[np.arange(3, 3+len(w2_ids)), w2_ids].sum(dtype=float) + probs3[np.arange(1, 1+len(w1_ids)), w1_ids].sum(dtype=float))}
              }

    return scores


def topical_lm(logits, psi, topic):
    # use before softmax
    return logits + psi[topic]


def _adjacent_mask_filling(model=None, model2=None, tokenizer=None, w1=None, w2=None, batch_size=1, template="good",
                           noun_topics=None, adj_topics=None, i=None, j=None, psi=None, temperature=None,
                           temperature_list=False):
    """

    :param model:
    :param tokenizer:
    :param w1: a list of length batch_size
    :param w2:
    :param batch_size:
    :return:
    """
    input_ids = tokenizer([f"The <extra_id_0> <extra_id_1> is {template}."] * batch_size, return_tensors="pt",
                          padding=True).input_ids
    input_ids0 = tokenizer([f"The <extra_id_0> is {template}."] * batch_size, return_tensors="pt",
                           padding=True).input_ids
    input_ids2 = tokenizer([f"The {_w1} <extra_id_1> is {template}." for _w1 in w1], return_tensors="pt",
                           padding=True).input_ids
    input_ids3 = tokenizer([f"The <extra_id_0> {_w2} is {template}." for _w2 in w2], return_tensors="pt",
                           padding=True).input_ids
    labels = tokenizer([f"<extra_id_0> {_w1} <extra_id_1> {_w2} <extra_id_2>" for _w1, _w2 in zip(w1, w2)],
                       return_tensors="pt", padding=True).input_ids
    labels0 = tokenizer([f"<extra_id_0> {_w1} {_w2} <extra_id_1>" for _w1, _w2 in zip(w1, w2)], return_tensors="pt",
                        padding=True).input_ids
    labels2 = tokenizer([f"<extra_id_1> {_w2} <extra_id_2>" for _w2 in w2], return_tensors="pt", padding=True).input_ids
    labels3 = tokenizer([f"<extra_id_0> {_w1} <extra_id_1>" for _w1 in w1], return_tensors="pt", padding=True).input_ids

    w1_ids = [tokenizer(_w1).input_ids[:-1] for _w1 in w1]
    w2_ids = [tokenizer(_w2).input_ids[:-1] for _w2 in w2]

    with torch.no_grad():
        out = model(input_ids=input_ids.to(dev), labels=labels.to(dev))
        out0 = model(input_ids=input_ids0.to(dev), labels=labels0.to(dev))
        out2 = model(input_ids=input_ids2.to(dev), labels=labels2.to(dev))
        out3 = model(input_ids=input_ids3.to(dev), labels=labels3.to(dev))
    # loss = out.loss
    # print(labels[0].numpy())
    logits = out.logits
    logits0 = out0.logits
    logits2 = out2.logits
    logits3 = out3.logits
    if noun_topics is not None:
        if noun_topics[i][0] > -np.inf:
            for k in range(batch_size):
                m = np.argmax(noun_topics[i])
                # _noun_topics = torch.tensor(noun_topics[:, m]).unsqueeze(0)
                _noun_topics = torch.tensor(psi[m, :]).unsqueeze(0).to(dev)
                logits[k, 2 + len(w1_ids[k]): 2 + len(w1_ids[k]) + len(w2_ids[k]), :tokenizer.vocab_size] = logits[k,
                                                                                                            2 + len(
                                                                                                                w1_ids[
                                                                                                                    k]): 2 + len(
                                                                                                                w1_ids[
                                                                                                                    k]) + len(
                                                                                                                w2_ids[
                                                                                                                    k]),
                                                                                                            :tokenizer.vocab_size] + _noun_topics.expand(
                    len(w2_ids[k]), -1)
    if adj_topics is not None:
        if adj_topics[j][0] > -np.inf:
            for k in range(batch_size):
                m = np.argmax(adj_topics[j])
                # _adj_topics = torch.tensor(adj_topics[:, m]).unsqueeze(0)
                _adj_topics = torch.tensor(psi[m, :]).unsqueeze(0).to(dev)
                logits[k, 1: 1 + len(w1_ids[k]), :tokenizer.vocab_size] = logits[k, 1: 1 + len(w1_ids[k]),
                                                                          :tokenizer.vocab_size] + _adj_topics.expand(
                    len(w1_ids[k]), -1)

    if not temperature_list:
        temperatures = [1.] if temperature is None else [temperature]
    else:
        temperatures = temperature
        _scores1 = np.zeros((logits.shape[0], len(temperatures)))
        _scores2 = np.zeros((logits.shape[0], len(temperatures)))

    test = []
    for t_i, t in enumerate(temperatures):
        # if temperature is not None:
        _logits = logits / t
        _logits0 = logits0 / t
        _logits2 = logits2 / t
        _logits3 = logits3 / t

        probs = _logits.log_softmax(dim=-1).cpu().numpy()
        probs0 = _logits0.log_softmax(dim=-1).cpu().numpy()
        probs2 = _logits2.log_softmax(dim=-1).cpu().numpy()
        probs3 = _logits3.log_softmax(dim=-1).cpu().numpy()

        if model2 is not None:
            with torch.no_grad():
                out = model2(input_ids=input_ids.to(dev), labels=labels.to(dev))
                out0 = model2(input_ids=input_ids0.to(dev), labels=labels0.to(dev))
                out2 = model2(input_ids=input_ids2.to(dev), labels=labels2.to(dev))
                out3 = model2(input_ids=input_ids3.to(dev), labels=labels3.to(dev))
            # loss = out.loss
            # print(labels[0].numpy())
            _probs = out.logits.log_softmax(dim=-1).cpu().numpy()
            _probs0 = out0.logits.log_softmax(dim=-1).cpu().numpy()
            _probs2 = out2.logits.log_softmax(dim=-1).cpu().numpy()
            _probs3 = out3.logits.log_softmax(dim=-1).cpu().numpy()

        if model2 is None:
            # test.append(float(probs[i, np.arange(1, 1 + len(w1_ids[i])), w1_ids[i]].sum(dtype=float)))
            scores = {"independent":
                          np.array([float(probs[i, np.arange(1, 1 + len(w1_ids[i])), w1_ids[i]].sum(dtype=float)
                                          + probs[i, np.arange(2 + len(w1_ids[i]), 2 + len(w1_ids[i]) + len(w2_ids[i])),
                                                  w2_ids[i]].sum(dtype=float))
                                    for i in range(batch_size)]),
                      "joint":
                          np.array([float(
                              probs0[i, np.arange(1, 1 + len(w1_ids[i]) + len(w2_ids[i])), w1_ids[i] + w2_ids[i]].sum(
                                  dtype=float))
                                    for i in range(batch_size)]),
                      "w1 first":
                          np.array([float(probs[i, np.arange(1, 1 + len(w1_ids[i])), w1_ids[i]].sum(dtype=float)
                                          + probs2[i, np.arange(1, 1 + len(w2_ids[i])), w2_ids[i]].sum(dtype=float))
                                    for i in range(batch_size)]),
                      # float(probs[np.arange(1, 1+len(w1_ids)), w1_ids].sum(dtype=float) + probs2[np.arange(1, 1+len(w2_ids)), w2_ids].sum(dtype=float)),
                      "w2 first":
                          np.array([float(probs[i, np.arange(2 + len(w1_ids[i]), 2 + len(w1_ids[i]) + len(w2_ids[i])),
                                                w2_ids[i]].sum(dtype=float)
                                          + probs3[i, np.arange(1, 1 + len(w1_ids[i])), w1_ids[i]].sum(dtype=float))
                                    for i in range(batch_size)]),
                      # float(probs[np.arange(3, 3+len(w2_ids)), w2_ids].sum(dtype=float) + probs3[np.arange(1, 1+len(w1_ids)), w1_ids].sum(dtype=float))}
                      }
        else:
            scores = {"independent":
                          np.array([float(probs[i, np.arange(1, 1 + len(w1_ids[i])), w1_ids[i]].sum(dtype=float)
                                          + _probs[
                                              i, np.arange(2 + len(w1_ids[i]), 2 + len(w1_ids[i]) + len(w2_ids[i])),
                                              w2_ids[i]].sum(dtype=float))
                                    for i in range(batch_size)]),
                      "joint":
                          np.array([float(
                              _probs0[i, np.arange(1, 1 + len(w1_ids[i]) + len(w2_ids[i])), w1_ids[i] + w2_ids[i]].sum(
                                  dtype=float))
                                    for i in range(batch_size)]),
                      "w1 first":
                          np.array([float(probs[i, np.arange(1, 1 + len(w1_ids[i])), w1_ids[i]].sum(dtype=float)
                                          + probs2[i, np.arange(1, 1 + len(w2_ids[i])), w2_ids[i]].sum(dtype=float))
                                    for i in range(batch_size)]),
                      # float(probs[np.arange(1, 1+len(w1_ids)), w1_ids].sum(dtype=float) + probs2[np.arange(1, 1+len(w2_ids)), w2_ids].sum(dtype=float)),
                      "w2 first":
                          np.array([float(probs[i, np.arange(2 + len(w1_ids[i]), 2 + len(w1_ids[i]) + len(w2_ids[i])),
                                                w2_ids[i]].sum(dtype=float)
                                          + probs3[i, np.arange(1, 1 + len(w1_ids[i])), w1_ids[i]].sum(dtype=float))
                                    for i in range(batch_size)]),
                      # float(probs[np.arange(3, 3+len(w2_ids)), w2_ids].sum(dtype=float) + probs3[np.arange(1, 1+len(w1_ids)), w1_ids].sum(dtype=float))}
                      }
        if temperature_list:
            _scores1[:, t_i] = scores["w1 first"]
            _scores2[:, t_i] = scores["w2 first"]
    return scores if not temperature_list else (_scores1, _scores2)


def _mask_filling(all_nouns, all_adjs, batch_size, noun_count, adj_count, adjacent=False, size="small", template="good",
                  random=False, noun_topics=None, adj_topics=None, psi=None, temperature=None, temperature_list=False,
                  single_token=False, version=""):
    all_scores_w1 = np.zeros((len(all_nouns), len(all_adjs)))
    # with open(f'/cs/snapless/oabend/eitan.wagner/c alibration/all_scores_w1.npy', 'rb') as f:
    #     all_scores_w1 = np.load(f)
    all_scores_w2 = np.zeros((len(all_nouns), len(all_adjs)))
    # with open(f'/cs/snapless/oabend/eitan.wagner/calibration/all_scores_w2.npy', 'rb') as f:
    #     all_scores_w2 = np.load(f)
    all_scores_i = np.zeros((len(all_nouns), len(all_adjs)))
    # with open(f'/cs/snapless/oabend/eitan.wagner/calibration/all_scores_i.npy', 'rb') as f:
    #     all_scores_i = np.load(f)
    all_scores_j = np.zeros((len(all_nouns), len(all_adjs)))

    _all_scores_w1 = np.zeros((noun_count // 100 + 1, 100, 50))
    _all_scores_w2 = np.zeros((noun_count // 100 + 1, 100, 50))
    _all_scores_i = np.zeros((noun_count // 100 + 1, 100, 50))
    _all_scores_j = np.zeros((noun_count // 100 + 1, 100, 50))
    # with open(f'/cs/snapless/oabend/eitan.wagner/calibration/all_scores_j.npy', 'rb') as f:
    #     all_scores_i = np.load(f)
    w1_sum = -np.inf
    w2_sum = -np.inf
    i_sum = -np.inf
    j_sum = -np.inf

    calculated_nouns = []
    # calculated_nouns = list(set(all_nouns[:8950]))

    if adjacent:
        if temperature_list:
            t_scores = np.zeros((noun_count // 100 + 1, len(temperature)))
            t_scores_r = np.zeros((noun_count // 100 + 1, len(temperature)))

        entropies1 = []
        entropies2 = []
        from scipy.stats import entropy

        print("For the adjacent case w2 (the adjective) is first")
        d = {"1v2": [],
             "1vi": [],
             "1vj": [],
             "2vj": [],
             "ukl1v2": [],
             "ukl1vi": [],
             "ukl1vj": [],
             "ukl2vj": [],
             "r1v2": [],
             "r1vi": [],
             "r1vj": [],
             "r2vj": [],
             "js1v2": [],
             "js1vi": [],
             "js1vj": [],
             "js2vj": [],
             "rjs1v2": [],
             "rjs1vi": [],
             "rjs1vj": [],
             "rjs2vj": []}

    with torch.no_grad():
        if size[0] == "/":  # from path
            path = size
            size = size.split("-")[-1]
        name = f"t5-{size}"
        if version.startswith("v1_1") or version.startswith("flan"):
            _size = size
            if size == "3b":
                _size = "xl"
            if size == "11b":
                _size = "xxl"
            if version.startswith("v1_1"):
                name = f"google/t5-v1_1-{_size}"
            else:
                name = f"google/flan-t5-{_size}"
        if size[0] == "/":  # from path
            # path = size
            # size = size.split("-")[-1]
            tokenizer = T5Tokenizer.from_pretrained(name)
            model = T5ForConditionalGeneration.from_pretrained(path)
            model2 = None
        elif not random:
            model = T5ForConditionalGeneration.from_pretrained(name)
            model2 = None
            tokenizer = T5Tokenizer.from_pretrained(name)
        else:
            from transformers import T5Config
            tokenizer = T5Tokenizer.from_pretrained(name)
            configuration = T5Config.from_pretrained(name)
            model = T5ForConditionalGeneration(config=configuration)
            model2 = T5ForConditionalGeneration(config=configuration)
            model2.to(dev)
        model.to(dev)
        print("batch_size", batch_size)
        print("noun_count", noun_count)
        print("adj_count", adj_count)

        if single_token:
            # noun_num_tokens = [len(tokenizer(n).input_ids[:-1]) for n in all_nouns]
            single_token_nouns = [n for n in all_nouns if len(tokenizer(n).input_ids[:-1]) == 1]
            # adj_num_tokens = [len(tokenizer(a).input_ids[:-1]) for a in all_adjs]
            single_token_adjs = [a for a in all_adjs if len(tokenizer(a).input_ids[:-1]) == 1]
            all_nouns = single_token_nouns
            all_adjs = single_token_adjs
            noun_count = len(single_token_nouns)
            adj_count = len(single_token_adjs)

        # i_range = tqdm.tqdm(enumerate(all_nouns[:noun_count]))
        max_j = len(all_adjs[:adj_count])
        j_range = range(0, max_j, batch_size)
        _i_range = tqdm.tqdm(range(0, noun_count, 100))

        for _e, _i in enumerate(_i_range):
            if temperature_list:
                j_scores1 = np.zeros((100, 50, len(temperature)))
                j_scores2 = np.zeros((100, 50, len(temperature)))
            # if _i > 6000:
            #     print(_i)
            i_range = range(_i, _i + 100)
            if adjacent:
                if _i // 2 + 50 > adj_count:
                    break
                j_range = range(_i // 2, _i // 2 + 50, batch_size)
                max_j = _i // 2 + 50
            for i in i_range:
                w1 = all_nouns[i]
                if w1 in calculated_nouns:
                    continue
                # for j, w2 in enumerate(all_adjs[:adj_count]):

                if not temperature_list:
                    for j in j_range:
                        w2 = all_adjs[j: min(j + batch_size, max_j)]
                        if not adjacent:
                            scores = _nonadjacent_mask_filling(model=model, tokenizer=tokenizer, w1=[w1] * len(w2),
                                                               w2=w2, batch_size=len(w2))
                        else:
                            scores = _adjacent_mask_filling(model=model, model2=model2, tokenizer=tokenizer, w1=w2,
                                                            w2=[w1] * len(w2),
                                                            batch_size=len(w2), template=template,
                                                            noun_topics=noun_topics,
                                                            adj_topics=adj_topics, i=i, j=j, psi=psi,
                                                            temperature=temperature)

                        all_scores_w1[i, j: min(j + batch_size, max_j)] = scores["w1 first"]
                        all_scores_w2[i, j: min(j + batch_size, max_j)] = scores["w2 first"]
                        all_scores_i[i, j: min(j + batch_size, max_j)] = scores["independent"]

                        _all_scores_w1[_e, i - _i, j - _i // 2: min(j + batch_size, max_j) - _i // 2] = scores[
                            "w1 first"]
                        _all_scores_w2[_e, i - _i, j - _i // 2: min(j + batch_size, max_j) - _i // 2] = scores[
                            "w2 first"]
                        _all_scores_i[_e, i - _i, j - _i // 2: min(j + batch_size, max_j) - _i // 2] = scores[
                            "independent"]

                        if adjacent:
                            all_scores_j[i, j: min(j + batch_size, max_j)] = scores["joint"]
                            _all_scores_j[_e, i - _i, j - _i // 2: min(j + batch_size, max_j) - _i // 2] = scores[
                                "joint"]

                    w1_sum = logsumexp([w1_sum, logsumexp(all_scores_w1[i, j: min(j + batch_size, max_j)])])
                    w2_sum = logsumexp([w2_sum, logsumexp(all_scores_w2[i, j: min(j + batch_size, max_j)])])
                    i_sum = logsumexp([i_sum, logsumexp(all_scores_i[i, j: min(j + batch_size, max_j)])])
                    if adjacent:
                        j_sum = logsumexp([j_sum, logsumexp(all_scores_j[i, j: min(j + batch_size, max_j)])])
                else:

                    for j in j_range:
                        w2 = all_adjs[j: min(j + batch_size, max_j)]
                        scores = _adjacent_mask_filling(model=model, model2=model2, tokenizer=tokenizer, w1=w2,
                                                        w2=[w1] * len(w2),
                                                        batch_size=len(w2), template=template, noun_topics=noun_topics,
                                                        adj_topics=adj_topics, i=i, j=j, psi=psi,
                                                        temperature=temperature, temperature_list=temperature_list)
                        j_scores1[i - _i, :, :] = scores[0]
                        j_scores2[i - _i, :, :] = scores[1]

                calculated_nouns = calculated_nouns + [w1]

            if temperature_list:
                for t_i, t in enumerate(temperature):
                    arr = np.copy(j_scores2[:, :, t_i])
                    js1v2 = distance.jensenshannon(np.exp(j_scores1[:, :, t_i].ravel()),
                                                   np.exp(arr.ravel()))
                    np.random.shuffle(arr.ravel())
                    rjs1v2 = distance.jensenshannon(np.exp(j_scores1[:, :, t_i].ravel()),
                                                    np.exp(arr.ravel()))
                    t_scores[_e, t_i] = js1v2
                    t_scores_r[_e, t_i] = rjs1v2

            # with open(f'/cs/snapless/oabend/eitan.wagner/calibration/all_scores{"_a" if adjacent else ""}_w1.npy', 'wb') as f:
            #     np.save(f, all_scores_w1)
            # with open(f'/cs/snapless/oabend/eitan.wagner/calibration/all_scores{"_a" if adjacent else ""}_w2.npy', 'wb') as f:
            #     np.save(f, all_scores_w2)
            # with open(f'/cs/snapless/oabend/eitan.wagner/calibration/all_scores{"_a" if adjacent else ""}_i.npy', 'wb') as f:
            #     np.save(f, all_scores_i)
            # if adjacent:
            #     with open(f'/cs/snapless/oabend/eitan.wagner/calibration/all_scores{"_a" if adjacent else ""}_j.npy', 'wb') as f:
            #         np.save(f, all_scores_j)
            # with open(f'/cs/snapless/oabend/eitan.wagner/calibration/calculated_noun_list{"_a" if adjacent else ""}.json', 'w') as outfile:
            #     json.dump(calculated_nouns, outfile)

            if adjacent and not temperature_list:
                j_range = range(_i // 2, _i // 2 + 50)
                # print("w1 vs w2:")
                arr = np.copy(all_scores_w2[np.ix_(i_range, j_range)])

                entropies2.append(entropy(np.exp(arr.ravel()), base=2))
                entropies1.append(entropy(np.exp(all_scores_w1[np.ix_(i_range, j_range)].ravel()), base=2))
                d["1v2"].append(np.abs(all_scores_w1[np.ix_(i_range, j_range)] - arr).mean())
                # arr1 = all_scores_w1[np.ix_(i_range, j_range)] - arr
                # d["1v2"].append(arr1[arr1 >= 0].mean())
                # unnormalized kl-divergence
                d["ukl1v2"].append(np.exp(all_scores_w1[np.ix_(i_range, j_range)].ravel()) @
                                   (all_scores_w1[np.ix_(i_range, j_range)].ravel() - arr.ravel()))
                # d["js1v2"].append(distance.jensenshannon(all_scores_w1[np.ix_(i_range, j_range)].ravel(),
                #                                          arr.ravel()))
                # F.cross_entropy()

                d["js1v2"].append(distance.jensenshannon(np.exp(all_scores_w1[np.ix_(i_range, j_range)].ravel()),
                                                         np.exp(arr.ravel())))
                np.random.shuffle(arr.ravel())
                d["r1v2"].append(np.abs(all_scores_w1[np.ix_(i_range, j_range)]
                                        - arr).mean())
                # arr1 = all_scores_w1[np.ix_(i_range, j_range)] - arr
                # d["r1v2"].append(arr1[arr1 >= 0].mean())
                # d["rjs1v2"].append(distance.jensenshannon(all_scores_w1[np.ix_(i_range, j_range)].ravel(),
                #                                           arr.ravel()))
                d["rjs1v2"].append(distance.jensenshannon(np.exp(all_scores_w1[np.ix_(i_range, j_range)].ravel()),
                                                          np.exp(arr.ravel())))
                # print(d["1v2"][-1])
                # print(d["r1v2"][-1])
                # print(d["js1v2"][-1])
                # print(d["rjs1v2"][-1])

                # print("w1 vs i:")
                arr = np.copy(all_scores_i[np.ix_(i_range, j_range)])
                d["1vi"].append(np.abs(all_scores_w1[np.ix_(i_range, j_range)] - arr).mean())
                # d["js1vi"].append(distance.jensenshannon(all_scores_w1[np.ix_(i_range, j_range)].ravel(),
                #                                          all_scores_i[np.ix_(i_range, j_range)].ravel()))

                d["ukl1vi"].append(np.exp(all_scores_w1[np.ix_(i_range, j_range)].ravel()) @
                                   (all_scores_w1[np.ix_(i_range, j_range)].ravel() - arr.ravel()))
                d["js1vi"].append(distance.jensenshannon(np.exp(all_scores_w1[np.ix_(i_range, j_range)].ravel()),
                                                         np.exp(all_scores_i[np.ix_(i_range, j_range)].ravel())))
                np.random.shuffle(arr.ravel())
                d["r1vi"].append(np.abs(all_scores_w1[np.ix_(i_range, j_range)]
                                        - arr).mean())
                # d["rjs1vi"].append(distance.jensenshannon(all_scores_w1[np.ix_(i_range, j_range)].ravel(),
                #                                           arr.ravel()))
                d["rjs1vi"].append(distance.jensenshannon(np.exp(all_scores_w1[np.ix_(i_range, j_range)].ravel()),
                                                          np.exp(arr.ravel())))
                # print(d["1vi"][-1])
                # print(d["r1vi"][-1])
                # print(d["js1vi"][-1])
                # print(d["rjs1vi"][-1])

                # print("w1 vs j:")
                arr = np.copy(all_scores_j[np.ix_(i_range, j_range)])
                d["1vj"].append(np.abs(all_scores_w1[np.ix_(i_range, j_range)] - arr).mean())
                # d["js1vj"].append(distance.jensenshannon(all_scores_w1[np.ix_(i_range, j_range)].ravel(),
                #                                          all_scores_j[np.ix_(i_range, j_range)].ravel()))

                d["ukl1vj"].append(np.exp(all_scores_w1[np.ix_(i_range, j_range)].ravel()) @
                                   (all_scores_w1[np.ix_(i_range, j_range)].ravel() - arr.ravel()))
                d["js1vj"].append(distance.jensenshannon(np.exp(all_scores_w1[np.ix_(i_range, j_range)].ravel()),
                                                         np.exp(all_scores_j[np.ix_(i_range, j_range)].ravel())))
                np.random.shuffle(arr.ravel())
                d["r1vj"].append(np.abs(all_scores_w1[np.ix_(i_range, j_range)]
                                        - arr).mean())
                # d["rjs1vj"].append(distance.jensenshannon(all_scores_w1[np.ix_(i_range, j_range)].ravel(),
                #                                           arr.ravel()))
                d["rjs1vj"].append(distance.jensenshannon(np.exp(all_scores_w1[np.ix_(i_range, j_range)].ravel()),
                                                          np.exp(arr.ravel())))
                # print(d["1vj"][-1])
                # print(d["r1vj"][-1])
                # print(d["js1vj"][-1])
                # print(d["rjs1vj"][-1])

                # print("w2 vs j:")
                arr = np.copy(all_scores_j[np.ix_(i_range, j_range)])
                d["2vj"].append(
                    np.abs(all_scores_w2[np.ix_(i_range, j_range)] - all_scores_j[np.ix_(i_range, j_range)]).mean())
                # d["js2vj"].append(distance.jensenshannon(all_scores_w2[np.ix_(i_range, j_range)].ravel(),
                #                                          all_scores_j[np.ix_(i_range, j_range)].ravel()))

                d["ukl2vj"].append(np.exp(all_scores_w2[np.ix_(i_range, j_range)].ravel()) @
                                   (all_scores_w2[np.ix_(i_range, j_range)].ravel() - arr.ravel()))
                d["js2vj"].append(distance.jensenshannon(np.exp(all_scores_w2[np.ix_(i_range, j_range)].ravel()),
                                                         np.exp(all_scores_j[np.ix_(i_range, j_range)].ravel())))
                np.random.shuffle(arr.ravel())
                d["r2vj"].append(np.abs(all_scores_w2[np.ix_(i_range, j_range)]
                                        - arr).mean())
                # d["rjs2vj"].append(distance.jensenshannon(all_scores_w2[np.ix_(i_range, j_range)].ravel(),
                #                                           arr.ravel()))
                d["rjs2vj"].append(distance.jensenshannon(np.exp(all_scores_w2[np.ix_(i_range, j_range)].ravel()),
                                                          np.exp(arr.ravel())))
                # print(d["2vj"][-1])
                # print(d["r2vj"][-1])
                # print(d["js2vj"][-1])
                # print(d["rjs2vj"][-1])
                sys.stdout.flush()
            if _e == len(_i_range) // 2:
                print(d)

        if temperature_list:
            print("Temperatures: ")
            print(temperature)
            print("t_scores:")
            print(t_scores)
            print("t_scores_r:")
            print(t_scores_r)
            with open(f'/cs/snapless/oabend/eitan.wagner/calibration/t_scores.npy',
                      'wb') as f:
                np.save(f, t_scores)
            with open(f'/cs/snapless/oabend/eitan.wagner/calibration/t_scores_r.npy',
                      'wb') as f:
                np.save(f, t_scores_r)
            return noun_count, adj_count

        d["ukl1v2"] = (np.array(d["ukl1v2"]) / np.exp(w1_sum) + w1_sum - w2_sum).tolist()
        d["ukl1vi"] = (np.array(d["ukl1vi"]) / np.exp(w1_sum) + w1_sum - i_sum).tolist()
        d["ukl1vj"] = (np.array(d["ukl1vj"]) / np.exp(w1_sum) + w1_sum - j_sum).tolist()
        d["ukl2vj"] = (np.array(d["ukl2vj"]) / np.exp(w2_sum) + w2_sum - j_sum).tolist()

        with open(f'/cs/snapless/oabend/eitan.wagner/calibration/all_scores{"_a" if adjacent else ""}_w1.npy',
                  'wb') as f:
            np.save(f, _all_scores_w1)
        with open(f'/cs/snapless/oabend/eitan.wagner/calibration/all_scores{"_a" if adjacent else ""}_w2.npy',
                  'wb') as f:
            np.save(f, _all_scores_w2)
        with open(f'/cs/snapless/oabend/eitan.wagner/calibration/all_scores{"_a" if adjacent else ""}_i.npy',
                  'wb') as f:
            np.save(f, _all_scores_i)
        if adjacent:
            with open(f'/cs/snapless/oabend/eitan.wagner/calibration/all_scores{"_a" if adjacent else ""}_j.npy',
                      'wb') as f:
                np.save(f, _all_scores_j)
        print(d)
        print("Entropy1: ")
        print(entropies1)
        print("Entropy2: ")
        print(entropies2)

    return noun_count, adj_count


# def load_w_temperature(t, adjacent=True):
#     with open(f'/cs/snapless/oabend/eitan.wagner/calibration/all_scores{"_a" if adjacent else ""}_w1.npy', 'rb') as f:
#         _all_scores_w1 = np.load(f)
#     with open(f'/cs/snapless/oabend/eitan.wagner/calibration/all_scores{"_a" if adjacent else ""}_w2.npy', 'rb') as f:
#         _all_scores_w2 = np.load(f)
#     with open(f'/cs/snapless/oabend/eitan.wagner/calibration/all_scores{"_a" if adjacent else ""}_i.npy', 'rb') as f:
#         _all_scores_i = np.load(f)
#     if adjacent:
#         with open(f'/cs/snapless/oabend/eitan.wagner/calibration/all_scores{"_a" if adjacent else ""}_j.npy',
#                   'rb') as f:
#             _all_scores_j = np.load(f)
#     return _all_scores_w1, _all_scores_w2, _all_scores_i, _all_scores_j

def make_freq():
    # from: https://github.com/hermitdave/FrequencyWords/blob/master/content/2018/en/en_full.txt
    with open(args.path + 'en_full.txt', 'r') as infile:
        lines = infile.readlines()
    freq_dict = {l.split()[0]: int(l.split()[1]) for l in lines}
    return freq_dict


def divergence2(noun_count, adj_count):
    with open(f'/cs/snapless/oabend/eitan.wagner/segmentation/all_scores_{noun_count}_{adj_count}.json', 'r') as infile:
        all_scores = json.load(infile)

    p1, p2 = [], []
    for t, s in all_scores.items():
        p1.append(s["w1 first"])
        p2.append(s["w2 first"])

    from scipy.spatial import distance
    p1, p2 = np.exp(p1), np.exp(p2)
    d = distance.jensenshannon(p1 / p1.sum(), p2 / p2.sum(), 2.)
    print("Jensen-Shannon: ")
    print(d)
    return d


def divergence(noun_count, adj_count):
    with open(f'/cs/snapless/oabend/eitan.wagner/calibration/all_scores_w1.npy', 'rb') as f:
        p1 = np.load(f)[:noun_count, :adj_count]
    with open(f'/cs/snapless/oabend/eitan.wagner/calibration/all_scores_w2.npy', 'rb') as f:
        p2 = np.load(f)[:noun_count, :adj_count]
    with open(f'/cs/snapless/oabend/eitan.wagner/calibration/all_scores_i.npy', 'rb') as f:
        all_scores_i = np.load(f)[:noun_count, :adj_count]

    from scipy.spatial import distance
    p1, p2 = np.exp(p1.ravel()), np.exp(p2.ravel())
    d = distance.jensenshannon(p1 / p1.sum(), p2 / p2.sum(), 2.)
    print("Jensen-Shannon: ")
    print(d)
    return d


# *********************

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
        with open(f'/cs/snapless/oabend/eitan.wagner/calibration/_extra_noise.json', 'r') as f:
            texts = json.load(f)[:num_samples]
    else:
        characters = string.ascii_letters + string.digits + string.punctuation
        texts = [''.join(random.choice(characters) for i in range(length)) for _ in range(num_samples)]
        with open(f'/cs/snapless/oabend/eitan.wagner/calibration/_extra_noise.json', 'w') as f:
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
        with open(f'/cs/snapless/oabend/eitan.wagner/calibration/_total_noise.json', 'r') as f:
            texts = json.load(f)[:num_samples]
    else:
        tokens = np.random.randint(tokenizer.vocab_size, size=(num_samples, length))
        texts = tokenizer.batch_decode(tokens, skip_special_tokens=True)
        with open(f'/cs/snapless/oabend/eitan.wagner/calibration/_total_noise.json', 'w') as f:
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

    if args.template:
        template = sys.argv[sys.argv.index("--template") + 1]

    print(f"Making synthetic data. {num_samples} texts.")
    print(f"Template: {template}")
    print(f"Noise: {noise}")
    inserted_id = tokenizer(" <NP>")['input_ids'][1]
    # inserted_id = 28696
    # obtain list of two-word noun-phrases
    if args.load:
        if "common" in sys.argv:
            print('Loading data - common NPs')
            with open(
                    f'/cs/snapless/oabend/eitan.wagner/calibration/common_nps2.json', 'r') as f:
                pairs = json.load(f)[:num_samples]
        else:
            print(f'Loading data - {tokenizer.name_or_path.split("/")[-1].split("-")[0]}')
            with open(f'/cs/snapless/oabend/eitan.wagner/calibration/{tokenizer.name_or_path.split("/")[-1].split("-")[0]}'
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
        with open(f'/cs/snapless/oabend/eitan.wagner/calibration/{tokenizer.name_or_path.split("/")[-1].split("-")[0]}'
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
    if "-set_num" in sys.argv:
        set_num = int(sys.argv[sys.argv.index("-set_num") + 1])
        set_num_s = f"s{set_num}"
    if not return_tokenizer:
        index = None
        deps = "deps" in sys.argv
        return_probs = "return_probs" in sys.argv
        deps2 = "deps2" in sys.argv
        use_underscore = "use_underscore" in sys.argv
        entropies = "entropies" in sys.argv
        texts = []
        print("Using loss (counting also the special tokens)")
        if dataset_name == 'wikitext-2':
            from datasets import load_dataset
            dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split="train")
            texts = dataset['text']
        elif dataset_name == 'news-sp500':
            from datasets import load_dataset
            dataset = load_dataset('edarchimbaud/news-sp500', split="train")
            texts = dataset['body']
        elif dataset_name == 'new_news':
            with open(f'/cs/snapless/oabend/eitan.wagner/calibration/news-2.7.2023.json', 'r') as file:
                texts = [t['text'] for t in json.load(file) if t['title'].find("Subscribe") == -1]
            with open(f'/cs/snapless/oabend/eitan.wagner/calibration/news-6.7.2023.json', 'r') as file:
                texts = texts + [t['text'] for t in json.load(file) if t['title'].find("Subscribe") == -1]
            if "shuffle" in sys.argv:
                import random
                texts = [" ".join(random.sample(t.split(), len(t.split()))) for t in texts]
        elif dataset_name == 'new_news2':
            with open(f'/cs/snapless/oabend/eitan.wagner/calibration/news-4.9.2023.json', 'r') as file:
                texts = [t['text'] for t in json.load(file) if t['title'].find("Subscribe") == -1]
            with open(f'/cs/snapless/oabend/eitan.wagner/calibration/news-18.9.2023.json', 'r') as file:
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
            tokenizer = AutoTokenizer.from_pretrained(name, legacy=False)
            if return_tokenizer:
                return tokenizer
            model = AutoModelForSeq2SeqLM.from_pretrained(name, device_map="auto",
                                                          quantization_config=double_quant_config)
        elif size == "11b" or size == "xxl":
            tokenizer = AutoTokenizer.from_pretrained(name, legacy=False)
            if return_tokenizer:
                return tokenizer
            model = T5ForConditionalGeneration.from_pretrained(name, device_map="auto", load_in_4bit=True)
        else:
            tokenizer = AutoTokenizer.from_pretrained(name, legacy=False)
            if return_tokenizer:
                return tokenizer
            model = T5ForConditionalGeneration.from_pretrained(name)

    elif "flan" in sys.argv:
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
        # from transformers import BitsAndBytesConfig
        # double_quant_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_use_double_quant=True,
        # )
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
    print("Using model.eval()")

    if entropies:
        return llama_entropies(texts, model, tokenizer)

    if "t5" in sys.argv:
        # pp = pair_probabilities(texts, model, tokenizer, temperature=1., deps=deps, deps2=deps2)
        if use_underscore:
            pp = pair_probabilities2(texts, model, tokenizer, temperature=1., deps=deps)
        else:
            pp = pair_probabilities3(texts, model, tokenizer, temperature=1., deps=deps)
    elif "llama" in sys.argv or "flan-ul2" in sys.argv or "flan" in sys.argv:
        if return_probs:
            probs, ids = llama_pair_probabilities(texts, model, tokenizer, fixed_pos=index, return_probs=True, set_num=set_num)
            with open(f'/cs/snapless/oabend/eitan.wagner/calibration/{dataset_name}{set_num_s}_scores_{name.split("/")[-1]}_probs.npy',
                      'wb') as outfile:
                np.save(outfile, probs)
            with open(f'/cs/snapless/oabend/eitan.wagner/calibration/{dataset_name}{set_num_s}_scores_{name.split("/")[-1]}_ids.npy',
                      'wb') as outfile:
                np.save(outfile, ids)
            return
        else:
            pp = llama_pair_probabilities(texts, model, tokenizer, fixed_pos=index, set_num=set_num)
    else:
        if return_probs:
            probs, ids = mlm_pair_probabilities(texts, model, tokenizer, temperature=1., deps=deps, deps2=deps2, fixed_pos=index, return_probs=True)
            with open(f'/cs/snapless/oabend/eitan.wagner/calibration/{dataset_name}_scores_{name.split("/")[-1]}_probs.npy', 'wb') as outfile:
                np.save(outfile, probs)
            with open(f'/cs/snapless/oabend/eitan.wagner/calibration/{dataset_name}_scores_{name.split("/")[-1]}_ids.npy', 'wb') as outfile:
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
    with open(f'/cs/snapless/oabend/eitan.wagner/calibration/{dataset_name}{set_num_s}_scores_{name.split("/")[-1]}'
              f'{"_deps" if deps else ""}{"_deps2" if deps2 else ""}{"_l1_2" if l1_2 else ""}{"_l2_1" if l2_1 else ""}'
              f'{"_u" if use_underscore else "_"}{na}{ranks}.json', 'w') as outfile:
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
            # print("AVG p12, p21 for cases H_w1 > H_w2, H_w1 < H_w2:")
            # p12_ent0 = sum([p[1] for p in pp_ent0]) / len(pp_ent0)
            # p21_ent0 = sum([p[2] for p in pp_ent0]) / len(pp_ent0)
            # p12_ent1 = sum([p[1] for p in pp_ent1]) / len(pp_ent1)
            # p21_ent1 = sum([p[2] for p in pp_ent1]) / len(pp_ent1)
            # print(p12_ent0, p21_ent0, p12_ent1, p21_ent1)
            # print("AVG p12 - p21 for cases H_w1 < H_w2, H_w1 < H_w2:")
            # p_ent0 = sum([p[1] - p[2] for p in pp_ent0]) / len(pp_ent0)
            # p_ent1 = sum([p[1] - p[2] for p in pp_ent1]) / len(pp_ent1)
            # print(p_ent0, p_ent1)
            # count0 = len([p for p in pp_ent0 if p[1] > p[2]]) / len(pp_ent0)
            # count1 = len([p for p in pp_ent1 if p[1] > p[2]]) / len(pp_ent1)
            # print("Count (p12 > p21) for case H_w1 > H_w2:")
            # print(count0)
            # print("Count (p12 > p21) for case H_w1 < H_w2:")
            # print(count1)

        print("\nlens for cases H_w2|1 > H_w1|2, H_w2|1 < H_w1|2:")
        print(len(pp_ent0_c), len(pp_ent1_c))
        if len(pp_ent0_c) > 0 and len(pp_ent1_c) > 0:
            print_by_cond(pp_ent0_c, pp_ent1_c, name0="H_w2|1 >= H_w1|2", name1="H_w2|1 < H_w1|2",)
            # print("AVG p12, p21 for cases H_w1 > H_w2, H_w1 < H_w2:")
            # p12_ent0_c = sum([p[1] for p in pp_ent0_c]) / len(pp_ent0_c)
            # p21_ent0_c = sum([p[2] for p in pp_ent0_c]) / len(pp_ent0_c)
            # p12_ent1_c = sum([p[1] for p in pp_ent1_c]) / len(pp_ent1_c)
            # p21_ent1_c = sum([p[2] for p in pp_ent1_c]) / len(pp_ent1_c)
            # print(p12_ent0_c, p21_ent0_c, p12_ent1_c, p21_ent1_c)
            # print("AVG p12 - p21 for cases H_w2|1 < H_w1|2, H_w2|1 < H_w1|2:")
            # p_ent0_c = sum([p[1] - p[2] for p in pp_ent0_c]) / len(pp_ent0_c)
            # p_ent1_c = sum([p[1] - p[2] for p in pp_ent1_c]) / len(pp_ent1_c)
            # print(p_ent0_c, p_ent1_c)
            # count0_c = len([p for p in pp_ent0_c if p[1] > p[2]]) / len(pp_ent0_c)
            # count1_c = len([p for p in pp_ent1_c if p[1] > p[2]]) / len(pp_ent1_c)
            # print("Count (p12 > p21) for case H_w2|1 > H_w1|2:")
            # print(count0_c)
            # print("Count (p12 > p21) for case H_w2|1 < H_w1|2:")
            # print(count1_c)

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
            # print("AVG p12, p21 for H_w12 >= H_w21 and then for H_w12 < H_w21:")
            # p12_ent12 = sum([p[1] for p in pp_ent12]) / len(pp_ent12)
            # p21_ent12 = sum([p[2] for p in pp_ent12]) / len(pp_ent12)
            # p12_ent21 = sum([p[1] for p in pp_ent21]) / len(pp_ent21)
            # p21_ent21 = sum([p[2] for p in pp_ent21]) / len(pp_ent21)
            # print(p12_ent12, p21_ent12, p12_ent21, p21_ent21)
            # print("AVG p12 - p21 for cases H_w12 >= H_w21, H_w12 < H_w21:")
            # p_ent12 = sum([p[1] - p[2] for p in pp_ent12]) / len(pp_ent12)
            # p_ent21 = sum([p[1] - p[2] for p in pp_ent21]) / len(pp_ent21)
            # print(p_ent12, p_ent21)
            # print("Random AVG p12 - p21 for cases H_w12 >= H_w21, H_w12 < H_w21:")
            # p1s, p2s = np.array([p[1] for p in pp_ent12]), np.array([p[2] for p in pp_ent12])
            # np.random.shuffle(p2s)
            # r_p_ent12 = sum(p1s - p2s) / len(p1s)
            # r_count12 = sum(p1s > p2s) / len(p1s)
            # p1s, p2s = np.array([p[1] for p in pp_ent21]), np.array([p[2] for p in pp_ent21])
            # np.random.shuffle(p2s)
            # r_p_ent21 = sum(p1s - p2s) / len(p1s)
            # r_count21 = sum(p1s > p2s) / len(p1s)
            # print(r_p_ent12, r_p_ent21)
            # count12 = len([p for p in pp_ent12 if p[1] > p[2]]) / len(pp_ent12)
            # count21 = len([p for p in pp_ent21 if p[1] > p[2]]) / len(pp_ent21)
            # print("Count (p12 > p21) for case H_w12 >= H_w21:")
            # print(count12)
            # print("Count (p12 > p21) for case H_w12 < H_w21:")
            # print(count21)
            # print("Random Count (p12 > p21) for case H_w12 >= H_w21:")
            # print(r_count12)
            # print("Random Count (p12 > p21) for case H_w12 < H_w21:")
            # print(r_count21)

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
            # print("AVG p12, p21 for H_w >= AvgH_w and then for H_w < AvgH_w:")
            # p12_ent_l = sum([p[1] for p in pp_ent_l]) / len(pp_ent_l)
            # p21_ent_l = sum([p[2] for p in pp_ent_l]) / len(pp_ent_l)
            # p12_ent_s = sum([p[1] for p in pp_ent_s]) / len(pp_ent_s)
            # p21_ent_s = sum([p[2] for p in pp_ent_s]) / len(pp_ent_s)
            # print(p12_ent_l, p21_ent_l, p12_ent_s, p21_ent_s)
            # print("AVG p12 - p21 for cases H_w >= AvgH_w, H_w < AvgH_w:")
            # p_ent_l = sum([p[1] - p[2] for p in pp_ent_l]) / len(pp_ent_l)
            # p_ent_s = sum([p[1] - p[2] for p in pp_ent_s]) / len(pp_ent_s)
            # print(p_ent_l, p_ent_s)
            # count_l = len([p for p in pp_ent_l if p[1] > p[2]]) / len(pp_ent_l)
            # count_s = len([p for p in pp_ent_s if p[1] > p[2]]) / len(pp_ent_s)
            # print("Count (p12 > p21) for case H_w >= AvgH_w:")
            # print(count_l)
            # print("Count (p12 > p21) for case H_w < AvgH_w:")
            # print(count_s)

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

def load_probs(dataset_name="wikitext", name="roberta-base", set_num=""):
    set_num_s = ""
    if set_num != "":
        set_num_s = f"s{set_num}"
    with open(f'/cs/snapless/oabend/eitan.wagner/calibration/{dataset_name}{set_num_s}_scores_{name.split("/")[-1]}_.json', 'r') as infile:
        pp = json.load(infile)
    return pp

def load_full_probs(dataset_name="wikitext", name="roberta-base", set_num=""):
    set_num_s = ""
    if set_num != "":
        set_num_s = f"s{set_num}"
    with open(f'/cs/snapless/oabend/eitan.wagner/calibration/{dataset_name}{set_num_s}_scores_{name.split("/")[-1]}_probs.npy', 'rb') as infile:
        probs = np.load(infile)
    with open(f'/cs/snapless/oabend/eitan.wagner/calibration/{dataset_name}{set_num_s}_scores_{name.split("/")[-1]}_ids.npy', 'rb') as infile:
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

def mlm_entropies(texts, model, tokenizer):
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

def pair_probabilities(texts, model, tokenizer, temperature=1., deps=False, deps2=False):
    """
    Measures probabilities for adjacent token pairs in the given texts
    :param texts: list of strings
    :param model:
    :param tokenizer:
    :return:
    """
    skip_word = "skip_word" in sys.argv
    l1_2 = "l1_2" in sys.argv
    l2_1 = "l2_1" in sys.argv
    deps3 = "deps3" in sys.argv
    if deps2 or deps3:
        deps = True
    if deps:
        import spacy
        nlp = spacy.load("en_core_web_md")
        # POS_tags = ["ADJ", "ADV", "NOUN", "PRON", "PROPN", "VERB"]
        # print(f"Using only {POS_tags}")
        dep_tags = ["amod"]
        print(f"Using only {dep_tags}")

    extra_ids = tokenizer.get_sentinel_token_ids()
    extra_ids.sort()
    l = []
    for text in tqdm.tqdm(texts[:250 if ("ul2" in sys.argv or "flan-ul2" in sys.argv) else (500 if ("3b" in sys.argv or "11b" in sys.argv) else 5000)]
                          if (not deps or deps3) else texts):
        if len(text) < 100 or not text.isascii():
            continue
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs2 = BatchEncoding(inputs)

        if deps:
            doc = nlp(text)
            t2st = {i: doc.char_span(s, e, alignment_mode="contract") for i, (s, e) in enumerate(inputs.encodings[0].offsets)}
            def is_skip(i):
                # if t2st[i] is None or t2st[i+1] is None or t2st[i][0].is_punct or t2st[i+1][0].is_punct or t2st[i][0].is_stop or t2st[i+1][0].is_stop:  # covers token that are not full words
                # if t2st[i] is None or t2st[i+1] is None or t2st[i][0].pos_ not in POS_tags or t2st[i+1][0].pos_ not in POS_tags:  # covers token that are not full words
                if t2st[i] is None or t2st[i+1] is None:  # covers token that are not full words
                    return False
                # return t2st[i+1][0].head.i < t2st[i][0].i and t2st[i][0].head.i == t2st[i+1][0].i and t2st[i+1][0].i > t2st[i][0].i
                if not deps2:  # an arc going "backwards"
                    return t2st[i][0].head.i == t2st[i+1][0].i and t2st[i+1][0].i > t2st[i][0].i and \
                           t2st[i][0].n_rights + t2st[i][0].n_lefts == 0 and t2st[i][0].dep_ in dep_tags
                else:
                    return t2st[i+1][0].head.i == t2st[i][0].i and t2st[i+1][0].i > t2st[i][0].i and t2st[i+1][0].n_rights + t2st[i+1][0].n_lefts == 0

        for i in range(1, len(inputs.input_ids[0]) - 2):
            j = i + 1
            if skip_word:
                j = i + 2
            if l1_2 or l2_1:
                if l1_2:
                    j = i + 2
                else:
                    j = i + 3
                if i == len(inputs.input_ids[0]) - 3:
                    continue
            if deps and deps3 and is_skip(i):
                continue
            elif deps and not deps3 and not is_skip(i):
                continue
            # TODO: batchify
            id1, id2 = inputs['input_ids'][0, i], inputs['input_ids'][0, j]

            # mask only first
            inputs2['input_ids'] = torch.clone(inputs.input_ids)
            labels = torch.full_like(inputs2['input_ids'], -100)
            if not l2_1:
                labels[0, :3] = torch.tensor([extra_ids[0], inputs2['input_ids'][0, i].item(), extra_ids[1]])
                inputs2['input_ids'][0, i] = extra_ids[0]
            else:
                labels[0, :4] = torch.tensor([extra_ids[0], inputs2['input_ids'][0, i].item(), inputs2['input_ids'][0, i+1].item(), extra_ids[1]])
                inputs2['input_ids'][0, i] = extra_ids[0]
                inputs2['input_ids'][0, i: -1] = inputs['input_ids'][0, i+1:]
                inputs2['input_ids'][0, -1] = tokenizer.pad_token_id
            with torch.no_grad():
                out = model(**inputs2.to(dev), labels=labels.to(dev))

            # mask second
            inputs2['input_ids'] = torch.clone(inputs.input_ids)
            labels = torch.full_like(inputs2['input_ids'], -100)
            if not l1_2:
                labels[0, :3] = torch.tensor([extra_ids[1], inputs2['input_ids'][0, j].item(), extra_ids[2]])
                inputs2['input_ids'][0, j] = extra_ids[1]
            else:  # second span is of length 2
                labels[0, :4] = torch.tensor([extra_ids[1], inputs2['input_ids'][0, j].item(), inputs2['input_ids'][0, j+1].item(), extra_ids[2]])
                inputs2['input_ids'][0, j] = extra_ids[1]
                inputs2['input_ids'][0, j: -1] = inputs['input_ids'][0, j+1:]
                inputs2['input_ids'][0, -1] = tokenizer.pad_token_id
            with torch.no_grad():
                out2 = model(**inputs2.to(dev), labels=labels.to(dev))

            # mask both
            inputs2['input_ids'] = torch.clone(inputs.input_ids)
            labels = torch.full_like(inputs2['input_ids'], -100)
            labels_a = torch.full_like(inputs2['input_ids'], -100)
            labels_b = torch.full_like(inputs2['input_ids'], -100)
            if not l1_2 and not l2_1:
                labels[0, :5] = torch.tensor([extra_ids[0], inputs2['input_ids'][0, i].item(), extra_ids[1],
                                              inputs2['input_ids'][0, j].item(), extra_ids[2]])
                labels_a[0, :3] = torch.tensor([extra_ids[0], inputs2['input_ids'][0, i].item(), extra_ids[1]])
                labels_b[0, :3] = torch.tensor([extra_ids[1],
                                              inputs2['input_ids'][0, j].item(), extra_ids[2]])
                inputs2['input_ids'][0, i] = extra_ids[0]
                inputs2['input_ids'][0, j] = extra_ids[1]
            elif l1_2:
                labels[0, :6] = torch.tensor([extra_ids[0], inputs2['input_ids'][0, i].item(), extra_ids[1],
                                              inputs2['input_ids'][0, j].item(), inputs2['input_ids'][0, j+1].item(), extra_ids[2]])
                labels_a[0, :3] = torch.tensor([extra_ids[0], inputs2['input_ids'][0, i].item(), extra_ids[1]])
                labels_b[0, :4] = torch.tensor([extra_ids[1],
                                              inputs2['input_ids'][0, j].item(), inputs2['input_ids'][0, j+1].item(), extra_ids[2]])
                inputs2['input_ids'][0, i] = extra_ids[0]
                inputs2['input_ids'][0, j] = extra_ids[1]
                inputs2['input_ids'][0, j: -1] = inputs['input_ids'][0, j+1:]   # TODO: i think this is wrong - no second extra id!!!
                inputs2['input_ids'][0, -1] = tokenizer.pad_token_id
            elif l2_1:
                labels[0, :6] = torch.tensor([extra_ids[0], inputs2['input_ids'][0, i].item(), inputs2['input_ids'][0, i+1].item(), extra_ids[1],
                                              inputs2['input_ids'][0, j].item(), extra_ids[2]])
                labels_a[0, :4] = torch.tensor([extra_ids[0], inputs2['input_ids'][0, i].item(), inputs2['input_ids'][0, i+1].item(), extra_ids[1]])
                labels_b[0, :3] = torch.tensor([extra_ids[1], inputs2['input_ids'][0, j].item(), extra_ids[2]])
                inputs2['input_ids'][0, j] = extra_ids[1]
                inputs2['input_ids'][0, i] = extra_ids[0]
                inputs2['input_ids'][0, i: -1] = inputs['input_ids'][0, i+1:]  # TODO: i think this is wrong as it overwrites j!!!
                inputs2['input_ids'][0, -1] = tokenizer.pad_token_id
            with torch.no_grad():
                out3 = model(**inputs2.to(dev), labels=labels.to(dev))
                out3_a = model(**inputs2.to(dev), labels=labels_a.to(dev))
                out3_b = model(**inputs2.to(dev), labels=labels_b.to(dev))

            if not skip_word and not l1_2 and not l2_1:
                # joint
                inputs2['input_ids'] = torch.clone(inputs.input_ids)
                labels = torch.full_like(inputs2['input_ids'], -100)
                labels[0, :4] = torch.tensor([extra_ids[0], inputs2['input_ids'][0, i].item(),
                                              inputs2['input_ids'][0, i+1].item(), extra_ids[1]])
                inputs2['input_ids'][0, i] = extra_ids[0]
                inputs2['input_ids'][0, i+1: -1] = inputs['input_ids'][0, i+2:]
                inputs2['input_ids'][0, -1] = tokenizer.pad_token_id
                with torch.no_grad():
                    out4 = model(**inputs2.to(dev), labels=labels.to(dev))

            logits = out.logits / temperature
            logits2 = out2.logits / temperature
            logits3 = out3.logits / temperature
            logits3a = out3_a.logits / temperature
            logits3b = out3_b.logits / temperature
            if not skip_word and not l1_2 and not l2_1:
                logits4 = out4.logits / temperature
            probs = logits.log_softmax(dim=-1).cpu().numpy()
            probs2 = logits2.log_softmax(dim=-1).cpu().numpy()
            probs3 = logits3.log_softmax(dim=-1).cpu().numpy()
            probs3a = logits3a.log_softmax(dim=-1).cpu().numpy()
            probs3b = logits3b.log_softmax(dim=-1).cpu().numpy()
            if not skip_word and not l1_2 and not l2_1:
                probs4 = logits4.log_softmax(dim=-1).cpu().numpy()

            # entropies of: w1 only, w2 only, w1|w2, w2|w1
            entropies = [float(H(logits3[0, 1])), float(H(logits3[0, 3])), float(H(logits[0, 1])), float(H(logits2[0, 1]))]
            # TODO: do the logits go by the input and not the labels? NO!!!

            # p12 = probs3[0, 1][inputs['input_ids'][0, i]] + probs2[0, 1][inputs['input_ids'][0, i+1]]
            # p12 = probs3a[0, 1][inputs['input_ids'][0, i]] + probs2[0, 1][inputs['input_ids'][0, i+1]]
            if not l1_2 and not l2_1:
                _p12 = -out3_a.loss * 3 - out2.loss * 3
            # p21 = probs3[0, 3][inputs['input_ids'][0, i+1]] + probs[0, 1][inputs['input_ids'][0, i]]
            # p21 = probs3b[0, 1][inputs['input_ids'][0, i+1]] + probs[0, 1][inputs['input_ids'][0, i]]
                _p21 = -out3_b.loss * 3 - out.loss * 3
                if not skip_word:
                    # pair_prob = probs4[0, 1][inputs['input_ids'][0, i]] + probs4[0, 2][inputs['input_ids'][0, i+1]]
                    _pair_prob = -out4.loss * 4
                else:
                    _pair_prob = -out3.loss * 5
            elif l1_2:
                _p12 = -out3_a.loss * 3 - out2.loss * 4
                _p21 = -out3_b.loss * 4 - out.loss * 3
                _pair_prob = -out3.loss * 6
            elif l2_1:
                _p12 = -out3_a.loss * 4 - out2.loss * 3
                _p21 = -out3_b.loss * 3 - out.loss * 4
                _pair_prob = -out3.loss * 6
            l.append((float(_pair_prob), float(_p12), float(_p21), [id1.item(), id2.item()], entropies))
            # l.append((float(_pair_prob), float(_p12), float(_p21), [tokenizer.decode(id1.item()), tokenizer.decode(id2.item())], entropies))
    return l

def pair_probabilities2(texts, model, tokenizer, temperature=1., deps=False):
    """
    Measures probabilities for adjacent token pairs in the given texts
    :param texts: list of strings
    :param model:
    :param tokenizer:
    :return:
    """
    print("using new method - different decoding order")
    print("sorted the extra ids in reverse order")
    print("fixed the is_skip function")
    print("fixed the underscore and r_inputs")
    print("using restrictions for deps3")
    print("doing without end token")
    skip_word = "skip_word" in sys.argv
    l1_2 = "l1_2" in sys.argv
    l2_1 = "l2_1" in sys.argv
    deps3 = "deps3" in sys.argv
    if deps3:
        deps = True
    if deps:
        import spacy
        nlp = spacy.load("en_core_web_md")
        dep_tags = ["amod"]
        print(f"Using only {dep_tags}")

    extra_ids = tokenizer.get_sentinel_token_ids()
    extra_ids.sort(reverse=True)
    underscore = tokenizer.convert_tokens_to_ids('▁')
    underscore_str = '▁'
    l = []
    for ti, text in tqdm.tqdm(enumerate(texts[:250 if ("ul2" in sys.argv or "flan-ul2" in sys.argv) else (500 if ("3b" in sys.argv or "11b" in sys.argv) else 5000)])
                          if (not deps or deps3) else enumerate(texts)):
        if len(text) < 100 or not text.isascii():
            continue
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs2 = BatchEncoding(inputs)
        r_inputs2 = BatchEncoding(inputs)

        if deps:
            doc = nlp(text)
            t2st = {i: doc.char_span(s, e, alignment_mode="contract") for i, (s, e) in enumerate(inputs.encodings[0].offsets)}
            def is_skip(i, j):
                if t2st[i] is None or t2st[j] is None:  # covers token that are not full words
                    return False
                return t2st[i][0].head.i == t2st[j][0].i and t2st[j][0].i > t2st[i][0].i and \
                       t2st[i][0].n_rights + t2st[i][0].n_lefts == 0 and t2st[i][0].dep_ in dep_tags
            def is_skip3(i, j):
                if t2st[i] is None or t2st[j] is None:  # covers token that are not full words
                    return False
                return t2st[j][0].i > t2st[i][0].i

        for i in range(1, len(inputs.input_ids[0]) - 4):
            j = i + 2
            if l1_2 or l2_1:
                if l2_1:
                    j = i + 3
                if i == len(inputs.input_ids[0]) - 5:
                    continue
                if tokenizer.convert_ids_to_tokens(inputs['input_ids'][0, i].item())[0] != underscore_str or \
                        tokenizer.convert_ids_to_tokens(inputs['input_ids'][0, j].item())[0] != underscore_str:
                    continue
            if deps and deps3:
                if is_skip(i, j) or not is_skip3(i, j):
                    continue
                # if inputs['input_ids'][0, i].item() != inputs['input_ids'][0, j].item():
                #     continue
                if tokenizer.convert_ids_to_tokens(inputs['input_ids'][0, i].item())[0] != underscore_str or \
                        tokenizer.convert_ids_to_tokens(inputs['input_ids'][0, j].item())[0] != underscore_str:
                    continue
            elif deps and not deps3:
                if tokenizer.convert_ids_to_tokens(inputs['input_ids'][0, i].item())[0] != underscore_str or \
                        tokenizer.convert_ids_to_tokens(inputs['input_ids'][0, j].item())[0] != underscore_str:
                    continue
                if not is_skip(i, j):
                    continue
            # TODO: batchify
            id1, id2 = inputs['input_ids'][0, i], inputs['input_ids'][0, j]

            # mask both
            inputs2['input_ids'] = torch.clone(inputs.input_ids)
            r_inputs2['input_ids'] = torch.clone(inputs.input_ids)
            labels = torch.full_like(inputs2['input_ids'], -100)
            r_labels = torch.full_like(inputs2['input_ids'], -100)
            if not l1_2 and not l2_1:
                # labels[0, :8] = torch.tensor([extra_ids[0], inputs2['input_ids'][0, i].item(), underscore, extra_ids[1],
                #                               inputs2['input_ids'][0, j].item(), underscore, extra_ids[2], 1])
                # r_labels[0, :8] = torch.tensor([extra_ids[0], inputs2['input_ids'][0, j].item(), underscore, extra_ids[1],
                #                                 inputs2['input_ids'][0, i].item(), underscore, extra_ids[2], 1])
                labels[0, :8] = torch.tensor([extra_ids[0], inputs2['input_ids'][0, i].item(), underscore, extra_ids[1],
                                              inputs2['input_ids'][0, j].item(), underscore, extra_ids[2], -100])
                r_labels[0, :8] = torch.tensor([extra_ids[0], inputs2['input_ids'][0, j].item(), underscore, extra_ids[1],
                                                inputs2['input_ids'][0, i].item(), underscore, extra_ids[2], -100])
                inputs2['input_ids'][0, j+3:] = inputs['input_ids'][0, j+1:-2]
                inputs2['input_ids'][0, i] = underscore
                inputs2['input_ids'][0, i+1] = extra_ids[0]
                inputs2['input_ids'][0, i+2] = inputs['input_ids'][0, i+1]
                inputs2['input_ids'][0, j+1] = underscore
                inputs2['input_ids'][0, j+2] = extra_ids[1]
                r_inputs2['input_ids'] = torch.clone(inputs2.input_ids)
                r_inputs2['input_ids'][0, i+1] = extra_ids[1]
                r_inputs2['input_ids'][0, j+2] = extra_ids[0]
            elif l1_2:
                labels[0, :9] = torch.tensor([extra_ids[0], inputs2['input_ids'][0, i].item(), underscore, extra_ids[1],
                                              inputs2['input_ids'][0, j].item(), inputs2['input_ids'][0, j+1].item(),
                                              underscore, extra_ids[2], 1])
                r_labels[0, :9] = torch.tensor([extra_ids[0], inputs2['input_ids'][0, j].item(),
                                                inputs2['input_ids'][0, j+1].item(), underscore, extra_ids[1], inputs2['input_ids'][0, i].item(),
                                                underscore, extra_ids[2], 1])
                inputs2['input_ids'][0, j+3:] = inputs['input_ids'][0, j+2:-1]
                inputs2['input_ids'][0, i] = underscore
                inputs2['input_ids'][0, i+1] = extra_ids[0]
                inputs2['input_ids'][0, i+2] = inputs['input_ids'][0, i+1]
                inputs2['input_ids'][0, j+1] = underscore
                inputs2['input_ids'][0, j+2] = extra_ids[1]
                r_inputs2['input_ids'] = torch.clone(inputs2.input_ids)
                r_inputs2['input_ids'][0, i+1] = extra_ids[1]
                r_inputs2['input_ids'][0, j+2] = extra_ids[0]
                #
                # inputs2['input_ids'][0, i] = extra_ids[0]
                # inputs2['input_ids'][0, j] = extra_ids[1]
                # inputs2['input_ids'][0, j+1: -1] = inputs['input_ids'][0, j+2:]
                # inputs2['input_ids'][0, -1] = tokenizer.pad_token_id
                # r_inputs2['input_ids'][0, i] = extra_ids[1]
                # r_inputs2['input_ids'][0, j] = extra_ids[0]
                # r_inputs2['input_ids'][0, j: -1] = inputs['input_ids'][0, j+1:]
                # r_inputs2['input_ids'][0, -1] = tokenizer.pad_token_id
            elif l2_1:
                labels[0, :9] = torch.tensor([extra_ids[0], inputs2['input_ids'][0, i].item(),
                                              inputs2['input_ids'][0, i+1].item(), underscore, extra_ids[1],
                                              inputs2['input_ids'][0, j].item(), underscore, extra_ids[2], 1])
                r_labels[0, :9] = torch.tensor([extra_ids[0], inputs2['input_ids'][0, j].item(), underscore, extra_ids[1],
                                                inputs2['input_ids'][0, i].item(), inputs2['input_ids'][0, i+1].item(),
                                                underscore, extra_ids[2], 1])
                inputs2['input_ids'][0, j+2:] = inputs['input_ids'][0, j+1:-1]
                inputs2['input_ids'][0, i] = underscore
                inputs2['input_ids'][0, i+1] = extra_ids[0]
                # inputs2['input_ids'][0, i+2] = inputs['input_ids'][0, i+2]
                inputs2['input_ids'][0, j] = underscore
                inputs2['input_ids'][0, j+1] = extra_ids[1]
                r_inputs2['input_ids'] = torch.clone(inputs2.input_ids)
                r_inputs2['input_ids'][0, i+1] = extra_ids[1]
                r_inputs2['input_ids'][0, j+1] = extra_ids[0]
                # inputs2['input_ids'][0, j] = extra_ids[1]
                # inputs2['input_ids'][0, i] = extra_ids[0]
                # # inputs2['input_ids'][0, i+1: -1] = inputs2['input_ids'][0, i+2:]
                # inputs2['input_ids'][0, :-1] = torch.cat([inputs2['input_ids'][0, :i+1], inputs2['input_ids'][0, i+2:]])
                # inputs2['input_ids'][0, -1] = tokenizer.pad_token_id
                # r_inputs2['input_ids'][0, j] = extra_ids[0]
                # r_inputs2['input_ids'][0, i] = extra_ids[1]
                # # r_inputs2['input_ids'][0, i+1: -1] = r_inputs2['input_ids'][0, i+2:]
                # r_inputs2['input_ids'][0, :-1] = torch.cat([r_inputs2['input_ids'][0, :i+1], r_inputs2['input_ids'][0, i+2:]])
                # r_inputs2['input_ids'][0, -1] = tokenizer.pad_token_id
            with torch.no_grad():
                out = model(**inputs2.to(dev), labels=labels.to(dev))
                r_out = model(**r_inputs2.to(dev), labels=r_labels.to(dev))
                out_x = model(**r_inputs2.to(dev), labels=labels.to(dev))
                r_out_x = model(**inputs2.to(dev), labels=r_labels.to(dev))

            logits = out.logits / temperature
            r_logits = r_out.logits / temperature
            # probs = logits.log_softmax(dim=-1).cpu().numpy()
            if not l1_2 and not l2_1:
                # TODO: does this actually represent language modeling??
                # _p12 = -out.loss * 8
                # _p21 = -r_out.loss * 8
                # _p12x = -out_x.loss * 8
                # _p21x = -r_out_x.loss * 8
                _p12 = -out.loss * 2
                _p21 = -r_out.loss * 2
                _p12x = -out_x.loss * 2
                _p21x = -r_out_x.loss * 2
            else:
                _p12 = -out.loss * 9
                _p21 = -r_out.loss * 9
                _p12x = -out_x.loss * 9
                _p21x = -r_out_x.loss * 9
            l.append((float(_p12), float(_p12), float(_p21), [id1.item(), id2.item()], [0., 0., 0., 0.], [ti, i], float(_p12x), float(_p21x)))
            # l.append((float(_pair_prob), float(_p12), float(_p21), [tokenizer.decode(id1.item()), tokenizer.decode(id2.item())], entropies))
    return l

def pair_probabilities3(texts, model, tokenizer, temperature=1., deps=False):
    """
    Measures probabilities for adjacent token pairs in the given texts
    :param texts: list of strings
    :param model:
    :param tokenizer:
    :return:
    """
    print("No extra underscore!!!")
    add_prompt_examples = "prompt_example" in sys.argv or "prompt_example2" in sys.argv or "i_prompt_example" in sys.argv or "i_prompt_example2" in sys.argv
    print(f"Add prompt examples: {add_prompt_examples}")
    # print("using restrictions for deps3")
    flan = "flan" in sys.argv
    llama = "llama" in sys.argv
    t5 = "t5" in sys.argv
    if flan or llama:
        print("Adding mask filling instruction")
    print("doing without end token")
    l1_2 = "l1_2" in sys.argv
    l2_1 = "l2_1" in sys.argv
    l2_2 = "l2_2" in sys.argv
    l1_3 = "l1_3" in sys.argv
    l3_1 = "l3_1" in sys.argv
    deps3 = "deps3" in sys.argv
    deps4 = "deps4" in sys.argv
    if deps4:
        deps3 = True
    if deps3:
        deps = True
    # if deps:
    import spacy
    nlp = spacy.load("en_core_web_md")
    # dep_tags = ["amod"]
    dep_tags = ["amod", "compound", "nummod", "poss"]
    # if deps4:
    #     dep_tags = ["obj", "dobj"]
    print(f"Using only {dep_tags}")
    print("For deps3 avoiding all backward tags")
    print("Avoiding spacy stopwords, punctuation, digits")

    if t5:
        extra_ids = tokenizer.get_sentinel_token_ids()
        extra_ids.sort(reverse=True)
        extra_tokens = tokenizer.convert_ids_to_tokens(extra_ids)
        underscore = tokenizer.convert_tokens_to_ids('▁')
    if llama:
        extra_tokens = ["<extra_id_0>", "<extra_id_1>", "<extra_id_2>"]
        tokenizer.add_tokens(new_tokens=extra_tokens)
        extra_ids = [tokenizer.added_tokens_encoder[t] for t in extra_tokens]
        pass
    underscore_str = '▁'
    l = []
    # prompt = "Fill in the words to replace the special tokens (<extra_id_0>, <extra_id_1>) in the following sentence. The answer should end with <extra_id_2>." \
    #          "\nSentence: "
    prompt = "Fill in the words to replace the sentinel tokens. The sentinel tokens can appeare in the sentence in differenent orders. The answer should always start with the sentinel token number 0." \
             "\nSentence: "
    if add_prompt_examples:
        prompt = f"Here are examples of filling masks in sentences. " \
                 f"The answer outputs <extra_id_0> and a word for it, <extra_id_1> and a word for it, and ends with <extra_id_2>." \
                f"\nExample #1, Sentence: The{extra_tokens[0]} walks in{extra_tokens[1]} park" \
                f"\nAnswer: {extra_tokens[0]} dog{extra_tokens[1]} the{extra_tokens[2]}" \
                f"\nExample #2, Sentence: I{extra_tokens[1]} breakfast every{extra_tokens[0]}" \
                f"\nAnswer: {extra_tokens[0]} morning{extra_tokens[1]} eat{extra_tokens[2]}\n" \
                f"\nFill the masks in the following sentence: "
        # prompt = f"Fill in the words to replace the special tokens (<extra_id_0>, <extra_id_1>). The answer should end with <extra_id_2>." \
        #          f"\nExample #1, Sentence: I{extra_tokens[0]} breakfast every{extra_tokens[1]}" \
        #          f"\nAnswer: {extra_tokens[0]} eat{extra_tokens[1]} morning{extra_tokens[2]}\n"
        if "prompt_example2" in sys.argv:
            # prompt = f"Fill in the words to replace the special tokens (<extra_id_0>, <extra_id_1>). The answer should end with <extra_id_2>." \
            #        f"\nExample #1, Sentence: I{extra_tokens[1]} breakfast every{extra_tokens[0]}" \
            #        f"\nAnswer: {extra_tokens[0]} morning{extra_tokens[1]} eat{extra_tokens[2]}" \
            #        f"\nExample #2, Sentence: The{extra_tokens[0]} walks in{extra_tokens[1]} park" \
            #        f"\nAnswer: {extra_tokens[0]} dog{extra_tokens[1]} the{extra_tokens[2]}\n"
            prompt = f"Fill in the words to replace the special tokens (<extra_id_1>, <extra_id_0>). The answer should end with <extra_id_2>." \
                   f"\nExample #1, Sentence: I{extra_tokens[1]} breakfast every{extra_tokens[0]}" \
                   f"\nAnswer: {extra_tokens[0]} morning{extra_tokens[1]} eat{extra_tokens[2]}\n"
                   # f"\nExample #2, Sentence: The{extra_tokens[0]} walks in{extra_tokens[1]} park" \
                   # f"\nAnswer: {extra_tokens[0]} dog{extra_tokens[1]} the{extra_tokens[2]}\n"
        if "i_prompt_example" in sys.argv:
            prompt = f"Fill in the words to replace the special tokens (<extra_id_0>, <extra_id_1>). The answer should end with <extra_id_2>." \
                   f"\nExample #1, Sentence: The{extra_tokens[1]} walks in{extra_tokens[0]} park" \
                   f"\nAnswer: {extra_tokens[0]} the{extra_tokens[1]} dog{extra_tokens[2]}" \
                   f"\nExample #2, Sentence: The{extra_tokens[0]} walks in{extra_tokens[1]} park" \
                   f"\nAnswer: {extra_tokens[0]} dog{extra_tokens[1]} the{extra_tokens[2]}\n"
        if "i_prompt_example2" in sys.argv:
            # prompt = f"Fill in the words to replace the special tokens (<extra_id_1>, <extra_id_0>). The answer should end with <extra_id_2>." \
            #        f"\nExample #1, Sentence: The{extra_tokens[1]} walks in{extra_tokens[0]} park" \
            #        f"\nAnswer: {extra_tokens[0]} the{extra_tokens[1]} dog{extra_tokens[2]}\n"
            prompt = f"Fill in the words to replace the special tokens (<extra_id_1>, <extra_id_0>). The answer should end with <extra_id_2>." \
                   f"\nExample #1, Sentence: The{extra_tokens[0]} walks in{extra_tokens[1]} park" \
                   f"\nAnswer: {extra_tokens[0]} dog{extra_tokens[1]} the{extra_tokens[2]}" \
                   f"\nExample #2, Sentence: The{extra_tokens[1]} walks in{extra_tokens[0]} park" \
                   f"\nAnswer: {extra_tokens[0]} the{extra_tokens[1]} dog{extra_tokens[2]}\n"
    print("Prompt: ")
    print(prompt)
    for ti, text in tqdm.tqdm(enumerate(texts[:250 if ("ul2" in sys.argv or "flan-ul2" in sys.argv) else (500 if ("3b" in sys.argv or "11b" in sys.argv) else 5000)])
                          if (not deps or deps3) else enumerate(texts)):
        if len(text) < 100 or not text.isascii():
            continue
        if flan:
            # text = prompt + "\nSentence: " + text + "\nAnswer: "
            text = prompt + text + "\nAnswer: "
        if llama and add_prompt_examples:
            text = f"Fill in the masks according to the mask tokens. " \
                    f"Example #1, Sentence: The{extra_tokens[0]} walks in{extra_tokens[1]} park" \
                    f"Answer: {extra_tokens[0]} dog{extra_tokens[1]} the{extra_tokens[2]}" \
                    f"Example #2, Sentence: I{extra_tokens[1]} breakfast every{extra_tokens[0]}" \
                    f"Answer: {extra_tokens[0]} morning{extra_tokens[1]} eat{extra_tokens[2]}" \
                    "\nSentence: " + text + "\nAnswer: "
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs2 = BatchEncoding(inputs)
        r_inputs2 = BatchEncoding(inputs)

        # if deps:
        doc = nlp(text)
        t2st = {i: doc.char_span(s, e, alignment_mode="contract") for i, (s, e) in enumerate(inputs.encodings[0].offsets)}
        def is_stop(i, j):
            # check if words are problematic
            if t2st[i] is None or t2st[j] is None:  # covers token that are not full words
                return True
            if t2st[i][0].is_stop or t2st[j][0].is_stop or t2st[i][0].is_digit or t2st[j][0].is_digit \
                    or t2st[i][0].is_punct or t2st[j][0].is_punct:
                return True

        def is_skip4(i, j):
            if t2st[i] is None or t2st[j] is None:  # covers token that are not full words
                return False
            return t2st[j][0].head.i == t2st[i][0].i and t2st[j][0].i > t2st[i][0].i and \
                   t2st[j][0].n_rights + t2st[j][0].n_lefts == 0 and t2st[j][0].dep_ in dep_tags
            # return t2st[i][0].pos_ in ["ADV", "ADJ", "ADP"] and t2st[j][0].pos_ in ["NOUN", "PRON", "PROPN"] or \
            #        t2st[j][0].pos_ in ["ADV", "ADJ", "ADP"] and t2st[i][0].pos_ in ["NOUN", "PRON", "PROPN"]
        def is_skip(i, j, all_tags=False):
            # if t2st[i] is None or t2st[j] is None:  # covers token that are not full words
            #     return False
            return t2st[i][0].head.i == t2st[j][0].i and t2st[j][0].i > t2st[i][0].i and \
                   t2st[i][0].n_rights + t2st[i][0].n_lefts == 0 and (all_tags or t2st[i][0].dep_ in dep_tags)
        def is_skip3(i, j):
            if t2st[i] is None or t2st[j] is None:  # covers token that are not full words
                return False
            return t2st[j][0].i > t2st[i][0].i

        for i in range(1, len(inputs.input_ids[0]) - 4):
            j = i + 2
            if l1_2 or l2_1 or l1_3 or l3_1 or l2_2:
                if l2_1 or l2_2:
                    j = i + 3
                if l3_1:
                    j = i + 4
                if i == len(inputs.input_ids[0]) - 6:
                    continue
                if tokenizer.convert_ids_to_tokens(inputs['input_ids'][0, i].item())[0] != underscore_str or \
                        tokenizer.convert_ids_to_tokens(inputs['input_ids'][0, j].item())[0] != underscore_str:
                    continue
            if flan and (i < (30 if not add_prompt_examples else 50)
                         or len(inputs.input_ids[0]) - i < (8 if not add_prompt_examples else 30)):
                continue
            if is_stop(i, j):
                continue
            if deps and deps3:
                if deps4:
                    if is_skip(i, j) or not is_skip4(i, j):
                        continue
                else:
                    # if is_skip(i, j) or not is_skip3(i, j):
                    #     continue
                    if is_skip(i, j, all_tags=True):
                        continue
                # if inputs['input_ids'][0, i].item() != inputs['input_ids'][0, j].item():
                #     continue

            elif deps and not deps3 and not is_skip(i, j):
                continue
            if tokenizer.convert_ids_to_tokens(inputs['input_ids'][0, i].item())[0] != underscore_str or \
                    tokenizer.convert_ids_to_tokens(inputs['input_ids'][0, j].item())[0] != underscore_str:
                continue
            if inputs['input_ids'][0, i].item() == inputs['input_ids'][0, j].item():
                continue
            # TODO: batchify
            id1, id2 = inputs['input_ids'][0, i], inputs['input_ids'][0, j]

            # mask both
            inputs2['input_ids'] = torch.clone(inputs.input_ids)
            r_inputs2['input_ids'] = torch.clone(inputs.input_ids)
            labels = torch.full_like(inputs2['input_ids'], -100)
            r_labels = torch.full_like(inputs2['input_ids'], -100)
            if not l1_2 and not l2_1 and not l1_3 and not l3_1 and not l2_2:
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
                                              inputs2['input_ids'][0, j].item(), inputs2['input_ids'][0, j+1].item(),
                                              extra_ids[2], -100])
                r_labels[0, :7] = torch.tensor([extra_ids[0], inputs2['input_ids'][0, j].item(),
                                                inputs2['input_ids'][0, j+1].item(), extra_ids[1], inputs2['input_ids'][0, i].item(),
                                                extra_ids[2], -100])
                inputs2['input_ids'][0, j+1:-1] = inputs['input_ids'][0, j+2:]
                inputs2['input_ids'][0, i] = extra_ids[0]
                inputs2['input_ids'][0, j] = extra_ids[1]
                r_inputs2['input_ids'] = torch.clone(inputs2.input_ids)
                r_inputs2['input_ids'][0, i] = extra_ids[1]
                r_inputs2['input_ids'][0, j] = extra_ids[0]
            elif l2_1:
                labels[0, :7] = torch.tensor([extra_ids[0], inputs2['input_ids'][0, i].item(),
                                              inputs2['input_ids'][0, i+1].item(), extra_ids[1],
                                              inputs2['input_ids'][0, j].item(), extra_ids[2], -100])
                r_labels[0, :7] = torch.tensor([extra_ids[0], inputs2['input_ids'][0, j].item(), extra_ids[1],
                                                inputs2['input_ids'][0, i].item(), inputs2['input_ids'][0, i+1].item(),
                                                extra_ids[2], -100])
                inputs2['input_ids'][0, i+1:-1] = inputs['input_ids'][0, i+2:]
                inputs2['input_ids'][0, i] = extra_ids[0]
                inputs2['input_ids'][0, j-1] = extra_ids[1]
                r_inputs2['input_ids'] = torch.clone(inputs2.input_ids)
                r_inputs2['input_ids'][0, i] = extra_ids[1]
                r_inputs2['input_ids'][0, j-1] = extra_ids[0]
            elif l3_1:
                labels[0, :8] = torch.tensor([extra_ids[0], inputs2['input_ids'][0, i].item(),
                                              inputs2['input_ids'][0, i+1].item(), inputs2['input_ids'][0, i+2].item(), extra_ids[1],
                                              inputs2['input_ids'][0, j].item(), extra_ids[2], -100])
                r_labels[0, :8] = torch.tensor([extra_ids[0], inputs2['input_ids'][0, j].item(), extra_ids[1],
                                                inputs2['input_ids'][0, i].item(), inputs2['input_ids'][0, i+1].item(), inputs2['input_ids'][0, i+2].item(),
                                                extra_ids[2], -100])
                inputs2['input_ids'][0, i+1:-2] = inputs['input_ids'][0, i+3:]
                inputs2['input_ids'][0, -2:] = 0
                inputs2['input_ids'][0, i] = extra_ids[0]
                inputs2['input_ids'][0, j-2] = extra_ids[1]  # = i+2
                r_inputs2['input_ids'] = torch.clone(inputs2.input_ids)
                r_inputs2['input_ids'][0, i] = extra_ids[1]
                r_inputs2['input_ids'][0, j-2] = extra_ids[0]
            elif l1_3:
                labels[0, :8] = torch.tensor([extra_ids[0], inputs2['input_ids'][0, i].item(), extra_ids[1],
                                              inputs2['input_ids'][0, j].item(), inputs2['input_ids'][0, j+1].item(), inputs2['input_ids'][0, j+2].item(), extra_ids[2], -100])
                r_labels[0, :8] = torch.tensor([extra_ids[0], inputs2['input_ids'][0, j].item(),
                                                inputs2['input_ids'][0, j+1].item(), inputs2['input_ids'][0, j+2].item(), extra_ids[1],
                                                inputs2['input_ids'][0, i].item(),
                                                extra_ids[2], -100])
                inputs2['input_ids'][0, j+1:-2] = inputs['input_ids'][0, j+3:]
                inputs2['input_ids'][0, -2:] = 0
                inputs2['input_ids'][0, i] = extra_ids[0]
                inputs2['input_ids'][0, j] = extra_ids[1]  # = i+2
                r_inputs2['input_ids'] = torch.clone(inputs2.input_ids)
                r_inputs2['input_ids'][0, i] = extra_ids[1]
                r_inputs2['input_ids'][0, j] = extra_ids[0]
            elif l2_2:
                labels[0, :8] = torch.tensor([extra_ids[0], inputs2['input_ids'][0, i].item(),
                                              inputs2['input_ids'][0, i+1].item(), extra_ids[1],
                                              inputs2['input_ids'][0, j].item(), inputs2['input_ids'][0, j+1].item(), extra_ids[2], -100])
                r_labels[0, :8] = torch.tensor([extra_ids[0], inputs2['input_ids'][0, j].item(), inputs2['input_ids'][0, j+1].item(), extra_ids[1],
                                                inputs2['input_ids'][0, i].item(), inputs2['input_ids'][0, i+1].item(),
                                                extra_ids[2], -100])
                inputs2['input_ids'][0, i+3:-2] = inputs['input_ids'][0, i+5:]
                inputs2['input_ids'][0, -2:] = 0
                inputs2['input_ids'][0, i] = extra_ids[0]
                inputs2['input_ids'][0, i+1] = inputs['input_ids'][0, i+2]
                inputs2['input_ids'][0, j-1] = extra_ids[1]  # i+2
                r_inputs2['input_ids'] = torch.clone(inputs2.input_ids)
                r_inputs2['input_ids'][0, i] = extra_ids[1]
                r_inputs2['input_ids'][0, j-1] = extra_ids[0]
            with torch.no_grad():
                out = model(**inputs2.to(dev), labels=labels.to(dev))
                r_out = model(**r_inputs2.to(dev), labels=r_labels.to(dev))
                out_x = model(**r_inputs2.to(dev), labels=labels.to(dev))
                r_out_x = model(**inputs2.to(dev), labels=r_labels.to(dev))

            logits = out.logits / temperature
            r_logits = r_out.logits / temperature
            if not l1_2 and not l2_1 and not l1_3 and not l3_1 and not l2_2:
                # TODO: does this actually represent language modeling??
                _p12 = -out.loss * 2
                _p21 = -r_out.loss * 2
                _p12x = -out_x.loss * 2
                _p21x = -r_out_x.loss * 2
            else:
                _p12 = -out.loss * 3
                _p21 = -r_out.loss * 3
                _p12x = -out_x.loss * 3
                _p21x = -r_out_x.loss * 3
            l.append((float(_p12), float(_p12), float(_p21), [id1.item(), id2.item()], [0., 0., 0., 0.], [ti, i], float(_p12x), float(_p21x)))
            # l.append((float(_pair_prob), float(_p12), float(_p21), [tokenizer.decode(id1.item()), tokenizer.decode(id2.item())], entropies))
    return l

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

    # what about spaces???
    if "llama" not in sys.argv:
        # maybe use # instead??
        extra_ids = tokenizer.convert_tokens_to_ids(["%", "@"])  # extra_ids[0] is the one to predict
        if "other_prompt2" in sys.argv:
            extra_ids = tokenizer.convert_tokens_to_ids(["^", "$"])  # extra_ids[0] is the one to predict
    else:
        extra_ids = tokenizer("%")["input_ids"][1], tokenizer("@")["input_ids"][1]  # extra_ids[0] is the one to predict
        if "other_prompt2" in sys.argv:
            extra_ids = tokenizer("^")["input_ids"][1], tokenizer("$")["input_ids"][1]  # extra_ids[0] is the one to predict
        # space_id = tokenizer(" %")["input_ids"][1]
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
                # if t2st[i] is None or t2st[j] is None:  # covers token that are not full words
                #     return False
                return t2st[i][0].head.i == t2st[j][0].i and t2st[j][0].i > t2st[i][0].i and \
                       t2st[i][0].n_rights + t2st[i][0].n_lefts == 0 and (all_tags or t2st[i][0].dep_ in dep_tags)
        if fixed_pos is None:
            # if first:
            #     print(text)
            #     first = False
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
            # token_range = range(1, len(inputs.input_ids[0]) - 4)
        # elif "flan" not in sys.argv:
        else:
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs2 = BatchEncoding(inputs)
            # inputs3 = BatchEncoding(inputs)
            new_fixed_pos = len(inputs["input_ids"][0]) - tail_len - text_len + fixed_pos
            token_range = range(new_fixed_pos, new_fixed_pos+1)
        # else:
        #     token_range = range(fixed_pos, fixed_pos+1)

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
            # if (fixed_pos is None or "flan" not in sys.argv) and is_skip(i, j, all_tags=False):
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
                    # labels = tokenizer(f": {o_text.split()[0]}", return_tensors="pt", padding=True, truncation=True)["input_ids"]
                    labels = tokenizer(f" {o_text.split()[0]}", return_tensors="pt", padding=True, truncation=True)["input_ids"]
                    # i = ((inputs2["input_ids"][0] == extra_ids[0]).nonzero(as_tuple=True)[0][-1])
                    # j = i + 1
                else:
                    if "other_prompt2" in sys.argv:
                        n_text = text[:t2st[i].start_char] + "^" + text[t2st[i].end_char:]
                    else:
                        n_text = text[:t2st[i].start_char] + "%" + text[t2st[i].end_char:]
                    inputs2 = tokenizer(n_text, return_tensors="pt", padding=True, truncation=True)
                    labels = tokenizer(t2st[i].text, return_tensors="pt", padding=True, truncation=True)["input_ids"]

                    # inputs2['input_ids'] = torch.clone(inputs.input_ids)
                    # # labels = torch.tensor([colon_id, inputs2['input_ids'][0, i].item(), tokenizer.eos_token_id]).unsqueeze(0)
                    # labels = torch.tensor([inputs2['input_ids'][0, i].item(), tokenizer.eos_token_id]).unsqueeze(0)
                    # inputs2['input_ids'][0, i] = extra_ids[0]
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
                    # labels = tokenizer(f": {o_text.split()[1]}", return_tensors="pt", padding=True, truncation=True)["input_ids"]
                    labels = tokenizer(f" {o_text.split()[1]}", return_tensors="pt", padding=True, truncation=True)["input_ids"]
                else:
                    if "other_prompt2" in sys.argv:
                        n_text = text[:t2st[j].start_char] + "$" + text[t2st[j].end_char:]
                    else:
                        n_text = text[:t2st[j].start_char] + "@" + text[t2st[j].end_char:]
                    inputs2 = tokenizer(n_text, return_tensors="pt", padding=True, truncation=True)
                    labels = tokenizer(t2st[j].text, return_tensors="pt", padding=True, truncation=True)["input_ids"]

                    # inputs2['input_ids'] = torch.clone(inputs.input_ids)
                    # labels = torch.tensor([inputs2['input_ids'][0, j].item(), tokenizer.eos_token_id]).unsqueeze(0)
                    # inputs2['input_ids'][0, j] = extra_ids[0]
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
                    # labels = tokenizer(f": {o_text.split()[0]}", return_tensors="pt", padding=True, truncation=True)["input_ids"]
                    labels = tokenizer(f" {o_text.split()[0]}", return_tensors="pt", padding=True, truncation=True)["input_ids"]
                else:
                    if "other_prompt2" in sys.argv:
                        n_text = text[:t2st[i].start_char] + "^ $" + text[t2st[j].end_char:]
                    else:
                        n_text = text[:t2st[i].start_char] + "% @" + text[t2st[j].end_char:]
                    inputs2 = tokenizer(n_text, return_tensors="pt", padding=True, truncation=True)
                    labels = tokenizer(t2st[i].text, return_tensors="pt", padding=True, truncation=True)["input_ids"]

                    # inputs2['input_ids'] = torch.clone(inputs.input_ids)
                    # # labels = torch.tensor([colon_id, inputs2['input_ids'][0, i].item(), tokenizer.eos_token_id]).unsqueeze(0)
                    # labels = torch.tensor([inputs2['input_ids'][0, i].item(), tokenizer.eos_token_id]).unsqueeze(0)
                    # inputs2['input_ids'][0, i] = extra_ids[0]
                    # inputs2['input_ids'][0, j] = extra_ids[1]
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
                    # labels = tokenizer(f": {o_text.split()[0]}", return_tensors="pt", padding=True, truncation=True)["input_ids"]
                    labels = tokenizer(f" {o_text.split()[0]}", return_tensors="pt", padding=True, truncation=True)["input_ids"]
                else:
                    if "other_prompt2" in sys.argv:
                        n_text = text[:t2st[i].start_char] + "^ $" + text[t2st[j].end_char:]
                    else:
                        n_text = text[:t2st[i].start_char] + "% @" + text[t2st[j].end_char:]
                    inputs2 = tokenizer(n_text, return_tensors="pt", padding=True, truncation=True)
                    labels = tokenizer(t2st[j].text, return_tensors="pt", padding=True, truncation=True)["input_ids"]

                    # inputs2['input_ids'] = torch.clone(inputs.input_ids)
                    # # labels = torch.tensor([colon_id, inputs2['input_ids'][0, j].item(), tokenizer.eos_token_id]).unsqueeze(0)
                    # labels = torch.tensor([inputs2['input_ids'][0, j].item(), tokenizer.eos_token_id]).unsqueeze(0)
                    # inputs2['input_ids'][0, i] = extra_ids[1]
                    # inputs2['input_ids'][0, j] = extra_ids[0]
            else:
                _labels = torch.tensor([inputs['input_ids'][0, j].item(), tokenizer.eos_token_id]).unsqueeze(0)
                inputs2['input_ids'] = torch.cat([torch.clone(inputs.input_ids)[:, :-1], _labels], dim=1)
                labels = torch.clone(inputs2['input_ids'])
                inputs2['input_ids'][0, i] = extra_ids[1]
                inputs2['input_ids'][0, j] = extra_ids[0]
                labels[0, :-2] = -100
            with torch.no_grad():
                # out3_b = model(**inputs2.to(dev), labels=labels_b.to(dev))
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
                    # print(len(_ranks))
                    # print(eos_id)
                    # print(inputs['input_ids'][0, i].item())
                    # _ = _ranks2[eos_id]
                    # _ = _ranks[inputs['input_ids'][0, i].item()]
                    # _ = _ranks1[inputs['input_ids'][0, j].item()]
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
            # if t2st[i] is None or t2st[i+1] is None or t2st[i][0].is_punct or t2st[i+1][0].is_punct or t2st[i][0].is_stop or t2st[i+1][0].is_stop:  # covers token that are not full words
            # if t2st[i] is None or t2st[i+1] is None or t2st[i][0].pos_ not in POS_tags or t2st[i+1][0].pos_ not in POS_tags:  # covers token that are not full words
            # if t2st[i] is None or t2st[i+1] is None:  # covers token that are not full words
            #     return False
            # if t2st[i].is_stop or t2st[i+1].is_stop or t2st[i].is_digit or t2st[i+1].is_digit or t2st[i].is_punct or t2st[i+1].is_punct:
            #     return False
            # return t2st[i+1][0].head.i < t2st[i][0].i and t2st[i][0].head.i == t2st[i+1][0].i and t2st[i+1][0].i > t2st[i][0].i
            if not deps2:  # an arc going "backwards"
                return t2st[i][0].head.i == t2st[i+1][0].i and t2st[i+1][0].i > t2st[i][0].i and \
                       t2st[i][0].n_rights + t2st[i][0].n_lefts == 0 and (all_tags or t2st[i][0].dep_ in dep_tags)
            else:
                return t2st[i+1][0].head.i == t2st[i][0].i and t2st[i+1][0].i > t2st[i][0].i and t2st[i+1][0].n_rights + t2st[i+1][0].n_lefts == 0

        if fixed_pos is None:
            token_range = range(1, len(inputs.input_ids[0]) - 2)
        else:
            # inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            # inputs2 = BatchEncoding(inputs)
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
            # labels[i: i+2] = torch.tensor([inputs2['input_ids'][0, i].item(), inputs2['input_ids'][0, i+1].item()])
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
            # logits3a = out3_a.logits / temperature
            # logits3b = out3_b.logits / temperature
            logits4 = out4.logits / temperature
            # probs = logits.log_softmax(dim=-1).cpu().numpy()
            # probs2 = logits2.log_softmax(dim=-1).cpu().numpy()
            # probs3a = logits3a.log_softmax(dim=-1).cpu().numpy()
            # probs3b = logits3b.log_softmax(dim=-1).cpu().numpy()
            # probs4 = logits4.log_softmax(dim=-1).cpu().numpy()
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


def in_nucleus(logits, i, p=0.95):
    probs = logits.softmax(dim=-1)
    sorted_probs, indices = torch.sort(probs, dim=-1, descending=True)
    cum_sum_probs = torch.cumsum(sorted_probs, dim=-1)
    # Find the cumulative sums less than $p$.
    nucleus = cum_sum_probs < p
    return i in indices[sum(nucleus)]


def to_bin(p):
    from bisect import bisect
    intervals = [0, 0.25, 0.5, 1, 2, 3, 5, 7.5, 10, 15, 20, 30, 40, 50, 75, 100, 125, 150]
    return intervals[bisect(intervals, p)-1]


def boxplot(name="", ylim=None, bins=None, dataset="wikitext-2", by_prob=False):
    # box plot of absolute log-distance
    import json

    with open(f'/cs/snapless/oabend/eitan.wagner/calibration/{dataset}_scores_{name}.json', 'r') as file:
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
    use_underscore = "use_underscore" in sys.argv

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

    with open(f'/cs/snapless/oabend/eitan.wagner/calibration/{dataset_name}_scores_{name.split("/")[-1]}'
              f'{"_deps" if deps else ""}{"_l1_2" if l1_2 else ""}{"_l2_1" if l2_1 else ""}{"_u" if use_underscore else "_"}.json', 'r') as infile:
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

    with open(f'/cs/snapless/oabend/eitan.wagner/calibration/news-2.7.2023.json', 'r') as file:
        texts = [t['text'] for t in json.load(file) if t['title'].find("Subscribe") == -1]
    with open(f'/cs/snapless/oabend/eitan.wagner/calibration/news-6.7.2023.json', 'r') as file:
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
                if use_underscore:
                    labels[0, :8] = torch.tensor([extra_ids[0], inputs2['input_ids'][0, i].item(), underscore, extra_ids[1],
                                                  inputs2['input_ids'][0, j].item(), underscore, extra_ids[2], -100])
                    r_labels[0, :8] = torch.tensor([extra_ids[0], inputs2['input_ids'][0, j].item(), underscore, extra_ids[1],
                                                    inputs2['input_ids'][0, i].item(), underscore, extra_ids[2], -100])
                    inputs2['input_ids'][0, j + 3:] = inputs['input_ids'][0, j + 1:-2]
                    inputs2['input_ids'][0, i] = underscore
                    inputs2['input_ids'][0, i + 1] = extra_ids[0]
                    inputs2['input_ids'][0, i + 2] = inputs['input_ids'][0, i + 1]
                    inputs2['input_ids'][0, j + 1] = underscore
                    inputs2['input_ids'][0, j + 2] = extra_ids[1]
                    r_inputs2['input_ids'] = torch.clone(inputs2.input_ids)
                    r_inputs2['input_ids'][0, i + 1] = extra_ids[1]
                    r_inputs2['input_ids'][0, j + 2] = extra_ids[0]
                else:
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
                if use_underscore:
                    labels[0, :9] = torch.tensor([extra_ids[0], inputs2['input_ids'][0, i].item(), underscore, extra_ids[1],
                                                  inputs2['input_ids'][0, j].item(), inputs2['input_ids'][0, j + 1].item(),
                                                  underscore, extra_ids[2], 1])
                    r_labels[0, :9] = torch.tensor([extra_ids[0], inputs2['input_ids'][0, j].item(),
                                                    inputs2['input_ids'][0, j + 1].item(), underscore, extra_ids[1],
                                                    inputs2['input_ids'][0, i].item(),
                                                    underscore, extra_ids[2], 1])
                    inputs2['input_ids'][0, j + 3:] = inputs['input_ids'][0, j + 2:-1]
                    inputs2['input_ids'][0, i] = underscore
                    inputs2['input_ids'][0, i + 1] = extra_ids[0]
                    inputs2['input_ids'][0, i + 2] = inputs['input_ids'][0, i + 1]
                    inputs2['input_ids'][0, j + 1] = underscore
                    inputs2['input_ids'][0, j + 2] = extra_ids[1]
                    r_inputs2['input_ids'] = torch.clone(inputs2.input_ids)
                    r_inputs2['input_ids'][0, i + 1] = extra_ids[1]
                    r_inputs2['input_ids'][0, j + 2] = extra_ids[0]
                else:
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
                if use_underscore:
                    labels[0, :9] = torch.tensor([extra_ids[0], inputs2['input_ids'][0, i].item(),
                                                  inputs2['input_ids'][0, i + 1].item(), underscore, extra_ids[1],
                                                  inputs2['input_ids'][0, j].item(), underscore, extra_ids[2], 1])
                    r_labels[0, :9] = torch.tensor([extra_ids[0], inputs2['input_ids'][0, j].item(), underscore, extra_ids[1],
                                                    inputs2['input_ids'][0, i].item(), inputs2['input_ids'][0, i + 1].item(),
                                                    underscore, extra_ids[2], 1])
                    inputs2['input_ids'][0, j + 2:] = inputs['input_ids'][0, j + 1:-1]
                    inputs2['input_ids'][0, i] = underscore
                    inputs2['input_ids'][0, i + 1] = extra_ids[0]
                    inputs2['input_ids'][0, j] = underscore
                    inputs2['input_ids'][0, j + 1] = extra_ids[1]
                    r_inputs2['input_ids'] = torch.clone(inputs2.input_ids)
                    r_inputs2['input_ids'][0, i + 1] = extra_ids[1]
                    r_inputs2['input_ids'][0, j + 1] = extra_ids[0]
                else:
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
            print(tokenizer.decode(labels[0][:7 if use_underscore else 5]))
            print(out.loss)
            print(probs[0, range(8 if use_underscore else 6), labels.numpy()[0][:8 if use_underscore else 6]])
            print(tokenizer.decode(r_inputs2['input_ids'][0]))
            print(tokenizer.decode(r_labels[0][:7 if use_underscore else 5]))
            print(r_out.loss)
            print(r_probs[0, range(8 if use_underscore else 6), r_labels.numpy()[0][:8 if use_underscore else 6]])

            print("reversing:")
            print(out_x.loss)
            print(probs_x[0, range(8 if use_underscore else 6), labels.numpy()[0][:8 if use_underscore else 6]])
            print(r_out_x.loss)
            print(r_probs_x[0, range(8 if use_underscore else 6), r_labels.numpy()[0][:8 if use_underscore else 6]])

            print("\n")
    print("\n\n")


# ****************************

def load_pp(dataset, name):
    f_path = f'/cs/snapless/oabend/eitan.wagner/calibration/{dataset}_scores_{name}_ranks.json'
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
    size = "small"
    batch_size = 64
    random = "random" in sys.argv
    topics = "topics" in sys.argv
    t_list = "t_list" in sys.argv
    single_token = "single_token" in sys.argv
    on_data = "on_data" in sys.argv
    entropies = "entropies" in sys.argv
    new_news = "new_news" in sys.argv
    shuffle = "shuffle" in sys.argv
    dataset = "wikitext-2"
    if new_news:
        dataset = "new_news"
    elif "new_news2" in sys.argv:
        dataset = "new_news2"
    if "synthetic" in sys.argv:
        dataset = "total_noise" if "total_noise" in sys.argv else "extra_noise" if "extra_noise" in sys.argv else "NPs" if "noise" not in sys.argv else "noise"

    version = ""
    if "v1_1" in sys.argv:
        version = "v1_1"
    if "flan" in sys.argv:
        version = "flan"
    if "ul2" in sys.argv:
        version = "ul2"
    if "flan-ul2" in sys.argv:
        version = "flan-ul2"
    if "base" in sys.argv:
        size = "base"
        batch_size = 16
    elif "large" in sys.argv:
        size = "large"
        batch_size = 8
    elif "xlarge" in sys.argv:
        size = "xlarge"
        batch_size = 2
    elif "xxlarge" in sys.argv:
        size = "xxlarge"
        batch_size = 1
    elif "3b" in sys.argv:
        size = "3b"
        batch_size = 8
    elif "xl" in sys.argv:
        size = "xl"
        batch_size = 8
    elif "11b" in sys.argv:
        size = "11b"
        batch_size = 2
    elif "7b" in sys.argv:
        size = "7b"
        batch_size = 2
    elif "13b" in sys.argv:
        size = "13b"
        batch_size = 1
    elif "xxl" in sys.argv:
        size = "xxl"
        batch_size = 2
    elif "finetuned" in sys.argv:
        size = f"/cs/snapless/oabend/eitan.wagner/segmentation/models/t5-masked-{size}"
        # batch_size = 64
    # else:
    #     batch_size = 64
    if "--template" in sys.argv:
        temp = sys.argv[sys.argv.index("--template") + 1]

    temperature = None
    if "--temperature" in sys.argv:
        temperature = float(sys.argv[sys.argv.index("--temperature") + 1])
    if t_list:
        temperature = [0.1, 0.4, 0.7, 1., 1.5, 2, 4, 7, 10, 50, 100, 1000, 10000]

    if on_data:
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



