#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import argparse
import numpy as np
import time


def read_txt(filename):
    """
    reads in txt file from working directory
    :param filename: string *.txt with sorted rows of data. each row is sorted and all rows are sorted in ascending order
    :return: dictionary with key = ID and value = integer list
    """
    result = {}
    i = 1

    with open(filename) as input_file:
        for row in input_file:
            row = tuple(map(int, row.split()))
            result[str(i)] = row
            i += 1

    return result


def verify(r, s, t, overlap, p_r, p_s):
    """
    :param r: tuple1
    :param s: tuple2
    :param t: float required overlap
    :param overlap: overlap till positions p_r and p_s
    :param p_r: prefix position
    :param p_s: prefix position
    :return: True if the overlap is sufficient for jaccard threshold
    """

    max_r = len(r) - p_r + overlap
    max_s = len(s) - p_s + overlap
    while overlap < t <= min(max_r, max_s):
        if r[p_r] == s[p_s]:
            p_r = p_r + 1
            p_s = p_s + 1
            overlap += 1
        elif r[p_r] < s[p_s]:
            p_r = p_r + 1
            max_r = max_r - 1
        else:
            p_s = p_s + 1
            max_s = max_s - 1

    return True if overlap >= t else False


def metrics(collection, t):
    """
    :param collection: dictionary with key: ID and value = list like output of read_txt
    :return: metrics collection for all-pair alg
    """
    result = {}
    for i in collection.keys():
        result[i] = {'length': len(collection[i]),
                     'eqo': eqo(collection[i], collection[i], t),
                     'lb': lb(collection[i], t),
                     'prob_prefix': probing_prefix_length(collection[i], t),
                     'ind_prefix': indexing_prefix_length(collection[i], t)}
    return result


def sim(r, s):
    ''' Calculate Jaccard-similarity between two sets of characters.
    @ r, s: two sets of characters (can have datatype list, tupe or set)'''
    intersection = set(r).intersection(set(s))
    union = set(r).union(set(s))
    return len(intersection) / float(len(union))


def eqo(r, s, t):
    return t / (t + 1) * (len(r) + len(s))


def lb(r, t):
    return len(r) * t


def probing_prefix_length(r, t):
    return int(len(r) - np.ceil(lb(r, t)) + 1)


def indexing_prefix_length(r, t):
    return int(len(r) - np.ceil(eqo(r, r, t)) + 1)


def AllPairs(Data, threshold=0.7):
    ''' @ Data: dict with key: index ('r1, r2, ...) and value: tuples of
                integers for similarity search
        return: list of matching tuples'''
    res = []  # result: collect pairs of similar tuples
    I = {}  # inverted list: key: character
    key_list = list(Data.keys())
    # np.random.shuffle(key_list)
    key_list = sorted(key_list, key=lambda x: len(Data[x]))
    metr = metrics(Data, threshold)

    start = time.process_time()
    for r in key_list:
        probe = Data[r]

        # fetch metrics:
        probing_prefix_len = metr[r]['prob_prefix']
        indexing_prefix_len = metr[r]['ind_prefix']
        lb_r = metr[r]['lb']

        M = {}
        for p in probe[0:probing_prefix_len]:  # for char in probing prefix
            if I.get(p) is not None:
                for s in I[p][:]:  # for 'r122...' in index in inverted list
                    if len(Data[s]) < lb_r:  # if other vector is shorter than lb_r
                        I[p].remove(s)
                    else:
                        if s not in M.keys():
                            M[s] = 0
                        M[s] += 1

        for p in probe[0:indexing_prefix_len]:  # for char in indexing prefix
            if p not in I.keys():
                I[p] = []
            I[p].append(r)
        for s, overlap in M.items():
            candidate = Data[s]
            candidate_ind_prefix = metr[s]['ind_prefix']
            pre_overlap = M[s]
            req_overlap = np.ceil(eqo(probe, candidate, jaccard_threshold))
            probing_prefix_position_r = min(probing_prefix_len, len(probe) - 1)
            indexing_prefix_position_s = min(candidate_ind_prefix, len(candidate) - 1)
            w_r = probe[probing_prefix_position_r]
            w_s = Data[s][indexing_prefix_position_s]
            if w_r < w_s:
                ret = verify(probe, candidate, t=req_overlap, overlap=pre_overlap, p_r=probing_prefix_len, p_s=pre_overlap)
            else:
                ret = verify(probe, candidate, t=req_overlap, overlap=pre_overlap, p_r=pre_overlap, p_s=candidate_ind_prefix)
            if ret:
                res.append((r, s))

    duration = time.process_time() - start
    return res, duration


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Returns an output size and a real CPU time', epilog='Done',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('filename', help="*.txt file in working directory",
                        type=str)
    parser.add_argument('jaccard_threshold', help="threshold for calculation",
                        type=float)
    args = parser.parse_args()

    jaccard_threshold = args.jaccard_threshold
    data = read_txt(args.filename)

    pairs = AllPairs(data, threshold=jaccard_threshold)

    print(len(pairs[0]))
    print(round(pairs[1], 2))
