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
            result['r' + str(i)] = row
            i += 1

    return result


def read_txt_list(filename):
    """
    reads in txt file from working directory
    :param filename: string *.txt with sorted rows of data. each row is sorted and all rows are sorted in ascending order
    :return: dictionary with key = ID and value = integer list
    """
    data_list = []
    with open(filename) as infile:
        for line in infile.readlines():
            data_list.append([int(x) for x in line.split()])
    return data_list


def verify(r, s, t, olap, p_r, p_s):
    """
    :param r: tuple1
    :param s: tuple2
    :param t: float required overlap
    :param overlap till positions p_r and p_s
    :param p_r: prefix position
    :param p_s: prefix position
    :return: True if the overlap is sufficient for jaccard threshold
    """
    overlap = olap
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
    # value: list of tuple-indices ('r1', 'r33',...) which contain the charater

    for r, probe in enumerate(Data):

        # calculate metrics:
        probing_prefix_len = probing_prefix_length(probe, threshold)
        indexing_prefix_len = indexing_prefix_length(probe, threshold)
        lb_r = lb(probe, threshold)

        M = {}
        for p in probe[0:probing_prefix_len]:  # for char in probing prefix
            if p in I.keys():
                for s in I[p][:]:  # for 'r122...' in index in inverted list
                    if len(Data[s]) < lb_r:  # if other vector is shorter than lb_r
                        I[p].remove(s)
                    else:
                        if s not in M.keys():
                            M[s] = 0
                        M[s] += 1

        for p in probe[0:indexing_prefix_len]:  # for char in indexing prefix
            if I.get(p) is None:
                I[p] = []
            I[p].append(r)
        for s, overlap in M.items():
            current_candidate = Data[s]
            req_overlap = np.ceil(eqo(probe, current_candidate, jaccard_threshold))
            indexing_prefix_len_s = indexing_prefix_length(current_candidate, threshold)
            probing_prefix_position_r = min(probing_prefix_len, len(probe) - 1)
            indexing_prefix_position_s = min(indexing_prefix_len_s, len(current_candidate) - 1)
            w_r = probe[probing_prefix_position_r]
            w_s = current_candidate[indexing_prefix_position_s]
            if w_r < w_s:
                ret = verify(probe, current_candidate, t=req_overlap, olap=M[s], p_r=probing_prefix_len, p_s=M[s])
            else:
                ret = verify(probe, current_candidate, t=req_overlap, olap=M[s], p_r=M[s], p_s=indexing_prefix_len_s)
            if ret:
                res.append((r, s))
    return I, res


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Returns an output size and a real CPU time', epilog='Done',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('filename', help="*.txt file in working directory",
                        type=str)
    parser.add_argument('jaccard_threshold', help="threshold for calculation",
                        type=float)
    args = parser.parse_args()

    start = time.time()

    jaccard_threshold = args.jaccard_threshold
    data = read_txt_list(args.filename)

    I, pairs = AllPairs(data, threshold=jaccard_threshold)

    end = time.time()

    print(len(pairs))
    print(round(end - start, 2))
