#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import argparse

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
            # result.append(row)
            result['r' + str(i)] = row
            i += 1

    return result


def eqo(r, s, t):
    return t / (t + 1) * (len(r) + len(s))


def lb(r, t):
    return len(r) * t


def probing_prefix_length(r, t):
    return int(len(r) - np.ceil(lb(r, t)) + 1)


def indexing_prefix_length(r, t):
    return int(len(r) - np.ceil(eqo(r, r, t)) + 1)


def verify(r, s, t, overlap, p_r, p_s):
    """
    :param r: tuple1
    :param s: tuple2
    :param t: float
    :param overlap: integer
    :param p_r: prefix position
    :param p_s: prefix position
    :return: tuple of tuples if the overlap to specified indices exceeds a given threshold
    """
    max_r = len(r) - p_r - 1 + overlap
    max_s = len(s) - p_s - 1 + overlap
    while max_r >= t & max_s >= t & overlap < t:
        if r[p_r-1] == s[p_s-1]:
            p_r = p_r + 1
            p_s = p_s + 1
            overlap += 1
        elif r[p_r-1] < s[p_s-1]:
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


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Returns an output size and a real CPU time', epilog='Done',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('filename', help="*.txt file in working directory",
                        type=str)
    parser.add_argument('jaccard_threshold', help="threshold for calcuation",
                        type=float)

    args = parser.parse_args()
    
    jaccard_threshold = args.jaccard_threshold
    data = read_txt(args.filename)
    
    metrics = metrics(data, jaccard_threshold)

    res = []  # result: pairs of similar vectors
    I = {}
    for r in data.keys():
        probe = data[r]
        print('r: ' + str(probe))
        M = {}
        for p in probe[0:metrics[r]['prob_prefix']]:  # for char in probing prefix
            print('p: %s' % p)
            if p in I.keys():
                for s in I[p]:  # for vector index in inverted list
                    if len(data[s]) < metrics[r]['lb']:  # if other vector shorter than lbr
                        I[p].remove(s)
                    else:
                        if s not in M.keys():
                            M[s] = 0
                        M[s] += 1
        print('M: %s' % M)
        for p in probe[0:metrics[r]['ind_prefix']]:  # for char in indexing prefix
            if p not in I.keys():
                I[p] = []
            I[p].append(r)
        print('I: %s' % I)

        for s, overlap in M.items():
            required_overlap = int(np.ceil(eqo(probe, data[s], jaccard_threshold)))
            if verify(probe, data[s], t=required_overlap, overlap=M[s], p_r=metrics[r]['prob_prefix'],
                      p_s=metrics[s]['prob_prefix']):
                res.append([r, s])
        print('res: %s' % res)

