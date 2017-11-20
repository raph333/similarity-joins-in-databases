#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import argparse
import numpy as np
import time


def take_process_time(function):
    def wrapper(*args, **kwars):
        # print('\nrunning function %s...' % function.__name__)
        start = time.process_time()
        result = function(*args, **kwars)
        end = time.process_time()
        execution_time = end - start
        # print('process time: %.2f' % execution_time)
        return result, execution_time

    return wrapper


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


len_diff = []  # track how many elements are removed from inverted list I


@take_process_time
def AllPairs(Data, threshold=0.7):
    ''' @ Data: list of tuples to be compared
        return: list of matching tuples'''
    res = []  # result: pairs of similar tuples
    I = {}

    key_list = list(data.keys())
    # np.random.shuffle(key_list)
    key_list = sorted(key_list, key=lambda x: len(Data[x]))
    for r in key_list:
        probe = Data[r]

        # calculate metrics:
        probing_prefix_len = probing_prefix_length(probe, threshold)
        indexing_prefix_len = indexing_prefix_length(probe, threshold)
        lb_r = lb(probe, threshold)

        M = {}
        after_common_token_position = {}
        verify_starting_position = {}
        for p in probe[0:probing_prefix_len]:  # for char in probing prefix
            if p in I.keys():
                for s in I[p]:  # for vector index in inverted list
                    if len(Data[s]) < lb_r:  # if other vector shorter than lbr
                        # before_len = len(I[p])
                        I[p] = [x for x in I[p] if x != s]  # I[p].remove(s)
                        # len_diff.append(before_len - len(I[p]))
                        # pass
                    else:
                        if s not in M.keys():
                            M[s] = 0
                        M[s] += 1
                        after_common_token_position[s] = Data[s].index(p) + 1
                        verify_starting_position[r] = indexing_prefix_len
            else:
                # print(type(list(I.keys())))
                if len(I.keys()) != 0:
                    if p > list(I.keys())[-1]:
                        verify_starting_position[r] = probe.index(p)
                # else:
                #     verify_starting_position[r] = indexing_prefix_len
                # print(after_common_token_position)

        # print(last_common_token)
        for p in probe[0:indexing_prefix_len]:  # for char in indexing prefix
            if p not in I.keys():
                I[p] = []
            I[p].append(r)
        for s, overlap in M.items():
            req_overlap = np.ceil(eqo(probe, Data[s], threshold))
            # s_starting_position = Data[s].index[last_common_token] + 1
            if verify(probe, Data[s], t=req_overlap, olap=M[s], p_r=indexing_prefix_len, p_s=after_common_token_position[s]):
                res.append((r, s))  # using tuples to make results hashable
    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Returns an output size and a real CPU time', epilog='Done',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('filename', help="*.txt file in working directory",
                        type=str)
    parser.add_argument('jaccard_threshold', help="threshold for calculation",
                        type=float)
    args = parser.parse_args()

    start = time.process_time()

    jaccard_threshold = args.jaccard_threshold
    data = read_txt(args.filename)

    res, exec_time = AllPairs(data, threshold=jaccard_threshold)

    end = time.process_time()

    print(len(res))
    print(round(end - start, 2))
