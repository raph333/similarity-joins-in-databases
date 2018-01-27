#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""

import argparse
import numpy as np
import time


def read_txt_list(filename):
    """
    reads in txt file from working directory
    :param filename: string *.txt with sorted rows of data. each row is
                     sorted and all rows are sorted in ascending order
    :return: dictionary with key = ID and value = integer list
    """
    data_list = []
    with open(filename) as infile:
        for line in infile.readlines():
            data_list.append([int(x) for x in line.split()])
    return data_list


def read_weights(filename):
    """
    @ filename: file with token-weights
    return: list with weight of a token n at position n
            e.g. token 42: get the weight of this token with: weight_array[42]
    """
    weight_array = [0]  # start with weight of token 1 at position 1
    with open(filename) as infile:
        for line in infile.readlines():
            row = line.strip().split(':')
            weight = float(row[1])
            weight_array.append(weight)
    return weight_array


# =============================================================================
# def jaccard(r, s):
#     r = set(r)
#     s = set(s)
#     return len(r.intersection(s)) / len(r.union(s))
# =============================================================================


def weighted_jaccard(r, s, weights):
    """
    For now: get weights bei dicitonary lookupt
    Maybe it's better to give a list of weights as it is used anyway in the
    AllPairs function.
    """
    r = set(r)
    s = set(s)
    intersect_weight = sum([weights[token] for token in r.intersection(s)])
    union_weight = sum([weights[token] for token in r.union(s)])
    #diff_weight = sum([token2weight[x] for x in r.symmetric_difference(s)])
    #union_weight = intersect_weight + diff_weight
    weighted_jaccard_similarity = intersect_weight / union_weight
    return weighted_jaccard_similarity


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


# FUNCTIONS FOR OLD ALGORITHM
# =============================================================================
def lb(r, t):
    return len(r) * t


def eqo(r, s, t):
    return t / (t + 1) * (len(r) + len(s))


def probing_prefix_length(r, t):
    return int(len(r) - np.ceil(lb(r, t)) + 1)


# for now: still used for weighted Alg; does prefix length matter :-) ?
def indexing_prefix_length(r, t):
    return int(len(r) - np.ceil(eqo(r, r, t)) + 1)
# =============================================================================


def weight_lb(total_set_weight, t):
    """
    Let r be a set with @total_set_weight.
    @t: Jaccard similarity threshold
    Any other set s needs at least a total set weight of 'lower_bound' to be
    (potentially) similar to r.
    """
    lower_bound = total_set_weight * t
    return lower_bound


def weighted_probing_prefix_length(weight_left, t):
    """
    @ total_set_weight: sum of the weights of all tokens in a set r
    @ t: Jaccard similarity threshold
    If the Alg. has checked this prefix of a set r and there's still no
    overlap with another set s, the the similarity between r and s cannot
    be larger or equal to threshold t.
    Hence, only the prefix of any set r hast to be checked.
    return: length of the probing prefix of set r
    """
    lower_bound = weight_lb(weight_left[0], t)
    index = 0  # check any set r until (including) the position 'index'
    while index < len(weight_left) and weight_left[index] >= lower_bound:
        index += 1
    probing_prefix_length = index + 1
    return probing_prefix_length


# NEW INDEXING PREFIX: under construction but not yet used
# =============================================================================
def weight_eqo(r_total_weight, s_total_weight, t):
    return t / (t + 1) * (r_total_weight + s_total_weight)


def weight_indexing_prefix_length(r_weight, t):
    ipl = int(r_weight - np.ceil(weight_eqo(r_weight, r_weight, t)) + 1)
    return ipl
# =============================================================================
        

# Die Funktion ist neu und berechent mir (vermutlich infeffizient...:-) )
# Wie viel Gewicht nach Position x noch übrig ist
def weight_remaining_after_position(weights):
    """
    @ weights: list of token weights
    return: list: the value at each position indicates the sum of the the
            weights of all positions after (and including) the current position
    """
    weight_left = [0] * len(weights)   # initialize 'empty' list
    weight_left[-1] = weights[-1]  # set last element
    
    # Accumulate weights starting at the end of the list:
    for i in range(len(weight_left)-2, -1, -1):
        weight_left[i] = weight_left[i+1] + weights[i]
    
    return weight_left


def AllPairs(Data, weights, t=0.7):
    """
    @ Data: dict with key: index (r1, r2, ...) and value: tuples of
            integers for similarity search
    return: list of matching tuples
    """
    res = []  # result: collect pairs of similar tuples
    I = {}  # inverted list (implemented as dictionary)
    
    for r, probe in enumerate(Data):
        M = {}  # set of potential pairs
        
        r_weights = [weights[token] for token in probe]
        r_weight_left = weight_remaining_after_position(r_weights)
        r_weights_sum = r_weight_left[0]  # sum of weights of all tokens in r
        
        for p in probe[0 : weighted_probing_prefix_length(r_weight_left, t)]:
            if p in I.keys():
                for i in range(len(I[p])-1, -1, -1):
                    s = I[p][i]
                    
                    s_weights_sum = sum([weights[token] for token in Data[s]])
                    if s_weights_sum <= weight_lb(r_weights_sum, t):
                        del I[p][i]
                    
                    else:
                        if M.get(s) is None:
                            M[s] = 0
                        M[s] += 1

        for p in probe[0 : indexing_prefix_length(probe, t)]:
            if I.get(p) is None:
                I[p] = []
            I[p].append(r)

        for s, overlap in M.items():
# =============================================================================
#             current_candidate = Data[s]
#             req_overlap = np.ceil(eqo(probe, current_candidate,
#                                       jaccard_threshold))
#             indexing_prefix_len_s = indexing_prefix_length(current_candidate,
#                                                            t)
#             probing_prefix_position_r = min(probing_prefix_len, len(probe) - 1)
#             indexing_prefix_position_s = min(indexing_prefix_len_s,
#                                              len(current_candidate) - 1)
#             w_r = probe[probing_prefix_position_r]
#             w_s = current_candidate[indexing_prefix_position_s]
#             if w_r < w_s:
#                 ret = verify(probe, current_candidate, t=req_overlap,
#                              olap=M[s], p_r=probing_prefix_len, p_s=M[s])
#             else:
#                 ret = verify(probe, current_candidate, t=req_overlap,
#                              olap=M[s], p_r=M[s], p_s=indexing_prefix_len_s)
#             if ret:
#                 res.append((r, s))
# =============================================================================
            if weighted_jaccard(probe, Data[s], weights) >= t:
                res.append((r, s))
    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description='Returns an output size and a real CPU time',
            epilog='Done',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
    parser.add_argument('filename', help="*.txt file in working directory",
                        type=str)
    parser.add_argument('weights_file', help="file with token weights",
                        type=str)
    parser.add_argument('jaccard_threshold', help="threshold for calculation",
                        type=float)
    args = parser.parse_args()

    jaccard_threshold = args.jaccard_threshold
    data = read_txt_list(args.filename)
    weight_dict = read_weights(args.weights_file)
    
    start = time.process_time()
    pairs = AllPairs(data, weight_dict, t=jaccard_threshold)
    end = time.process_time()

    print(len(pairs))
    print(round(end - start, 2))
