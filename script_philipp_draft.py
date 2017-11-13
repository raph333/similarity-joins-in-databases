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


test_input = read_txt('test.txt')


def eqo(r, s, t=0.5):
    return t / (t + 1) * (len(r) + len(s))


def lb(r, t=0.5):
    return len(r) * t


def probing_prefix_length(r):
    return int(len(r) - np.ceil(lb(r)) + 1)


def indexing_prefix_length(r):
    return int(len(r) - np.ceil(eqo(r, r)) + 1)


def verify(r, s, t, olap, p_r, p_s):
    """
    :param r: tuple1
    :param s: tuple2
    :param t: float
    :param overlap: integer
    :param p_r: prefix position
    :param p_s: prefix position
    :return: tuple of tuples if the overlap to specified indices exceeds a given threshold
    """
    overlap = olap
    max_r = len(r) - p_r + overlap
    max_s = len(s) - p_s + overlap
    print('max_r:{}'.format(max_r))
    print('max_s:{}'.format(max_s))
    print('required_overlap:{}'.format(t))
    while overlap < t <= min(max_r, max_s):
        print('p_r:{}'.format(p_r))
        print('p_s:{}'.format(p_s))
        print('r:{}'.format(r))
        print('s:{}'.format(s))
        print('max_r:{}'.format(max_r))
        print('max_s:{}'.format(max_s))
        print('r[p_r]:{}'.format(r[p_r]))
        print('s[p_s]:{}'.format(s[p_s]))
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
            print('updated p_s{0} and max_s{1}'.format(p_s, max_s))

    return True if overlap >= t else False


def metrics(collection):
    """
    :param collection: dictionary with key: ID and value = list like output of read_txt
    :return: metrics collection for all-pair alg
    """
    result = {}
    for i in collection.keys():
        result[i] = {'length': len(collection[i]),
                     'eqo': eqo(collection[i], collection[i]),
                     'lb': lb(collection[i]),
                     'prob_prefix': probing_prefix_length(collection[i]),
                     'ind_prefix': indexing_prefix_length(collection[i])}

    return result


metrics_test = metrics(test_input)

if __name__ == '__main__':

    # parser = argparse.ArgumentParser(description='Outputs an html file containing a label report', epilog='Done',
    #                                  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('--remove_redundant', help="remove all redundant documents from dataset",
    #                     choices=["y", "n"], default="y")
    # parser.add_argument('--doc_ids', help="list of document ids you want to investigate. default: all ids",
    #                     default=all_ids, type=str)
    # parser.add_argument('--htmlpath', help="path to the output html file. default: current working dir",
    #                     default=str(getcwd()))
    #
    # args = parser.parse_args()

    res = []  # result: pairs of similar vectors
    I = {}
    for r in test_input.keys():
        probe = test_input[r]
        print('r: ' + str(probe))
        M = {}
        for p in probe[0:metrics_test[r]['prob_prefix']]:  # for char in probing prefix
            print('p: %s' % p)
            if p in I.keys():
                for s in I[p]:  # for vector index in inverted list
                    if len(test_input[s]) < metrics_test[r]['lb']:  # if other vector shorter than lbr
                        I[p].remove(s)
                    else:
                        if s not in M.keys():
                            M[s] = 0
                        M[s] += 1
        print('M: %s' % M)
        for p in probe[0:metrics_test[r]['ind_prefix']]:  # for char in indexing prefix
            if p not in I.keys():
                I[p] = []
            I[p].append(r)
        print('I: %s' % I)

        for s, overlap in M.items():
            print('probe: ' + str(probe))
            print('s: ' + str(test_input[s]))
            required_overlap = int(np.ceil(eqo(probe, test_input[s])))
            print('required olap: ' + str(required_overlap))
            print('overlap:{}'.format(M[s]))
            if verify(probe, test_input[s], t=required_overlap, olap=M[s], p_r=metrics_test[r]['prob_prefix']-1,
                      p_s=metrics_test[s]['prob_prefix']):
                res.append([r, s])
        print('res: %s' % res)





