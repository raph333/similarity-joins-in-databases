#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import csv

R = [(1,3,5),  # R[0] ... r1
     (1,2,3,4),  # R[1] ... r2
     (1,2,4,9,11),  # r3
     (1,3,5,8,9,10,11,12,13,14)  # r4
     ]



def read_txt(filename):
    """
    reads in txt file from working directory
    :param filename: string *.txt
    :return: list of lists where each line from the txt file represents an integer list
    """
    result = []

    with open(filename) as input_file:
        for row in csv.reader(input_file):
            row = list(map(int, row))
            result.append(row)

    return result


def sim(r, s):
    ''' Calculate Jaccard-similarity between two sets of characters.
    @ r, s: two sets of characters (can have datatype list, tupe or set)'''
    intersection = set(r).intersection(set(s))
    union = set(r).union(set(s))
    return len(intersection) / len(union)


def eqo(r, s, t=0.5):
    return int( np.ceil( t/(t+1) * (len(r) + len(s)) ) )


def lb(r, t=0.5):
    return len(r) * t


def probing_prefix_length(r):
    return int(len(r) - np.ceil( lb(r) ) + 1)


def indexing_prefix_length(r):
    return int(len(r) - np.ceil(eqo(r,r)) + 1)


def verify1(r, s, threshold, overlap, p_r, p_s):
    """
    :param r: tuple1
    :param s: tuple2
    :param threshold: float
    :param overlap: integer
    :param p_r: integer index for tuple r
    :param p_s: integer index for tupls s
    :return: tuple of tuples if the overlap to specified indices exceeds a given threshold
    """
    #todo muss noch mit den richtigen Argumenten in den Alg eingebaut werden.
    t = eqo(r, s, threshold)
    max_r, max_s = len(r) - p_r + overlap, len(s) - p_s + overlap
    while max_r >= t & max_s >= t & overlap < t:
        if r[p_r] == s[p_r]:
            p_r, p_s, overlap = p_r + 1, p_s + 1, overlap + 1
        elif r[p_r] < s[p_r]:
            p_r, max_r = p_r + 1, max_r - 1
        else:
            p_s, max_s = p_s + 1, max_s - 1

    return r, s


def Verify(r, M, t):
    results = []
    print(r)
    for s, overlap in M.items():
        print(r)
        print(s)
        print(overlap, type(overlap))
        print(probing_prefix_length(r), type(probing_prefix_length(r)))
        print(len(s)-1)
        res =verify1(r, s, t, overlap, probing_prefix_length(r), len(s)-1)
        results.append(res)
    return results



if __name__ == '__main__':
    
    # test functions:
    R2 = read_txt('test.txt')
    verify1(R[2], R[1], 0.5, overlap=3, p_r=2, p_s=3)
    
    # problem:
    verify1((1, 2, 4, 9, 11), (20,20), 0.9, overlap=0, p_r=2, p_s=3)
    # no 90% overlap: no pair should be returned
    
    # ALL PAIRS ALGORITHM
    res = []  # result: pairs of similar vectors
    I = {}
    for r in R:
        print('r: %s' % (r,) )
        M = {}
        for p in r[0:probing_prefix_length(r)]:  # for char in probing prefix
            print('p: %s' % p)
            if p in I.keys():
                for s in I[p]:  # for vector index in inverted list
                    print('length(s): %s' % len(s))
                    if len(s) < lb(r):  # if other vector shorter than lbr
                        I[p].remove(s)
                    else:
                        if s not in M.keys():
                            M[s] = 0
                        M[s] += 1
        print('M: %s' % M)
        for p in r[0:indexing_prefix_length(r)]:  # for char in indexing prefix
            if p not in I.keys():
                I[p] = []
            I[p].append(r)
        print('I: %s' % I)
        res += Verify(r, M, 0.5)
        print('res: %s' % res)
    