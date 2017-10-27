#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 18:13:38 2017

@author: raph
"""

import numpy as np

R = [[1,3,5],  # R[0] ... r1
     [1,2,3,4],  # R[1] ... r2
     [1,2,4,9,11],  # r3
     [1,3,5,8,9,10,11,12,13,14]  # r4
     ]


def sim(r, s):
    ''' Calculate Jaccard-similarity between two sets of characters.
    @ r, s: two sets of characters (can have datatype list, tupe or set)'''
    intersection = set(r).intersection(set(s))
    union = set(r).union(set(s))
    return len(intersection) / len(union)


def Verify(r, M, t=0.5):
    verified_pairs = []
    for candidate in M.keys():
        intersect = set(R[r]).intersection(R[candidate])
        if len(intersect) >= eqo(R[r], R[candidate]):
            verified_pairs.append( (r, candidate) )
    return verified_pairs


def eqo(r, s, t=0.5):
    return t/(t+1) * (len(r) + len(s))


def lb(r, t=0.5):
    return len(r) * t


def probing_prefix_length(r):
    return int(len(r) - np.ceil( lb(r) ) + 1)


def indexing_prefix_length(r):
    return int(len(r) - np.ceil(eqo(r,r)) + 1)


res = []  # result: pairs of similar vectors
I = {}
for r in range(len(R)):
    print('r: %s' % R[r])
    M = {}
    for p in R[r][0:probing_prefix_length(R[r])]:  # for char in probing prefix
        print('p: %s' % p)
        if p in I.keys():
            for s in I[p]:  # for vector index in inverted list
                if len(R[s]) < lb(R[r]):  # if other vector shorter than lbr
                    I[p].remove(s)
                else:
                    if s not in M.keys():
                        M[s] = 0
                    M[s] += 1
    print('M: %s' % M)
    for p in R[r][0:indexing_prefix_length(R[r])]:  # for char in indexing prefix
        if p not in I.keys():
            I[p] = []
        I[p].append(r)
    print('I: %s' % I)
    res += Verify(r, M)
    print('res: %s' % res)
    