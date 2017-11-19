#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 18:13:38 2017

@author: raph
"""

import numpy as np
import AllPairs as AP

# =============================================================================
# R = [[1,3,5],  # R[0] ... r1
#      [1,2,3,4],  # R[1] ... r2
#      [1,2,4,9,11],  # r3
#      [1,3,5,8,9,10,11,12,13,14]  # r4
#      ]
# =============================================================================

test = AP.read_txt('test.txt')
spotify = AP.read_txt('spotify-track-dedup-raw.txt')


def sim(r, s):
    ''' Calculate Jaccard-similarity between two sets of characters.
    @ r, s: two sets of characters (can have datatype list, tupe or set)'''
    intersection = set(r).intersection(set(s))
    union = set(r).union(set(s))
    return len(intersection) / len(union)


def Verify(Data, k, M, t):
    r = Data[k]
    verified_pairs = []
    for key in M.keys():
        intersect = set(r).intersection(set(Data[key]))
        if len(intersect) >= eqo(r, Data[key], t):
            verified_pairs.append( (k, key) )
    return verified_pairs


def eqo(r, s, t):
    return t/(t+1) * (len(r) + len(s))


def lb(r, t):
    return len(r) * t


def probing_prefix_length(r, t):
    return int(len(r) - np.ceil( lb(r, t) ) + 1)


def indexing_prefix_length(r, t):
    return int(len(r) - np.ceil(eqo(r, r, t)) + 1)


# =============================================================================
# def AllPairs(Data, thres=0.7, all_verify=False):
#     ''' @ Data: list of tuples to be compared
#         return: list of matching tuples'''
#     res = []  # result: pairs of similar tuples
#     I = {}   # inverted list
#     for r, probe in Data.items():
#         M = {}
#         #print('\nr: %s' % r)
#         for p in probe[0:probing_prefix_length(probe, thres)]:  # for char in probing prefix
#             #print('p: %s' % p)
#             if p in I.keys():
#                 for s in I[p]:                    
#                     if len(Data[s]) < lb(r, thres):  # if other tuple is shorter than lbr
#                         I[p].remove(s)
#                     else:
#                         if s not in M.keys():
#                             M[s] = 0
#                         M[s] += 1
# #            except:
# #                pass
#             
#         #print('M: %s' % M)
#         for p in probe[0:indexing_prefix_length(probe, thres)]:  # for char in indexing prefix
#             if p not in I.keys():
#                 I[p] = []
#             I[p].append(r)
#         #print('I: %s' % I)
# 
#         for s, overlap in M.items():
#             req_overlap = np.ceil(eqo(probe, Data[s], thres))
#             if AP.verify(probe, Data[s], t=req_overlap, olap=0, p_r=0, p_s=0):
#                 res.append( (r, s) )  # using tuples to make results hashable
#     return res
# 
# #def all_verify(Data, thres)
# 
# =============================================================================
    
def AllPairs_all_verify(Data, threshold=0.7):
    ''' @ Data: list of tuples to be compared
        return: list of matching tuples'''
    res = []  # result: pairs of similar vectors
    I = {}
    logstr = ''
    for r, probe in Data.items():
        #logstr += 'r: %s' % (probe,)
        M = {}
        for p in probe[0:probing_prefix_length(probe, threshold)]:  # for char in probing prefix
            logstr += 'p: %s' % p
            if p in I.keys():
                for s in I[p]:  # for vector index in inverted list
                    if s not in M.keys():
                        M[s] = 0
                    M[s] += 1
        #logstr += 'M: %s' % M
        
        for p in probe[0:indexing_prefix_length(probe, threshold)]:  # for char in indexing prefix
            if p not in I.keys():
                I[p] = []
            I[p].append(r)
        #logstr += 'I: %s' % I
        #logstr += 'candidate dict M: %s' % M
        #res += Verify(R, r, M, threshold)
        for s, overlap in M.items():
            req_overlap = np.ceil(eqo(probe, Data[s], threshold))
            if AP.verify(probe, Data[s], t=req_overlap, olap=0, p_r=0, p_s=0):
                res.append( (r, s) )  # using tuples to make results hashable
    return res

def AllPairs(Data, threshold=0.7):
    ''' @ Data: list of tuples to be compared
        return: list of matching tuples'''
    res = []  # result: pairs of similar vectors
    I = {}
    logstr = ''
    for r, probe in Data.items():
        #logstr += 'r: %s' % (probe,)
        M = {}
        for p in probe[0:probing_prefix_length(probe, threshold)]:  # for char in probing prefix
            logstr += 'p: %s' % p
            if p in I.keys():
                for s in I[p]:  # for vector index in inverted list                    
                    if len(Data[s]) < lb(probe, threshold):  # if other vector shorter than lbr
                        I[p].remove(s)
                    else:
                        if s not in M.keys():
                            M[s] = 0
                        M[s] += 1
        #logstr += 'M: %s' % M
        
        for p in probe[0:indexing_prefix_length(probe, threshold)]:  # for char in indexing prefix
            if p not in I.keys():
                I[p] = []
            I[p].append(r)
        #logstr += 'I: %s' % I
        #logstr += 'candidate dict M: %s' % M
        #res += Verify(R, r, M, threshold)
        for s, overlap in M.items():
            req_overlap = np.ceil(eqo(probe, Data[s], threshold))
            if AP.verify(probe, Data[s], t=req_overlap, olap=0, p_r=0, p_s=0):
                res.append( (r, s) )  # using tuples to make results hashable
    return res

def compare(result, verified_result):
    print('\nFalse negatives:')
    #print('These are the pairs missed by the normal version:')
    fn = [x for x in verified_result if x not in result]
    print(fn)
    
    print('False positives:')
    #print('These are the pairs found by the normal version but not by all-verify:')
    fp = [x for x in result if x not in verified_result]
    print(fp)
    return fn+fp
    

print('normal version:')
res = AllPairs(spotify)
print(len(res))

print('all verify version:')
res_ver = AllPairs_all_verify(spotify)
print(len(res_ver))

diff = compare(res, res_ver)

    