#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import numpy as np
import AllPairs as AP
import time


def take_process_time(function):
    def wrapper(*args, **kwars):
        print('\nrunning function %s...' % function.__name__)
        start = time.process_time()
        result = function(*args, **kwars)
        end = time.process_time()
        print('process time: %.2f' % (end - start))
        return result
    return wrapper
    

def compare(result, verified_result):
    print('\nComparing results:')
    print('\nfalse negatives:')
    #print('These are the pairs missed by the normal version:')
    fn = [x for x in verified_result if x not in result]
    print(fn)
    
    print('false positives:')
    #print('These are the pairs found by the normal version but not by all-verify:')
    fp = [x for x in result if x not in verified_result]
    print(fp)
    return fn+fp


def build_subsample_file(data_dict, sample_keys, flag, writemode='a'):
    sample_dict = {k:v for k,v in data_dict.items() if k in sample_keys}
    with open('sample_data.txt', writemode) as outfile:
        for key, value in sample_dict.items():
            tup = ' '.join([str(x) for x in value])
            outfile.write('%s, %s, %s\n' % (key, flag, tup))


def sim(r, s):
    ''' Calculate Jaccard-similarity between two sets of characters.
    @ r, s: two sets of characters (can have datatype list, tupe or set)'''
    intersection = set(r).intersection(set(s))
    union = set(r).union(set(s))
    return len(intersection) / len(union)


def eqo(r, s, t):
    return t/(t+1) * (len(r) + len(s))

def lb(r, t):
    return len(r) * t

def probing_prefix_length(r, t):
    return int(len(r) - np.ceil( lb(r, t) ) + 1)

def indexing_prefix_length(r, t):
    return int(len(r) - np.ceil(eqo(r, r, t)) + 1)


@take_process_time
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


@take_process_time
def AllPairs(Data, threshold=0.7, check_for=[]):
    ''' @ Data: list of tuples to be compared
        return: list of matching tuples'''
    res = []  # result: pairs of similar vectors
    I = {}
    for r, probe in Data.items():
        #logstr = ''
        #logstr += '\n\nr: %s\n' % (probe,)
        M = {}
        for p in probe[0:probing_prefix_length(probe, threshold)]:  # for char in probing prefix
            #logstr += 'p: %s\n' % p
            if p in I.keys():
                for s in I[p]:  # for vector index in inverted list
                    if len(Data[s]) < lb(probe, threshold):  # if other vector shorter than lbr
                        #if s in check_for:
                        #    print('\nremoving s:')
                        #    print('r %s: %s' % (r, probe))
                        #    print('s %s: %s' % (s, Data[s]))       
                        #I[p].remove(s)
                        pass
                    else:
                        if s not in M.keys():
                            M[s] = 0
                        M[s] += 1
        #logstr += 'M: %s\n' % M
        
        for p in probe[0:indexing_prefix_length(probe, threshold)]:  # for char in indexing prefix
            if p not in I.keys():
                I[p] = []
            I[p].append(r)
        #logstr += 'I: %s' % I
        #logstr += 'candidate dict M: %s\n' % M
        for s, overlap in M.items():
            req_overlap = np.ceil(eqo(probe, Data[s], threshold))
            #logstr += 'found matches:\n'
            if AP.verify(probe, Data[s], t=req_overlap, olap=0, p_r=0, p_s=0):
                res.append( (r, s) )  # using tuples to make results hashable
                #logstr += '(%s, %s)' % (r,s)
        
        #logstr += '--------------------------------------------------------\n'
        #with open('logfile_AllPairs.txt', 'w') as outfile:
        #    outfile.write(logstr)
    
    return res


if __name__ == '__main__':
    
    test = AP.read_txt('test.txt')
    spotify = AP.read_txt('spotify-track-dedup-raw.txt')
    #sample = AP.read_txt('sample_data.txt')  # selected tuples for debugging


    res = AllPairs(spotify, threshold=0.7, check_for=['r237015', 'r219243', 'r240857', 'r217618', 'r247549', 'r205952', 'r250199', 'r219198', 'r269027', 'r245701'])
    print('pairs: %s' % len(res))
    
    res_ver = AllPairs_all_verify(spotify, threshold=0.7)
    print('pairs: %s' % len(res_ver))
    
    diff = compare(res, res_ver)
    diff_flat = [x for tup in diff for x in tup]
    
    # make subsample file
    FN = [x for sub in diff for x in sub]
    TP = [x for sub in res[0:5] for x in sub]  # add true positives
    TN_all = set(spotify.keys()).difference(set([x for sub in res_ver for x in sub]))
    TN = np.random.choice(list(TN_all), 10)
    fn = {k:v for k,v in spotify.items() if k in FN}
    tp = {k:v for k,v in spotify.items() if k in TP}
    tn = {k:v for k,v in spotify.items() if k in TN}
    
    r = AllPairs(fn, check_for=['r237015', 'r219243', 'r240857', 'r217618', 'r247549', 'r205952', 'r250199', 'r219198', 'r269027', 'r245701'])
    print('pairs: %s' % len(r))
    
    rver = AllPairs_all_verify(fn)
    print('pairs: %s' % len(rver))

    #build_subsample_file(spotify, sample_keys=fn, flag='fn', writemode='w')
    #build_subsample_file(spotify, sample_keys=tp, flag='tp', writemode='a')
    #build_subsample_file(spotify, sample_keys=tn, flag='tn', writemode='a')
    #with open('sample_data.txt', 'w') as outfile:
    #    for key, tup in subsample.items():
    #        line = ' '.join([str(x) for x in tup])
    #        outfile.write('%s: %s\n' % (key, line))
