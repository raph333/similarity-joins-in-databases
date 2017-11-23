#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 19:04:57 2017

@author: raph
"""

import numpy as np
import time

def take_process_time(function):
    def wrapper(*args, **kwars):
        print('\nrunning function %s...' % function.__name__)
        start = time.process_time()
        result = function(*args, **kwars)
        end = time.process_time()
        execution_time = round(end - start, 2)
        print('process time: %.2f' % execution_time)
        return result, execution_time
    return wrapper

l = list(np.random.uniform(low=0.0, high=1.0, size=10000000))
s = set(l)

probes = np.random.choice(l, 100)  # remove these elements from list/set


@take_process_time
def remove_from_list(input_list,elements_to_remove):
    for element in elements_to_remove:
        input_list.remove(element)
    return None


@take_process_time
def remove_from_set(input_set,elements_to_remove):
    for element in elements_to_remove:
        input_set.remove(element)
    return None


remove_from_list(l, probes)
remove_from_set(s, probes)