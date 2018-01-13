#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import collections
import AllPairs as ap


def get_weights(data):
    flat_token_list = [x for sublist in data for x in sublist]
    multiset = collections.Counter(flat_token_list)
    token2weight = {token:1/multiset[token] for token in multiset.keys()}
    #token2weight = {k:v/len(flat_token_list) for k,v in token2count.items()}
    return token2weight


def write_weight_file(weight_dict, outfile_name='weights.txt'):
    with open(outfile_name, 'w') as outfile:
        for token, weight in weight_dict.items():
            outfile.write('%s:%s\n' % (token, weight))
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description='Returns an output size and a real CPU time',
            epilog='Done',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
    parser.add_argument('filename', help="*.txt file in working directory",
                        type=str)
    args = parser.parse_args()
    
    data = ap.read_txt_list(args.filename)
    weights = get_weights(data)
    write_weight_file(weights)

        
    
    
