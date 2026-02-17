# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 22:10:39 2026

@author: Karuna
"""

# import os
# import time
# import psutil
# start_time = time.time()
from pyfaidx import Fasta
import numpy as np
import pandas as pd
from itertools import product

thefasta = Fasta('fastafile.fa')
def markov(order, myfastafile): 
    '''
    Markov model prediction

    Parameters
    ----------
    order : markov model order
    myfastafile: fasta file given (must be .fa)

    Returns
    -------
    loglik_dict : dictionary containing log-likelihood for each seq

    '''
    combos = [''.join(p) for p in product('ATGC', repeat=order)] #all possible order-mers
    tm_dict = {c: {n: 1 for n in 'ATGC'} for c in combos}  # transition matrix
    for fastakey in myfastafile.keys():        
        nctds = str(myfastafile[fastakey])
        for j in range(len(nctds) - order):
            ordermer = (nctds[j:j+order]).upper()
            if 'N' in ordermer:
                continue
            next_letter = (nctds[j+order]).upper()
            # add to count to the transition matrix 
            tm_dict[ordermer][next_letter] += 1
    for keys,vals in tm_dict.items(): #normalising the transition matrices
            #keys: prev nucleotide order-mer
            #vals: next nucleotide 
        count_sum = sum(vals.values())
        for nuc in vals:
            vals[nuc] = np.log(vals[nuc]/count_sum)
        # TESTING:
    
    for fastakey in myfastafile.keys():        
        nctds = str(myfastafile[fastakey])
        prob_log = 0
        for j in range(len(nctds)- order):
                if 'N' in ordermer:
                    continue
                ordermer_test =  (nctds[j:j+order]).upper()
                next_letter_test = (nctds[j+order]).upper()
                prob_log += tm_dict[ordermer_test][next_letter_test] 
        print(prob_log)
    return prob_log

order_inp = int(input('Input markov order (0-10): '))
prob_log = markov(order_inp, thefasta)