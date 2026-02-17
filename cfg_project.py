# -*- coding: utf-8 -*-
"""
CFG Project, Midsem code

Team: Anshita Singh, Karuna Prakash
"""
import os
import time
import psutil
start_time = time.time()
from pyfaidx import Fasta
import numpy as np
import pandas as pd
from itertools import product
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc

def markov(order, tf, train_indices, test_indices, df, sequence, chr_name): 
    '''
    Markov model prediction

    Parameters
    ----------
    order : markov model order
    tf : transcription factor name
    train_indices : indices (from .tsv file) to use for training
    test_indices : indices (from .tsv file) to use for testing
    df : dataframe from .tsv file
    sequence : genome sequence
    chr_name : name of the chromosome

    Returns
    -------
    test_df : dataframe containing log-likelihood of for each bin in testing data

    '''
    test_df  = df.loc[test_indices]  
    p_col = []
    # TRAINING:
    combos = [''.join(p) for p in product('ATGC', repeat=order)] #all possible order-mers
    tm_bound_dict = {c: {n: 1 for n in 'ATGC'} for c in combos}  # transition matrix for bound
    tm_unbound_dict = {c: {n: 1 for n in 'ATGC'} for c in combos} #transition matrix for unbound
    for i in train_indices:
        start_pos = df.loc[i]['start']
        end_pos = df.loc[i]['end']
        nctds = str(sequence[f"chr{str(chr_name)}"][start_pos:end_pos].seq) #nucleotides
        if 'N' in nctds: #ignore the Ns
            continue
        state = df.loc[i][tf] #bound or unbound
        for j in range(len(nctds) - order):
            ordermer = (nctds[j:j+order]).upper()
            next_letter = (nctds[j+order]).upper()
            # add to count to the corresponding transition matrix
            if state == 'B':
                tm_bound_dict[ordermer][next_letter] += 1
            elif state == 'U':
                tm_unbound_dict[ordermer][next_letter] += 1
    for keys,vals in tm_bound_dict.items(): #normalising the tranition matrices
        #keys: prev nucleotide order-mer
        #vals: next nucleotide 
        count_sum = sum(vals.values())
        for nuc in vals:
            vals[nuc] = np.log(vals[nuc]/count_sum)
    for keys,vals in tm_unbound_dict.items():
        count_sum = sum(vals.values())
        for nuc in vals:
            vals[nuc] = np.log(vals[nuc]/count_sum)
    # TESTING:
    for i in test_indices:
        start_pos = df.loc[i]['start']
        end_pos = df.loc[i]['end']
        test_nctds = str(sequence[f"chr{str(chr_name)}"][start_pos:end_pos].seq) #nucleotides
        prob_bound_log, prob_unbound_log = order*np.log(0.25), order*np.log(0.25) #assume equal probability for first few nctds
        for j in range(len(test_nctds)- order):
            ordermer_test =  (test_nctds[j:j+order]).upper()
            next_letter_test = (test_nctds[j+order]).upper()
            prob_bound_log += tm_bound_dict[ordermer_test][next_letter_test] 
            prob_unbound_log += tm_unbound_dict[ordermer_test][next_letter_test]
        log_lik = prob_bound_log-prob_unbound_log #log likelihood
        p_col.append(log_lik)
    test_df[f'log_lik'] = p_col
    return test_df

def k_fold(k, order,tf, sequence, chr_name):
    '''
    Performs k-fold validation.
    Separates bound and unbound sequences first, then divides into k parts
    Therefore ratio B:U is the same across all the k folds
    Displays both curves, with AUCs 
    
    Parameters
    ----------
    k : number of folds
    order : markov model order
    tf : transcription factor name
    sequence : genome sequence
    chr_name : name of the chromosome

    Returns
    -------
    pr_auc : area under precision recall
    roc_auc : area under ROC

    '''
    chr_df = pd.read_csv(f"chr{str(chr_name)}_200bp_bins.tsv", sep = "\t")
    bound_chr_df = chr_df[chr_df.loc[:, tf]=='B'] 
    unbound_chr_df = chr_df[chr_df.loc[:, tf]=='U'] 
    indices_bound = np.arange(len(bound_chr_df))
    indices_unbound = np.arange(len(unbound_chr_df))
    rng = np.random.default_rng(seed=42)
    rng.shuffle(indices_bound)
    rng.shuffle(indices_unbound)
    #np.random.shuffle(indices_bound)
    #np.random.shuffle(indices_unbound)
    final_bound = np.array_split(indices_bound,k)   
    final_unbound = np.array_split(indices_unbound,k)   
    auc_dict = {'pr':[], 'roc': []}
    for i in range(k):
        print(f"Currently in fold: {i+1}/{k}")
        k_parts = list(np.arange(k))
        k_parts.remove(i)
        test = np.concatenate((final_bound[i], final_unbound[i]))
        train_bound = np.concatenate([final_bound[ind] for ind in k_parts])
        train_unbound = np.concatenate([final_unbound[ind] for ind in k_parts])
        train = np.concatenate((train_bound, train_unbound))
        result = markov(order, tf, train, test, chr_df, sequence, chr_name)
        fpr, tpr, _ = roc_curve(result['CTCF'], result['log_lik'], pos_label='B')
        prec, recall, _ = precision_recall_curve(result['CTCF'], result['log_lik'], pos_label='B')
        roc_auc = auc(fpr,tpr)
        pr_auc = auc(recall, prec)
        auc_dict['pr'].append(pr_auc)
        auc_dict['roc'].append(roc_auc)
        plt.figure(figsize = (8,8))
        plt.plot(fpr, tpr, label='ROC')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic, auROC = {roc_auc:.4f}, TF = {tf}, Order = {order}')
        plt.show()
        plt.figure(figsize = (8,8))
        plt.plot(recall, prec, label='PRC')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall, auPRC = {pr_auc:.4f}, TF = {tf}, Order = {order}')
        plt.show()
    pr_avg = sum(auc_dict['pr'])/k
    roc_avg = sum(auc_dict['roc'])/k
    return pr_avg, roc_avg

genes = Fasta("hg38.fa")
num_folds = int(input('Enter number of folds for crossvalidation (3-5): '))
markov_order = int(input('Enter markov order (0-10): '))
tf_name = input('Enter name of TF (CTCF, REST or EP300) in all caps: ')
chr_num = int(input('Enter chromosome number (eg: 11. Do NOT enter chr11): '))
print("-----")
print("Starting...")
pr,roc=  k_fold(num_folds, markov_order,tf_name, genes, chr_num) 
print(f"AVERAGE auROC = {roc:.3f}")       
print(f"AVERAGE auPRC = {pr:.3f}")       

# code stats
end_time = time.time()
elapsed = end_time - start_time
process = psutil.Process(os.getpid())
mem = process.memory_info().rss/1024**2
print(f"memory: {mem:.3f} MB")
print(f"time: {elapsed:.3f} s, approx {(elapsed/60):.3f} min")             
                
            
            
            