"""
CFG Project - Feb 2026, Markov model based classifier 
=====================================================

Team: 
Anshita Singh <anshita.singh@students.iiserpune.ac.in>
Karuna Prakash <karuna.prakash@students.iiserpune.ac.in>
"""
import os
import time
import psutil
start_time = time.time()
from Bio import SeqIO
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn import metrics
import matplotlib.pyplot as plt

### provide chromosome inputs: chr_seq (fasta), chip data (tsv), tf name4
genome_fa_file = r"\hg38.fa"
chr_num = int(input("Enter chromosome number (eg: 4. DO NOT enter chr4):")) 
chip_data = rf"\chr{chr_num}_200bp_bins.tsv"
tf_name = 'REST' #str(input("Enter transcription factor name (CTCF, REST or EP300):"))
m = int(input("Order for markov model (0-10):"))
k = int(input("Number of folds for cross-validation (3-5):"))

df = pd.read_csv(chip_data, sep='\t', header=0)

records = SeqIO.to_dict(SeqIO.parse(genome_fa_file, "fasta"))
chr_seq = records[f'chr{chr_num}'].seq.upper()

### MODEL
def markov_model(seq_tsv):
  """
  Markov Model
  
  Parameters
  ----------
  seq_tsv : .tsv file for the training data (for a given chromosome, in 200bp bins)

  Returns
  -------
  pwm : position weight matrix for the first m-1 positions
  transition_prob : transition matrix for order m markov model

  """

# position weight matrix for the first m-1 positions
  nucleotides = "ACGT"
  pwm_counts = np.ones((4,m))
  for i in range(len(seq_tsv)):
    seq = str(chr_seq[seq_tsv.iloc[i,1]:seq_tsv.iloc[i,2]])
    for j in range(m):
      for k in range(4):
        pwm_counts[k,j] += 1 if seq[m] == nucleotides[k] else 0
  pwm = pwm_counts/(len(seq_tsv)+4)

# training markov model for order m
  counts = defaultdict(lambda: defaultdict(int))
  transition_prob = {}
  for i in range(len(seq_tsv)):
    seq = str(chr_seq[seq_tsv.iloc[i,1]:seq_tsv.iloc[i,2]]) #get seq for every bin
    for j in range(m, 200):
      prefix = seq[j-m:j] # length m prefix
      nuc = seq[j] # (m+1)-th nucleotide
      counts[prefix][nuc] += 1 if nuc in nucleotides else 0
  for prefix, nuc_counts in counts.items():
    denominator = sum(nuc_counts.values()) #total occurences of the prefix
    transition_prob[prefix] = {}
    for nuc, count in nuc_counts.items():
      transition_prob[prefix][nuc] = count/denominator 
  return pwm, transition_prob

# calculating log-odds ratio
def log_odds_score(test_tsv): 
  """
  Log-likelihood for a test sequence. 

  Parameters
  ----------
  test_tsv : .tsv file for the testing set

  Returns
  -------
  scores : list of log-likelihood scores for bins in the testing set

  """
  scores = []
  nucleotides = "ACGT"
  bound_pwm, bound_model = markov_model(bound_train)
  unbound_pwm, unbound_model = markov_model(unbound_train)

  for i in range(len(test_tsv)):
    seq = str(chr_seq[test_tsv.iloc[i,1]:test_tsv.iloc[i,2]]) #get seq for every bin
    score = 0
    for j in range(m): #for the first m-1 nucleotides
      nuc = seq[j]
      for k in range(4):
        log_prob_pwm = np.log(bound_pwm[k, j]) - np.log(unbound_pwm[k, j]) if nuc == nucleotides[k] else 0
        score += log_prob_pwm
    for j in range(m, 200): 
      prefix = seq[j-m:j] # length m prefix
      nuc = seq[j] # (m+1)-th nucleotide
      if prefix not in bound_model.keys():
        log_prob = 0
      elif prefix not in unbound_model.keys():
        log_prob = 0
      else: 
        log_prob = np.log(float(bound_model[prefix].get(nuc, 1))) - np.log(float(unbound_model[prefix].get(nuc, 1)))
      score += log_prob
    scores.append(score)
  return scores

### K-FOLD CROSS VALIDATION

# separating the bound and unbound regions
bound_condition = df.loc[:, tf_name] == "B"
bound_seq = df[bound_condition]
unbound_seq = df[~bound_condition]

# spliting data into k parts
def k_folds(k, seq_tsv):
  """
  Splits data into k parts (after shuffling).

  Parameters
  ----------
  k: number of folds
  seq_tsv : tsv file for chip seq data (for a given chromosome)

  Returns
  -------
  folds_df : list of k pd.dataframes

  """
  shuffled_seq = seq_tsv.sample(frac=1, ignore_index=True, random_state=1) # random_state=1 for reproducibilty, None otherwise
  folds = np.array_split(shuffled_seq, k) #list of arrays
  folds_df = [pd.DataFrame(f, columns=seq_tsv.columns) for f in folds] #list of dataframes
  return(folds_df)

parts_bound = k_folds(k, bound_seq)
parts_unbound = k_folds(k, unbound_seq)

roc_auc = np.array([])
prc_auc = np.array([])

for i in range(k):
  test = pd.concat([parts_bound[i], parts_unbound[i]]).sample(frac=1, ignore_index=True, random_state=1)
  bound_folds_train = [parts_bound[j] for j in range(k) if j != i]
  bound_train = pd.concat(bound_folds_train)
  unbound_folds_train = [parts_unbound[j] for j in range(k) if j != i]
  unbound_train = pd.concat(unbound_folds_train)
  ## making roc, prc
  y_score = np.array(log_odds_score(test))
  y_true = test[tf_name]
  fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score, pos_label="B")
  roc_auc = np.append(roc_auc, np.array(metrics.auc(fpr, tpr)))
  roc_display = metrics.RocCurveDisplay.from_predictions(y_true, y_score, pos_label="B", plot_chance_level=True, chance_level_kw={'color':'grey', 'linestyle':'--'}).plot()
  roc_display.ax_.set_title(f"ROC {i+1} of {k} for {tf_name} in chr {chr_num}, order {m} MM")
  #plt.show()
  plt.savefig(f"chr{chr_num}_order{m}_roc{i+1}of{k}_{tf_name}.png")
  precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_score, pos_label="B")
  prc_auc = np.append(prc_auc, np.array(metrics.auc(recall, precision)))
  prc_display = metrics.PrecisionRecallDisplay.from_predictions(y_true, y_score, pos_label="B", plot_chance_level=True, chance_level_kw={'color':'grey', 'linestyle':'--'}).plot()
  prc_display.ax_.set_title(f"PRC {i+1} of {k} for {tf_name} in chr {chr_num}, order {m} MM")
  #plt.show()
  plt.savefig(f"chr{chr_num}_order{m}_prc{i+1}of{k}_{tf_name}.png")

print(f"Mean area under ROC: {roc_auc.mean():.3f}")
print(f"Mean area under PRC: {prc_auc.mean():.3f}")

end_time = time.time()
elapsed = end_time - start_time
process = psutil.Process(os.getpid())
mem = process.memory_info().rss/1024**2
print(f"memory: {mem:.3f} MB")
print(f"time: {elapsed:.3f} s, approx {(elapsed/60):.3f} min")