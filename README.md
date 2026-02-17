# Computational Functional Genomics, Jan 2026 - Project

### Team
* Anshita Singh <anshita.singh@students.iiserpune.ac.in>
* Karuna Prakash <karuna.prakash@students.iiserpune.ac.in>

## Requirements
* Language: Python 3.14.3
* Libraries: os, time, psutil, pyfaidx, pandas, numpy, collections, sklearn.metrics (roc_curve, precision_recall_curve, auc), matplotlib
* Fasta file: 'hg38.fa' (download 'hg38.fa.gz' from here: https://hgdownload.cse.ucsc.edu/goldenpath/hg38/bigZips/, and gunzip it)

## File structure
* 'cfg_project.py' - primary script file
* 'SimplerVersion.py' - simpler version of the primary script 
* 'chr{p}_200bp_bins.tsv' - ChIP seq data, contains information about which parts of the chr p (p being any of the chromosome numbers except 3,10 and 17) are bound and unbound to various transcription factors (EP300, CTCF, REST). Please note that the unknown tsvs are not uploaded here.
* 'fastafile.fa' - sample fasta file to test SimplerVersion.py on


## How to run
* For both scripts: Run the script from terminal (eg: python cfg_project.py) and follow the input instructions that are shown on the screen.

  
  
  


