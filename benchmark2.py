import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--tasks",help="tasks",dest='tasks',nargs='*')
    parser.add_argument("--encoder",help="Which encoder to use.",dest='encoder',default=0,type=int)
    parser.add_argument("--head",help="Which head to use.",dest='head',default=0,type=int)
    parser.add_argument("--data_dir",help="path to data dir",dest='data_dir',
                        default='/home/harveyaa/Documents/fMRI/data/ukbb_9cohorts/')
    parser.add_argument("--data_format",help="data format code",dest='data_format',default=1,type=int)
    parser.add_argument("--log_dir",help="path to log_dir",dest='log_dir',default=None)
    parser.add_argument("--batch_size",help="batch size for training/test loaders",default=16,type=int)
    parser.add_argument("--lr",help="learning rate for training",default=1e-3,type=float)
    parser.add_argument("--num_epochs",help="num_epochs for training",default=100,type=int)
    args = parser.parse_args()

    p_pheno = '/home/harveyaa/Documents/fMRI/data/ukbb_9cohorts/pheno_26-01-22.csv'
    p_conn = '/home/harveyaa/Documents/fMRI/data/ukbb_9cohorts/connectomes_01-12-21.csv'
    p_ids = '/home/harveyaa/Documents/masters/MTL/conf_balancing/dataset_ids'

    pheno = pd.read_csv(p_pheno,index_col=0)
    conn = pd.read_csv(p_conn,index_col=0)

    cases = ['SZ',
        'ASD',
        'BIP',
        'DEL22q11_2',
        'DUP22q11_2',
        'DEL16p11_2',
        'DUP16p11_2',
        'DEL1q21_1',
        'DUP1q21_1']

    conf = ['AGE',
            'SEX',
            'SITE',
            'mean_conn',
            'FD_scrubbed']

    clfs = {'SVC_1':SVC(C=1,class_weight='balanced'),
            'SVC_10':SVC(C=10,class_weight='balanced'),
            'SVC_100':SVC(C=100,class_weight='balanced'),
            'LR':LogisticRegression(class_weight='balanced'),
            'kNN_5':KNeighborsClassifier()}

    mean_acc_conf = {}
    mean_acc_conn = {}
    for clf in clfs:
            mean_acc_conf[clf] = []
            mean_acc_conn[clf] = []
    
    for case in cases:
        print(case)
        dataset_ids = pd.read_csv(os.path.join(p_ids,f"{case}.txt"),header=None)[0].to_list()
        df = pheno[pheno.index.isin(dataset_ids)]

        X = pd.get_dummies(df[conf],columns=['SEX','SITE'],drop_first=True)
        X_conn = conn[conn.index.isin(dataset_ids)]
        y = df[case]

        acc_conf = {}
        acc_conn = {}
        for clf in clfs:
            acc_conf[clf] = []
            acc_conn[clf] = []

        for i in range(5):
            for clf in clfs:
                if os.path.exists(os.path.join(p_ids,f"{case}_test_set_{i}.txt")):
                    # LOAD IDS
                    fold_test_ids = pd.read_csv(os.path.join(p_ids,f"{case}_test_set_{i}.txt"),header=None)[0].to_list()
                    test_mask = df.index.isin(fold_test_ids)

                    # TRAIN/TEST SPLIT
                    X_train, X_test = X[~test_mask], X[test_mask]
                    X_conn_train, X_conn_test = X_conn[~test_mask], X_conn[test_mask]
                    y_train, y_test = y[~test_mask], y[test_mask]

                    
                    # PRED FROM CONFOUNDS
                    clfs[clf].fit(X_train,y_train)
                    pred = clfs[clf].predict(X_test)
                    acc_conf[clf].append(accuracy_score(y_test,pred))

                    # PRED FROM CONNECTOMES
                    clfs[clf].fit(X_conn_train,y_train)
                    pred_conn = clfs[clf].predict(X_conn_test)
                    acc_conn[clf].append(accuracy_score(y_test,pred_conn))
                    n_folds +=1
        for clf in clfs:
            mean_acc_conf[clf].append(np.mean(acc_conf[clf]))
            mean_acc_conn[clf].append(np.mean(acc_conn[clf]))