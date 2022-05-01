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
    parser.add_argument("--p_pheno",help="path to pheno file",dest='p_pheno',
                        default='/home/harveyaa/projects/def-pbellec/harveyaa/data/pheno_01-12-21.csv')
    parser.add_argument("--p_conn",help="path to connectomes file",dest='p_conn',
                        default='/home/harveyaa/projects/def-pbellec/harveyaa/data/connectomes_01-12-21.csv')
    parser.add_argument("--p_ids",help="path to dataset ids",dest='p_ids')
    parser.add_argument("--p_out",help="path to outputs",dest='p_out')
    args = parser.parse_args()

    print('#############')
    print('# BENCHMARK #')
    print('#############\n')

    # Load data
    print('Loading data...')
    pheno = pd.read_csv(args.p_pheno,index_col=0)
    conn = pd.read_csv(args.p_conn,index_col=0)
    print('Done!')

    # Define cases
    cases = ['SZ',
        'ASD',
        'BIP',
        'DEL22q11_2',
        'DUP22q11_2',
        'DEL16p11_2',
        'DUP16p11_2',
        'DEL1q21_1',
        'DUP1q21_1']

    # Define confounds
    conf = ['AGE',
            'SEX',
            'SITE',
            'mean_conn',
            'FD_scrubbed']

    # Define classifiers
    clfs = {'SVC_1':SVC(C=1,class_weight='balanced'),
            'SVC_10':SVC(C=10,class_weight='balanced'),
            'SVC_100':SVC(C=100,class_weight='balanced'),
            #'LR':LogisticRegression(class_weight='balanced'),
            'kNN_5':KNeighborsClassifier()}

    #############
    # BENCHMARK #
    #############

    mean_acc_conf = {}
    mean_acc_conn = {}
    for clf in clfs:
            mean_acc_conf[clf] = []
            mean_acc_conn[clf] = []
    
    print('Beginning prediction...')
    for case in cases:
        print(case)
        # Load ids
        if os.path.exists(os.path.join(args.p_ids,f"{case}.csv")):
            dataset_ids = pd.read_csv(os.path.join(args.p_ids,f"{case}.csv"),index_col=0)

            # Confound matrix
            df = pheno[pheno.index.isin(dataset_ids.index)]
            X = pd.get_dummies(df[conf],columns=['SEX','SITE'],drop_first=True)

            # Connectomes
            X_conn = conn[conn.index.isin(dataset_ids.index)]

            # Labels
            y = dataset_ids[case]

            acc_conf = {}
            acc_conn = {}
            for clf in clfs:
                acc_conf[clf] = []
                acc_conn[clf] = []
                
            for i in range(5):
                for clf in clfs:
                    if f'fold_{i}' in dataset_ids.columns:
                        # Test set ids for fold
                        test_mask = (dataset_ids[f'fold_{i}'] == 1).to_numpy()

                        # Train/test split
                        X_train, X_test = X[~test_mask], X[test_mask]
                        X_conn_train, X_conn_test = X_conn[~test_mask], X_conn[test_mask]
                        y_train, y_test = y[~test_mask], y[test_mask]

                        # Pred from confounds
                        clfs[clf].fit(X_train,y_train)
                        pred = clfs[clf].predict(X_test)
                        acc_conf[clf].append(accuracy_score(y_test,pred))

                        # Pred from connectomes
                        clfs[clf].fit(X_conn_train,y_train)
                        pred_conn = clfs[clf].predict(X_conn_test)
                        acc_conn[clf].append(accuracy_score(y_test,pred_conn))
        else:
            print('No CV folds found.')
        for clf in clfs:
            mean_acc_conf[clf].append(np.mean(acc_conf[clf]))
            mean_acc_conn[clf].append(np.mean(acc_conn[clf]))
    print('Done!\n')

    ###########
    # RESULTS #
    ###########
    print('Collecting results...')
    results = {}
    for clf in clfs:
        results[clf] = pd.DataFrame([mean_acc_conf[clf],mean_acc_conn[clf]],columns=cases,index=['conf','conn']).transpose()
    
    print('Generating plot...')
    title = 'Confound Accuracy'
    xlabels=None

    fig,ax = plt.subplots(1,len(cases),figsize=(int(len(cases)/2)+7,6),sharey=True,sharex=True)

    plt.yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
    plt.ylim(0,1)
    sns.set_style('whitegrid')

    colors = ['navy','darkorchid','red','orange','dodgerblue','forestgreen']
    for j,case in enumerate(cases):
        for i,clf in enumerate(clfs):
            conf_acc = results[clf].loc[case,'conf']
            conn_acc = results[clf].loc[case,'conn']

            if conf_acc > conn_acc:
                mfc = 'white'
                ls =''
            else:
                mfc = colors[i]
                ls = '-'
            ax[j].plot(i,conn_acc,marker='o',color=colors[i],ms=4,markerfacecolor=mfc)
            ax[j].plot(i,conf_acc,marker='o',color=colors[i],ms=4,markerfacecolor=mfc,label=clf)
            ax[j].plot((i,i),(conf_acc,conn_acc),color=colors[i],ls=ls,label=clf)
            ax[j].set_xticklabels([])
            ax[j].set_xticks([])
            ax[j].set_xlim(-0.75,5.75)

            if not xlabels is None:
                ax[j].set_xlabel(xlabels[j],rotation=90) 
            else:
                ax[j].set_xlabel(case,rotation=270)

            lines = ax[j].get_ygridlines()
            b = lines[5]
            b.set_color('black')
            b.set_linewidth(1.15)
            b.set_linestyle('--')

        handles, _ = ax[-1].get_legend_handles_labels()
        labels = clfs.keys()

    ax[0].set_ylabel('Accuracy')
    fig.legend(handles, labels, loc=(0.1,0.2))
    ax[int(len(cases)/2)].set_title(title)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0)
    plt.savefig(os.path.join(args.p_out,'conf_acc.png'),dpi=300)
    print('Done!\n')