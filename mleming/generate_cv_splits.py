import os
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from general_class_balancer import *

from argparse import ArgumentParser

def save_ids(ids, path_out, case,fold=None):
    tag = f'{case}.txt' if fold is None else f'{case}_test_set_{fold}.txt'
    filename = os.path.join(path_out,tag)
    with open(filename, 'w') as file:
        for i in ids:
            file.write(f"{i}\n")


def generate_cv_splits(case,ids,conf,pheno,path_out,plim=0.05, min_folds=5,force_save=False,min_train=0.5, min_members=2):
    print(case)
    control = 'CON_IPC' if case in ['ASD','BIP','SZ'] else 'non_carriers'

    df = pheno[pheno.index.isin(ids[0].to_list())]
    df.loc[:,case] = df.loc[:,case].astype(int)

    subsets = []
    fail_attempts = 0
    while len(subsets) != min_folds:
        confounds = df[conf].transpose().values
        classes = df[case].values.astype(int)
        
        train_size = 0
        ddd = df.copy()
        while train_size < min_train:
            selection = class_balance(classes,confounds,plim=plim,min_members=min_members)
            train_size =  1 - len(selection)/len(df)

            if train_size < min_train:
                classes = classes[selection]
                confounds = confounds[:,selection]
                ddd = ddd[selection]

        selection = df.index.isin(ddd.index)
        fold_ids = df[selection].index.to_list()
        if len(subsets) == 0: # if None so far append
            subsets.append(fold_ids)
        else: # Otherwise check if duplcate with existing selections
            duplicate = []
            for subset in subsets:
                diff = set(subset).difference(set(fold_ids))
                duplicate.append(len(diff) == 0)
                
            if not np.any(duplicate):
                subsets.append(fold_ids)
            else:
                fail_attempts += 1
                
            if fail_attempts > 50:
                print('Failed!')
                break

    print('N subsets: ',len(subsets))
    
    if (len(subsets) == min_folds) | force_save:
        print('Saving fold ids...\n')
        for i in range(len(subsets)):
            save_ids(subsets[i],path_out,case,i)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--p_pheno",help="path to pheno file",
                        default='/home/harveyaa/projects/def-pbellec/harveyaa/data/pheno_01-12-21.csv')
    parser.add_argument("--p_ids",help="path to dataset ids",
                        default='/home/harveyaa/projects/def-pbellec/harveyaa/data/ids/datasets')
    parser.add_argument("--p_out",help="path to outputs")
    parser.add_argument("--plim",help="p value cutoff for class balancing",default=5,type=float)
    parser.add_argument("--min_train",help="min proportion of training sample",default=70,type=float)
    parser.add_argument("--min_members",help="min number of cases in test set",default=3,type=int)
    parser.add_argument("--generate_figures",help="plots of test folds",action='store_true')
    args = parser.parse_args()

    print('#############')
    print('# CV SPLITS #')
    print('#############\n')

    # LOAD DATA
    print('Loading data...')
    pheno = pd.read_csv(args.p_pheno,index_col=0)
    print('Done!\n')

    conf = ['AGE',
            'SEX',
            'SITE',
            'mean_conn',
            'FD_scrubbed']
    
    cases = ['SZ',
        'ASD',
        'BIP',
        'DEL22q11_2',
        'DUP22q11_2',
        'DEL16p11_2',
        'DUP16p11_2',
        'DEL1q21_1',
        'DUP1q21_1']

    
    # LOAD DATASET IDS
    sel_ids = []
    for case in cases:
        df = pd.read_csv(os.path.join(args.p_ids,f"{case}.txt"),header=None)
        sel_ids.append(df)
    sel_ids = dict(zip(cases,sel_ids))

    # GENERATE CV SPLITS
    print('Generating CV splits...')
    temp_dir = tempfile.TemporaryDirectory()
    cv_ids = []
    for case in cases:
        generate_cv_splits(case,
        sel_ids[case],
        conf,
        pheno,
        temp_dir.name,
        plim=args.plim/100,
        force_save=False,
        min_train=args.min_train/100,
        min_members=args.min_members)
    print('Done!\n')

    # CLEAN UP .txt -> .csv
    print('Cleaning up...')
    tag = '{}.txt'
    tag_split = '{}_test_set_{}.txt'

    dfs = []
    for case in cases:
        dataset_ids = sel_ids[case].copy()
        dataset_ids.set_index(0,inplace=True)

        for i in range(5):
            p_split_ids = os.path.join(temp_dir.name,tag_split.format(case,i))
            if os.path.exists(p_split_ids):
                s_ids = pd.read_csv(p_split_ids,header=None)
                dataset_ids[f'fold_{i}'] = 1*dataset_ids.index.isin(s_ids[0].to_list())

        dataset_ids[case] = pheno[pheno.index.isin(dataset_ids.index)][case].values.astype(int)
        dataset_ids.to_csv(os.path.join(args.p_out,f"{case}.csv"))
        dfs.append(dataset_ids)
    print('Done!\n')

    # REPORT
    print('Avg training set sizes...')
    for df in dfs:
        print(df.columns[-1])
        train_sizes = []
        for i in range(5):
            if f'fold_{i}' in df.columns:
                ts = 1-df[f'fold_{i}'].sum()/len(df)
                train_sizes.append(ts)
        print('Avg train %: ',np.mean(train_sizes))
    print('\n')
    
    # PLOT
    if args.generate_figures:
        print('Generating figures...')
        for case in cases:
            all_ids = sel_ids[case]

            # PLOT TEST SET
            fig, ax = plt.subplots(len(conf),5,figsize=(15,12))
            for i,c in enumerate(conf):
                    for fold in range(5):
                            ids = pd.read_csv(os.path.join(temp_dir.name,f"{case}_test_set_{fold}.txt"),header=None)
                            
                            sns.histplot(x=c,data=pheno[pheno.index.isin(ids[0].to_list())],hue=case,bins=25,ax=ax[i,fold])
                            if i == 0:
                                    ax[i,fold].set_title(f'fold {fold}')
                            if fold == 0:
                                    ax[i,fold].set_xlabel('')
                                    ax[i,fold].set_ylabel(c)
                            else:
                                    ax[i,fold].set_xlabel('')
                                    ax[i,fold].set_ylabel('')
                                    ax[i,fold].set_yticklabels([])
            plt.tight_layout()
            plt.subplots_adjust(wspace=0.1,hspace=0.2)
            plt.savefig(os.path.join(args.p_out,f"{case}_test.png"),dpi=300)

            # PLOT TRAIN SET
            fig, ax = plt.subplots(len(conf),5,figsize=(15,12))
            for i,c in enumerate(conf):
                    for fold in range(5):
                            ids_train = list(set(all_ids[0].to_list()).difference(set(ids[0].to_list())))
                            
                            sns.histplot(x=c,data=pheno[pheno.index.isin(ids_train)],hue=case,bins=25,ax=ax[i,fold])
                            if i == 0:
                                    ax[i,fold].set_title(f'fold {fold}')
                            if fold == 0:
                                    ax[i,fold].set_xlabel('')
                                    ax[i,fold].set_ylabel(c)
                            else:
                                    ax[i,fold].set_xlabel('')
                                    ax[i,fold].set_ylabel('')
                                    ax[i,fold].set_yticklabels([])
            plt.tight_layout()
            plt.subplots_adjust(wspace=0.1,hspace=0.2)
            plt.savefig(os.path.join(args.p_out,f"{case}_train.png"),dpi=300)
        print('Done!\n')
    
    temp_dir.cleanup()