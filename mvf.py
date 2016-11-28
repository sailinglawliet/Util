#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 09:52:25 2016

@Author : Adryan
@MailTo : lihongyuan@hzgit.com
@Version: 1.0
"""


import scipy.stats as spst
import pandas as pd
import numpy as np
import sklearn.ensemble as sken
import sklearn.linear_model as skli
import matplotlib.pyplot as plt


def analyze(filename1, filename2):
    df1 = pd.read_json(filename1)
    df2 = pd.read_json(filename2)
    df = pd.concat([df1, df2], ignore_index=True)
    df = df.fillna(0)
#    cor_mat = np.corrcoef(df)
#    for coef in cor_mat[len(cor_mat)-1]:
#        print coef
    colList = []
    for col in df.columns:
        if col!='flow':
            print col
            colList.append(col)
            r, v = spst.pearsonr(df[col], df['flow'])
            print r, v
            r, v = spst.spearmanr(df[col], df['flow'])
            print r, v
            r, v = spst.kendalltau(df[col], df['flow'])
            print r, v           
            print
    tmpdf = pd.DataFrame(df[['flow']])
    tmpdf['res'] = map(lambda x: 1 if x>0.3 else 0, tmpdf['flow'])
    clf = sken.RandomForestClassifier(n_estimators=100, max_features=10)
    df = df[df.columns[0:len(df.columns)-1]]
    clf = clf.fit(df, tmpdf['res'])
    for col in colList:
        print colList.index(col), col, clf.feature_importances_[colList.index(col)]
    std = np.std([tree.feature_importances_ for tree in clf.estimators_],
             axis=0)
    indices = np.argsort(clf.feature_importances_)[::-1]
    plt.figure(figsize=(8, 6))
    plt.title("Feature importances")
    plt.bar(range(df.shape[1]), clf.feature_importances_[indices],
           color="r", yerr=std[indices], align="center")
    plt.xticks(range(df.shape[1]), indices)
    plt.xlim([-1, df.shape[1]])
    plt.show()
#    lr = skli.LinearRegression()
#    lr.fit(df, tmpdf['res'])
#    print lr.coef_, lr.intercept_

            
def main():
    analyze('sqlRes1.json', 'sqlRes2.json')

if __name__=='__main__':
    main()