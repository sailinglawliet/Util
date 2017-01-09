#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 01 14:26:37 2016

@Author : Adryan
@MailTo : lihongyuan@hzgit.com
@Version: 1.0
"""

import scipy.stats as spst
import pandas as pd
import numpy as np
import sklearn.ensemble as sken
import sklearn.linear_model as skli
import sklearn.cluster as skcl
import matplotlib.pyplot as plt

def analyze(path):
    df = pd.read_excel(path)
#    ser = df['Host']
    df = df.drop([u'访问次数', 'DestIP', 'Host'], axis=1)
#    print df.head()
    clf = skcl.KMeans(n_clusters=2)
    clf.fit(df)
    res = clf.labels_.tolist()
    for k, l in enumerate(res):
        print k, l
    colList = []
    for col in df.columns:
        print col
        colList.append(col)
        r, v = spst.pearsonr(df[col], res)
        print r, v
        r, v = spst.spearmanr(df[col], res)
        print r, v
        r, v = spst.kendalltau(df[col], res)
        print r, v           
        print
    clf = sken.RandomForestClassifier(n_estimators=100, max_features=10)
    clf = clf.fit(df, res)
    for col in colList:
        print colList.index(col), col, clf.feature_importances_[colList.index(col)]
        

def analyze2d(path):
    df = pd.read_excel(path)
    df = df.drop(['DestIP', 'Host'], axis=1)
#    clf = skcl.KMeans(n_clusters=2)
#    clf.fit(df)
#    res = clf.labels_.tolist()
    res = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
    colList = df.columns.tolist()
    tmp = [0 for _ in xrange(50)] 
    for i in range(100):
        clf = sken.RandomForestClassifier(n_estimators=100, max_features=10)
        clf = clf.fit(df, res)
        for col in colList:
            tmp[colList.index(col)] += clf.feature_importances_[colList.index(col)]
#            print colList.index(col), col, clf.feature_importances_[colList.index(col)]
    for k, v in enumerate(tmp):
        if v > 0:
            print colList[k], v

def main():
#    analyze(r'C:\Users\Adryan\Documents\hostCDR.xls')
    analyze2d(r'C:\Users\Adryan\Documents\hostCDR2d.xls')

if __name__=='__main__':
    main()