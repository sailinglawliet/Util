#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 10:14:47 2016
This is IP and Host lookup peogram, also can calculate similarity between Hosts

@Author : Adryan
@MailTo : lihongyuan@hzgit.com
@Version: 1.0
"""

import os
import os.path
import gzip
import re

import collections as cl
import pandas as pd
import numpy as np
import scipy.sparse as ss

import algebraUTIL as alg

PA_IP_PAT       = re.compile(r'(?<![\.\d])(?:\d{1,3}\.){3}\d{1,3}(?![\.\d])')
PA_MAX_BEL      = 5
PA_HOST_JSON    = 'host2ip.json'
PA_DETAIL_JSON  = 'tripleValues.json'


def get_largest(row, N=10):
    if N >= row.nnz:
        best = zip(row.data, row.indices)
    else:
        ind = np.argpartition(row.data, -N)[-N:]
        best = zip(row.data[ind], row.indices[ind])
    return sorted(best, reverse=True)


def calculate_similar(similarity, host, hostid):
    neighbours = similarity[hostid]
    top = get_largest(neighbours)
    return [(host[other], score, i) for i, (score, other) in enumerate(top)]


def bm25weight(data, K1=100, B=0.8):
    """ Weighs each row of the matrix data by BM25 weighting """
    # calculate idf per term (user)
    N = float(data.shape[0])
    idf = np.log(N / (1 + np.bincount(data.col)))

    # calculate length_norm per document (artist)
    row_sums = np.squeeze(np.asarray(data.sum(1)))
    average_length = row_sums.sum() / N
    length_norm = (1.0 - B) + B * row_sums / average_length

    # weight matrix rows by bm25
    ret = ss.coo_matrix(data)
    ret.data = ret.data * (K1 + 1.0) / (K1 * length_norm[ret.row] + ret.data) * idf[ret.col]
    return ret
    

def bm25(data):
    data = bm25weight(data)
    return data.dot(data.T)
    

class HostnIPData:  
    def __init__(self, path, bType=None):
        self.dataHostMap    = cl.defaultdict(list)
        self.dataIPMap      = cl.defaultdict(list)
        self.dataDetail     = cl.defaultdict(dict)
        self.df             = None
        if bType and bType == '.json':
            self.initFromJson(path)
        else:
            self.walkFiles(path)
        
            
    def genTbl4Host(self, path):
        if os.path.exists(path):
            with gzip.open(path, 'r') as gf:
                gzfile = gf.read()
                gzlist = gzfile.split('\n')
                for line in gzlist:
                    '''split by BEL'''
                    tmpLinesByBel = line.split("")
                    if len(tmpLinesByBel) >= PA_MAX_BEL:
                        host = tmpLinesByBel[1]
                        mObj = PA_IP_PAT.match(host)
                        fdetail = tmpLinesByBel[4].split(',')
                        if len(fdetail)<=10:
                            continue
                        destIP = fdetail[6]
                        '''which means user'''
                        srcIP = fdetail[4]
                        if destIP and not mObj:
                            if destIP not in self.dataHostMap[host]:
                                self.dataHostMap[host].append(destIP)
                        if srcIP:
                            self.dataDetail[srcIP][host] = self.dataDetail[srcIP].get(host, 0) + 1
            gf.close()
                                
    
    def genTbl4IP(self):
        for host, infoList in self.dataHostMap.iteritems():
            for ip in infoList:
                if host not in self.dataIPMap[ip]:
                    self.dataIPMap[ip].append(host)
                
                    
    def initFromJson(self, path):
        '''host and ip'''
        df = pd.read_json(os.path.join(path, PA_HOST_JSON))
#        def sumList(arr):
#            tmp = []
#            for i in arr:           
#                tmp.append(i)
#            return tmp
#        df = df.groupby('Host').agg(sumList)
#        print(df.head())
        for index, rowItem in df.iterrows():
            host, destIP = rowItem[0].encode("ascii") , rowItem[1].encode("ascii") 
            if destIP not in self.dataHostMap[host]:
                self.dataHostMap[host].append(destIP)
            if host not in self.dataIPMap[destIP]:
                self.dataIPMap[destIP].append(host)
        '''detail'''
        self.df = pd.read_json(os.path.join(path, PA_DETAIL_JSON))

                
    def write2Json(self, path):
        '''host and ip'''
        tmpList = []
        for host, infoList in self.dataHostMap.iteritems():
            for ip in infoList:
                tmp = {}
                tmp['Host'] = host
                tmp['IP']   = ip
                tmpList.append(tmp)
        df = pd.DataFrame(tmpList)
        json = df.to_json()
        oPath = os.path.join(path, PA_HOST_JSON)
        f = open(oPath, 'w')
        f.write(json)
        f.close()
        '''detail'''
        tmpList = []
        for srcIP, info in self.dataDetail.iteritems():
            for host, cnt in info.iteritems():
                tmp = {}
                tmp['SrcIP'] = srcIP
                tmp['Host']  = host
                tmp['Count'] = cnt
                tmpList.append(tmp)
        df = pd.DataFrame(tmpList)
        self.df = df
        json = df.to_json()
        oPath = os.path.join(path, PA_DETAIL_JSON)
        f = open(oPath, 'w')
        f.write(json)
        f.close()
                   
        
    def walkFiles(self, path):
        if os.path.isdir(path):
            for f in os.listdir(path):
                if f.endswith('gz') and os.path.isfile(os.path.join(path, f)):
                    self.genTbl4Host(os.path.join(path, f))
            self.genTbl4IP()
            self.write2Json(path)
            
            
    def showTriTree(self):
        pass
    
    

class IPClassifier:
    def __init__(self, dataFrame):
        self.df = dataFrame
        #self.hostSets   = dict((host, set(srcIP)) for host, srcIP in self.df.groupby('Host')['SrcIP'])
        #self.srcIPSets  = dict((srcIP, set(host))for srcIP, host in self.df.groupby('SrcIP')['Host'])
        srcIPids = cl.defaultdict(lambda: len(srcIPids))
        self.df['SrcIPId'] = self.df['SrcIP'].map(srcIPids.__getitem__)
        self.hostMatrix = dict((host, ss.csr_matrix((np.array(group['Count']), (np.zeros(len(group)), group['SrcIPId'])),
                                                    shape=[1, len(srcIPids)])) for host, group in self.df.groupby('Host'))
        N = len(self.hostMatrix)
        self.idf = [1. + np.log(N / (1. + p)) for p in self.df.groupby('SrcIPId').size()]
        self.averageCnt = self.df['Count'].sum() / float(N)
        
        
    def inQuery(self):
        pass
    
    
    
def main():
    hostNIpData = HostnIPData(r'C:\Users\Adryan\Documents\data2do', '.json')
    ipc = IPClassifier(hostNIpData.df)
    #print alg.tfidf(ipc.hostMatrix['mvimg2.meitudata.com'], ipc.hostMatrix['imgsrc.baidu.com'], ipc.idf)
    ipc.df['SrcIP']   = ipc.df['SrcIP'].astype("category")
    ipc.df['Host']    = ipc.df['Host'].astype("category")
    host = dict(enumerate(ipc.df['Host'].cat.categories))
    cnts = ss.coo_matrix((ipc.df['Count'].astype(float), (ipc.df['Host'].cat.codes.copy(), ipc.df['SrcIP'].cat.codes.copy())))
    src_cnts = ipc.df.groupby('Host').size()
    to_generate = sorted(list(host), key=lambda x: -src_cnts[x])
    similarity = bm25(cnts)
#    print similarity
#    print similarity
    with open("host_simlarity.tsv", "wb") as ofile:
        for hostid in to_generate:
            name = host[hostid]
            for otherHost, score, rank in calculate_similar(similarity, host, hostid):
                #print name, otherHost, score, rank
                ofile.write("%s\t%s\t%s\t%s\t\n" % (name, otherHost, score, rank+1))
                if (rank+1)==10:
                    ofile.write("\n")
    

if __name__ == '__main__':
    main()