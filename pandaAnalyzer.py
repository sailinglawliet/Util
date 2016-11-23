# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a Analyzer used Pandas, so make sure that u have installed it.
Pandas based on Python and numpy.
"""
#Author     = "Adryan@hzg"
#MailTo     = "lihongyuan@hzgit.com"
#Time       = "2016/10/19"
#Version    = "1.0"


import os
import os.path  
import gzip
import re

import pandas as pd
import numpy as np
#import matplotlib
#import matplotlib.pyplot as plt
#matplotlib.style.use('ggplot')
#pd.set_option('mpl_style', 'default')

IP_PAT      = re.compile(r'(?<![\.\d])(?:\d{1,3}\.){3}\d{1,3}(?![\.\d])')
MAX_BEL     = 5
MAX_PFIX    = 5
FIG_HOLD    = 10
TOP_HOLD    = 7
TYPE_HOLD   = 20            

class PandaAnalyzer:
    '''private attrs'''
    __gzFile    = None
    __path      = None
    __oPath     = None
    '''public attrs'''
    mapHostAndMIME = {}
    mapForPandas   = {}
    mapMimeTypes   = []
    
    def __init__(self, path, oPath=None):
        """constructor"""
        self.__path = path
        self.__oPath = oPath
        
    def __del__(self):
        """deconstructor"""
        if self.__gzFile:
            self.__gzFile.close()
    
    
    def readGZFile(self):
        """unzip and read file all"""
        if os.path.exists(self.__path):
            with gzip.open(self.__path, 'r') as gf:
                self.__gzFile = gf
                return gf.read()
        else:
            print('the path [{}] is not exist!'.format(self.__path))
        
            
    def readGZFileIter(self):
        """unzip and read file in iterator way"""
        if os.path.exists(self.__path):
            with gzip.open(self.__path, 'r') as gf:
                self.__gzFile = gf
                for line in gf:
                    yield line
        else:
            print('the path [{}] is not exist!'.format(self.__path))
            
        
    def preScan(self):
        """do some predos and put into list"""
        gzfile = self.readGZFileIter()
        if getattr(gzfile, '__iter__', None):
            tmpMap = {}
            tmpList = []
            lineList = []
            for line in gzfile:
                lineList.append(line)
                """split by BEL"""
                tmpLineByBel = line.split("")
                if len(tmpLineByBel) >= MAX_BEL:             
                    host = tmpLineByBel[1]
                    uri = tmpLineByBel[3]
                    '''get sum by postfix'''    
                    tmpUri = uri.rsplit('.', 1)
                    if (len(tmpUri) == 2) and (len(tmpUri[1]) <= MAX_PFIX):
                        tmpMap[tmpUri[1]] = tmpMap.get(tmpUri[1], 0) + 1
                    else:
                        '''Qustion Mark'''
                        tmpUri = uri.split('?', 1)
                        if (len(tmpUri) == 2):
                            tmpUri = tmpUri[0].rsplit('.', 1)
                            if (len(tmpUri) == 2) and (len(tmpUri[1]) <= MAX_PFIX):
                                tmpMap[tmpUri[1]] = tmpMap.get(tmpUri[1], 0) + 1
            for k, v in tmpMap.iteritems():
                tmpList.append((k, v))
            tmpList.sort(key=lambda x:x[1], reverse=True)
            tmpList = tmpList[0:TYPE_HOLD]
            tList = []
            for Ftype in tmpList:
                tList.insert(len(tList), Ftype[0])
            self.mapMimeTypes = tList
            print(tList)
            for line in lineList:
                #print(line)
                """split by BEL"""
                tmpLineByBel = line.split("")
                if len(tmpLineByBel) >= MAX_BEL:             
                    host = tmpLineByBel[1]
                    '''
                    tmpLine1 = tmpLineByBel[0].split(",")
                    date = tmpLine1[0]
                    protocol = tmpLine1[1]
                    '''
                    uri = tmpLineByBel[3]
                    '''strong match perhaps Bayes?'''
                    for Mtype in self.mapMimeTypes:
                        if uri.endswith(Mtype):
                            if not self.mapHostAndMIME.get(host, False):
                                self.mapHostAndMIME[host] = {}
                            self.mapHostAndMIME[host][Mtype] = self.mapHostAndMIME[host].get(
                                Mtype, 0) +1
                            break
                        else:
                            '''deal with ?'''
                            tmpUri = uri.split('?', 1)
                            if (len(tmpUri) == 2) and tmpUri[0].endswith(Mtype):
                                if not self.mapHostAndMIME.get(host, False):
                                    self.mapHostAndMIME[host] = {}
                                self.mapHostAndMIME[host][Mtype] = self.mapHostAndMIME[host].get(
                                    Mtype, 0) +1
                                break
                        
    def transMap(self):
        """tansform a {k1:{}, k2:{}} like map to a {k1:[], k2:[]} like map"""
        if self.mapHostAndMIME:
            '''c subnet merge'''
            tmp = {}
            tmpCheck = {}
            for k, v in self.mapHostAndMIME.iteritems():
                mObj = IP_PAT.match(k)
                if mObj:
                    ipStr = k.rsplit('.', 1)
                    if len(ipStr) == 2:
                        if tmpCheck.get(ipStr[0], False):
                            '''hitted'''
                            realKey = tmpCheck[ipStr[0]]
                            for i, j in self.mapHostAndMIME[realKey].iteritems():
                                if not tmp.get(realKey, False):
                                    tmp[realKey] = {}
                                tmp[realKey][i] = j + v.get(i, 0)
                        else:
                            tmp[k] = v
                            tmpCheck[ipStr[0]] = k
                else:
                    tmp[k] = v
            self.mapHostAndMIME = tmp
            '''top rank'''
            tmp = {}
            tmpList = []
            for k, v in self.mapHostAndMIME.iteritems():
                tmpList.append((k, sum(v.values())))
            tmpList.sort(key=lambda x:x[1], reverse=True)
            tmpList = tmpList[0:TOP_HOLD]
            for t in tmpList:
                key = t[0]
                tmp[key] = self.mapHostAndMIME[key]
            self.mapHostAndMIME = tmp
            '''threshold'''
            for k, v in self.mapHostAndMIME.iteritems():
                #print '%s:%s' % (k, v)
                if sum(v.values()) > FIG_HOLD:    
                    if not self.mapForPandas.get(k, False):
                        self.mapForPandas[k] = [np.nan] * (len(self.mapMimeTypes))
                    for i, j in v.iteritems():
                        self.mapForPandas[k][self.mapMimeTypes.index(i)] = j
            
                
    def analyseHostnMIME2d(self):
        """real analyses core"""
        self.transMap()
        if self.mapForPandas:
            df = pd.DataFrame(data=self.mapForPandas, index=self.mapMimeTypes)
            #print(df)
            #plt.figure()
            #df.plot.barh(stacked=True)
            #df.plot.bar(stacked=True, figsize=(10, 12), grid=True)
            png = df.plot.bar(stacked=True, figsize=(10, 12), grid=True).get_figure()
            self.fileWriter('json', df, png)
            
            
    def fileWriter(self, FType, DataFrame, Pic=None):
        """output writer"""
        if (FType == 'json'):
            json = DataFrame.to_json()
            realOutPath = None
            if self.__oPath:
                #realOutPath = os.path.join(self.__oPath, 'output.json')
                realOutPath = self.__oPath
            else:
                tmpPath = os.path.dirname(self.__path)
                realOutPath = tmpPath
            opath = os.path.join(realOutPath, 'o.json')
            f = open(opath, 'w')
            f.write(json)
            f.close()
            if Pic:
                Pic.savefig(os.path.join(realOutPath, 'o.png'))
        
        
        
def main():
    #print(os.getcwd())
    Analyzer = PandaAnalyzer(r"C:\Users\Adryan\Documents\test.gz")
    Analyzer.preScan()
    Analyzer.analyseHostnMIME2d()
    
    
if __name__ == '__main__':
    main()