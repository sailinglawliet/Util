#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 09:25:05 2017

@Author: Adryan
@MailTo: lihongyuan@hzgit.com
@Version: 1.0
"""

import os
import os.path
import gzip
import argparse
import time
import re

import pandas as pd
import numpy as np
import collections as cl

PA_IP_PAT      = re.compile(r'(?<![\.\d])(?:\d{1,3}\.){3}\d{1,3}(?![\.\d])')
PA_MAX_BEL     = 5
STR = ['SESSION_ENDED', 'CMDBMISS', 'DOWNLOADER', 'TIMEOUT', 'ERROR', 'RST', 'FIN', 'UPLPOADER', 'UNKNOW', 'FORMAT_CHANGED']
CDR_TYPE = ['CACHE_OUT', 'CACHE_IN', 'FORWARD', 'VERIFY', 'VERIFY_AT']
CDR_STR = []
for cdr in CDR_TYPE:
    for str_type in STR:
        CDR_STR.append(cdr+'|'+str_type)
DAY_MIN     = 31
DAY_MAX     = 0
        
        
def getTimeDate(line):
    '''13-10-16 23:59:27.11'''
    global DAY_MIN, DAY_MAX 
    line = line.split(' ')[0]
    line = line.split('-')
#    month = line[1]
    day = line[0]
    dayInt = int(day)
    if dayInt < DAY_MIN:
        DAY_MIN = dayInt
    if dayInt > DAY_MAX:
        DAY_MAX = dayInt
    return day


class Host2CDR:
    def __init__(self, path, oPath, hostList):
        """constructor"""
        self.__path     = path
        self.__oPath    = oPath
        self.__gzFile   = None
        self.__hostList = hostList
        self.df         = None
    
    
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
            
            
    def preScan(self):
        if not self.__path:
            return
        gzfile = self.readGZFile()
        mapTmp = cl.defaultdict(dict)
        gzlist = gzfile.split('\n')
        for line in gzlist:
            '''split by BEL'''
            tmpLinesByBel = line.split("")
            if len(tmpLinesByBel) >= PA_MAX_BEL:
                time = tmpLinesByBel[0].split(',')[0]
                day = getTimeDate(time)
                host = tmpLinesByBel[1]
#                uri = tmpLinesByBel[3]
                fdetail = tmpLinesByBel[4].split(',')
                if len(fdetail)<=10:
                    continue
                cdrType = fdetail[10]
                destIP = fdetail[6]
                sessionR = fdetail[1]
                k = cdrType+'|'+sessionR
                for tmpHost in self.__hostList:
                    mObj = PA_IP_PAT.match(tmpHost)
                    if mObj:
                        '''IP'''
                        if (host.find(tmpHost)!=-1) or (destIP.find(tmpHost)!=-1):
                            if not mapTmp[day].get(host, False):
                                mapTmp[day][host] = {}
                            if not mapTmp[day][host].get(destIP, False):
                                mapTmp[day][host][destIP] = {}
                            mapTmp[day][host][destIP][k] = mapTmp[day][host][destIP].get(k, 0)+1
                    else:
                        if host.find(tmpHost)!=-1:
                            if not mapTmp[day].get(host, False):
                                mapTmp[day][host] = {}
                            if not mapTmp[day][host].get(destIP, False):
                                mapTmp[day][host][destIP] = {}
                            mapTmp[day][host][destIP][k] = mapTmp[day][host][destIP].get(k, 0)+1
        return mapTmp
        
        
    def transMap2List(self, mapTmp):
        listForPandas = []
        if mapTmp:
            for day, dayInfo in mapTmp.iteritems():
                for host, destInfo in dayInfo.iteritems():
                    for destIP, info in destInfo.iteritems():                    
                        tmp = {}
                        tmp['Day']          = day
                        tmp['Host']         = host
                        tmp['DestIP']       = destIP
                        for btype in CDR_STR:
                            tmp[btype] = info.get(btype, 0)
                        listForPandas.append(tmp)
        return listForPandas
        
        
    def preData(self):
        mapTmp = self.preScan()
        listForPandas = self.transMap2List(mapTmp)
        self.df = pd.DataFrame(listForPandas)
        
        
    def __add__(self, other):
        if not other.df.empty:
            df = pd.concat([self.df, other.df], ignore_index=True)
            df = df.groupby(['Host', 'Day', 'DestIP']).agg(np.sum)
            df = df.reset_index()
            self.df = df
        return self
        
    
    def analyze(self):
        """real analyses core"""
        wPath = self.getWritePath()
        xlsPath = os.path.join(wPath, 'hostCDR2dDay.xls')
        #self.df = self.df.sort_values(by=['OptReq', 'FileSize'], ascending=False)
        with pd.ExcelWriter(xlsPath) as writer:
            self.df.to_excel(writer, sheet_name='Status')
            writer.save()
        
        
    def getWritePath(self):
        realOutPath = None
        if self.__oPath:
            realOutPath = self.__oPath
        else:
            realOutPath = os.path.dirname(self.__path)
        if not os.path.exists(realOutPath):
            os.makedirs(realOutPath)
        return realOutPath
        

        
def getHostListByFile(fpath):
    if os.path.isfile(fpath):
        f = open(fpath, 'r')
        r = []
        for line in f.readlines():
            line = line.strip()
            r.append(line)
        f.close()
        return r

    
def main():
#    argParser = argparse.ArgumentParser(description="Get cdr status of specific Hosts", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#    argParser.add_argument('--inpath', type=str, dest='ipath', help='the cdr file dictory, can be recursive', required=True)
#    argParser.add_argument('--outpath', type=str, dest='opath', help='the output file dictory')
#    argParser.add_argument('--hostpath', type=str, dest='hostpath', help='the specific hosts file', required=True)
#    args = argParser.parse_args()
#    hostList = getHostListByFile(args.hostpath)
    hostList = getHostListByFile(r'C:\Users\Adryan\Documents\hostList2dday.txt')
    '''analysis'''
    print('Procedure begins..')
    start = time.clock()
    rootDir = r'E:\data2do'
#    rootDir = args.ipath
    AllHandler = Host2CDR(None, r'C:\Users\Adryan\Documents', hostList)
#    AllHandler = Host2CDR(None, args.opath, hostList)
    AllHandler.preData()
    index = 0
    for path, dirname, filelist in os.walk(rootDir):
        for filename in filelist:
            f = os.path.join(path, filename)
            if f.endswith('gz'):
                index = index + 1
                print('Handled '+str(index)+' file: '+f)
                Handler = Host2CDR(f, None, hostList)
                Handler.preData()
                AllHandler += Handler
    AllHandler.analyze()
    end = time.clock()
    print('Procedure ended..')
    print("All time cost: %f s" % (end - start))
    

if __name__=='__main__':
    main()