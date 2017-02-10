# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 09:40:47 2016

@Author : Adryan
@MailTo : lihongyuan@hzgit.com
@Version: 1.0
"""

import time
import os
#os.environ['THEANO_FLAGS'] = 'exception_verbosity=high'
import os.path

import numpy as np
import pandas as pd
import keras.models as km
import keras.layers as kl
import keras.optimizers as ko

MAX_STEP = 2000
MAX_DIM  = 4
LAT_MIN  = 30 
LAT_MAX  = 32
LON_MIN  = 103
LON_MAX  = 105
LAT_AVE  = 30.65191963
LON_AVE  = 104.059916478

#def calc_distance(lon1, lat1, lon2, lat2):
#    dx = lon1 - lon2
#    dy = lat1 - lat2
#    b = (lat1 + lat2) / 2.0
#    Lx = (dx/57.2958) * 6371004.0* math.cos(b/57.2958)
#    Ly = 6371004.0 * (dy/57.2958)
#    return math.sqrt(Lx * Lx + Ly * Ly)


def timeConvert(timeStr):
    return int(time.mktime(time.strptime(timeStr, r"%Y/%m/%d %H:%M:%S")))
    
    
def getHour(timeStr):
    return time.strptime(timeStr, r"%Y/%m/%d %H:%M:%S").tm_hour
    

def getDay(timeStr):
    return time.strptime(timeStr, r"%Y/%m/%d %H:%M:%S").tm_mday


def build_model_from_file(struct_file, weights_file):
    model = km.model_from_json(open(struct_file, 'r').read())
    model.compile(loss="mse", optimizer='adam')
    model.load_weights(weights_file)
    return model
    

def save_model_to_file(model, struct_file, weights_file):
    # save model structure
    model_struct = model.to_json()
    open(struct_file, 'w').write(model_struct)
    # save model weights
    model.save_weights(weights_file, overwrite=True)
    

def buildLSTMModel(input_size, max_output_seq_len, hidden_size):
    model = km.Sequential()
    layer0 = kl.Masking(mask_value=0, input_shape=(max_output_seq_len, input_size))
    model.add(layer0)
#    print layer0.input_shape, layer0.output_shape
    layer1 = kl.LSTM(input_dim=input_size, output_dim=hidden_size, return_sequences=False)
    model.add(layer1)
#    print layer1.input_shape, layer1.output_shape
    layer2 = kl.Dense(hidden_size, activation='relu')
    model.add(layer2)
#    print layer2.input_shape, layer2.output_shape
    layer3 = kl.RepeatVector(max_output_seq_len)
    model.add(layer3)
#    print layer3.input_shape, layer3.output_shape
    layer4 = kl.LSTM(hidden_size, return_sequences=True)
    model.add(layer4)
#    print layer4.input_shape, layer4.output_shape
    layer5 = kl.TimeDistributed(kl.Dense(output_dim=1, activation="linear"))
    model.add(layer5)
#    print layer5.input_shape, layer5.output_shape
    model.compile(loss='mse', optimizer='adam')
    return model
    

def buildGRUModel(input_size, seq_len, hidden_size):
    model = km.Sequential()
    model.add(kl.GRU(input_dim=input_size, output_dim=hidden_size, return_sequences=False))
    model.add(kl.Dense(hidden_size, activation="relu"))
    model.add(kl.RepeatVector(seq_len))
    model.add(kl.GRU(hidden_size, return_sequences=True))
    model.add(kl.TimeDistributed(kl.Dense(output_dim=input_size, activation="linear")))
    model.compile(loss="mse", optimizer='adam')
    return model
    

def buildAutoEncoder(inputSize, outputSize):
    Encoder = km.Sequential([kl.Dense(outputSize, input_dim=inputSize), kl.Activation('sigmoid')])
    Decoder = km.Sequential([kl.Dense(inputSize, input_dim=outputSize), kl.Activation('sigmoid')])
    AutoEncoder = km.Sequential()
    AutoEncoder.add(Encoder)
    AutoEncoder.add(Decoder)
    AutoEncoder.compile(loss='mse', optimizer='rmsprop')
    return AutoEncoder
    
    
def buildMultiInputLSTM():
    mainInput = kl.Input(shape=(MAX_STEP, 2), name='mainInput')
    mainInput1 = kl.Masking(mask_value=0)(mainInput)
    auxInput1 = kl.Input(shape=(MAX_STEP,), name='auxInput1')
    auxInput2 = kl.Input(shape=(MAX_STEP,), name='auxInput2')
    auxInput3 = kl.Input(shape=(MAX_STEP,), name='auxInput3')
    auxOutput1 = kl.Embedding(output_dim=2, input_dim=MAX_STEP, mask_zero=True)(auxInput1)
    auxOutput2 = kl.Embedding(output_dim=2, input_dim=MAX_STEP, mask_zero=True)(auxInput2)
    auxOutput3 = kl.Embedding(output_dim=2, input_dim=MAX_STEP, mask_zero=True)(auxInput3)
    out = kl.merge([mainInput1, auxOutput1, auxOutput2, auxOutput3], mode='concat')
    lstmOut = kl.LSTM(128)(out)
    mainOutput = kl.Dense(MAX_STEP, activation='softmax', name='mainOutput')(lstmOut)
    model = km.Model(input=[mainInput, auxInput1, auxInput2, auxInput3], output=[mainOutput])
    model.compile(optimizer='rmsprop', loss='binary_crossentropy')
    return model
    

def pdTrain(path, model):
    df = pd.read_csv(path, header=None, names=['taxi_id', 'lat', 'lon', 'busy', 'time'])
#    latAverage, latMin, latMax = df['lat'].mean(), df['lat'].min(), df['lat'].max()
#    lonAverage, lonMin, lonMax = df['lon'].mean(), df['lon'].min(), df['lon'].max(), 
#    print latAverage, latMin, latMax, lonAverage, lonMin, lonMax
    idlist = list(set(df['taxi_id']))
    mInputList      = []
    axInputList1    = []
    axInputList2    = []
    axInputList3    = []
    outPutList      = []
    for taxiId in idlist:
        if taxiId<=100:
            tmpdf = df[(df.taxi_id==taxiId)]
            tmpdf = tmpdf.sort_values(by=['time'], ascending=True).reset_index(drop=True)
            tmpdf['edge'] = abs(tmpdf['busy']-tmpdf['busy'].shift())
            signdf = tmpdf[(tmpdf.edge==1.0)]
            indexList = list(signdf.index)
            length = len(indexList)
            if length>1:
                for i in range(0, length, 2):
                    start = i
                    end = i+1
                    if (start<length) and (end<length):
                        start = indexList[start]
                        end = indexList[end]+1
                        fdf = tmpdf[start:end]
                        fdf = fdf.drop(['busy', 'edge'], axis=1)
                        fdf['timeSs'] = map(timeConvert, fdf['time'])
                        fdf['period'] = map(getHour, fdf['time'])
                        fdf['day'] = map(getDay, fdf['time'])
                        fdf['gap'] = fdf['timeSs']-fdf['timeSs'].shift()
                        fdf = fdf.drop(['timeSs', 'time'], axis=1).fillna(0)
                        y = fdf['gap'].reset_index(drop=True).tolist()
                        x = fdf.drop(['gap'], axis=1).reset_index(drop=True)
                        x['lat'] = (x['lat'] - LAT_AVE)/float(LAT_MAX - LAT_MIN)*100
                        x['lon'] = (x['lon'] - LON_AVE)/float(LON_MAX - LON_MIN)*100
                        m = list(x['taxi_id'])
                        n = list(x['period'])
                        l = list(x['day'])
                        x = x[['lat', 'lon']]
                        if x.shape[0] > MAX_STEP:
                            continue
                        t = np.zeros(shape=(MAX_STEP, 2))
                        for ind, row in x.iterrows():
                            j = 0
                            for col_name in x.columns:
                                t[ind, j] = row[col_name]
                                j+=1
                        t1 = np.zeros(shape=(MAX_STEP))
                        for key1, value1 in enumerate(m):
                            t1[key1] = value1
                        t2 = np.zeros(shape=(MAX_STEP))
                        for key2, value2 in enumerate(n):
                            t2[key2] = value2
                        t3 = np.zeros(shape=(MAX_STEP))
                        for key3, value3 in enumerate(l):
                            t3[key3] = value3
                        v = np.zeros(shape=(MAX_STEP))
                        for key3, value3 in enumerate(y):
                            v[key3] = value3
    #                    loss = model.train_on_batch({'mainInput':t, 'auxInput1':t1, 'auxInput2':t2}, {'mainOutput':v})
    #                    print loss
                        mInputList.append(t)
                        axInputList1.append(t1)
                        axInputList2.append(t2)
                        axInputList3.append(t3)
                        outPutList.append(v)
    model.fit([np.asarray(mInputList), np.asarray(axInputList1), np.asarray(axInputList2), np.asarray(axInputList3)], [np.asarray(outPutList)], batch_size=64, nb_epoch=5)

    
def pdGenerator(path, batch_size=64):
    df_reader = pd.read_csv(path, header=None, names=['taxi_id', 'lat', 'lon', 'busy', 'time'], chunksize=10240)
    for df in df_reader:
        mInputList      = []
        axInputList1    = []
        axInputList2    = []
        axInputList3    = []
        outPutList      = []
        idlist = list(set(df['taxi_id']))
        for taxiId in idlist:
            tmpdf = df[(df.taxi_id==taxiId)]
            tmpdf = tmpdf.sort_values(by=['time'], ascending=True).reset_index(drop=True)
            tmpdf['edge'] = abs(tmpdf['busy']-tmpdf['busy'].shift())
            signdf = tmpdf[(tmpdf.edge==1.0)]
            indexList = list(signdf.index)
            length = len(indexList)
            if length>1:
                for i in range(0, length, 2):
                    start = i
                    end = i+1
                    if (start<length) and (end<length):
                        start = indexList[start]
                        end = indexList[end]+1
                        fdf = tmpdf[start:end]
                        fdf = fdf.drop(['busy', 'edge'], axis=1)
                        fdf['timeSs'] = map(timeConvert, fdf['time'])
                        fdf['period'] = map(getHour, fdf['time'])
                        fdf['day'] = map(getDay, fdf['time'])
                        fdf['gap'] = fdf['timeSs']-fdf['timeSs'].shift()
                        fdf = fdf.drop(['timeSs', 'time'], axis=1).fillna(0)
                        y = fdf['gap'].reset_index(drop=True).tolist()
                        x = fdf.drop(['gap'], axis=1).reset_index(drop=True)
                        x['lat'] = (x['lat'] - LAT_AVE)/float(LAT_MAX - LAT_MIN)*100
                        x['lon'] = (x['lon'] - LON_AVE)/float(LON_MAX - LON_MIN)*100
                        m = list(x['taxi_id'])
                        n = list(x['period'])
                        l = list(x['day'])
                        x = x[['lat', 'lon']]
                        if x.shape[0] > MAX_STEP:
                            continue
                        t = np.zeros(shape=(MAX_STEP, 2))
                        for ind, row in x.iterrows():
                            j = 0
                            for col_name in x.columns:
                                t[ind, j] = row[col_name]
                                j+=1
                        t1 = np.zeros(shape=(MAX_STEP))
                        for key1, value1 in enumerate(m):
                            t1[key1] = value1
                        t2 = np.zeros(shape=(MAX_STEP))
                        for key2, value2 in enumerate(n):
                            t2[key2] = value2
                        t3 = np.zeros(shape=(MAX_STEP))
                        for key3, value3 in enumerate(l):
                            t3[key3] = value3
                        v = np.zeros(shape=(MAX_STEP))
                        for key3, value3 in enumerate(y):
                            v[key3] = value3
                        mInputList.append(t)
                        axInputList1.append(t1)
                        axInputList2.append(t2)
                        axInputList3.append(t3)
                        outPutList.append(v)
                        if len(outPutList) >= batch_size:
                            yield(([np.asarray(mInputList), np.asarray(axInputList1), np.asarray(axInputList2), np.asarray(axInputList3)], [np.asarray(outPutList)]))
                            mInputList      = []
                            axInputList1    = []
                            axInputList2    = []
                            axInputList3    = []
                            outPutList      = []

                            
def pdTrainGenerator(path, model):
    #do some statics
    f = open(path, 'r')
    cnt = 0
    sample_map = {}
    while True:
        line = f.readline()
        if not line:
            break
        tmp_line = line.split(',')
        taxiid = tmp_line[0]
        busy = tmp_line[3]
        if sample_map.get(taxiid, False) and sample_map[taxiid] != busy:
            cnt+=1
        sample_map[taxiid] = busy
    cnt/=2
    print cnt
    model.fit_generator(pdGenerator(path), samples_per_epoch=cnt, nb_epoch=5)
    
def main():
#    model = buildLSTMModel(MAX_DIM, MAX_STEP, 100)
    model = buildMultiInputLSTM()
    rootDir = r'F:\data2do'
    for path, dirname, filelist in os.walk(rootDir):
        for filename in filelist:
            f = os.path.join(path, filename)
            if f.endswith('train.txt'):
#                pdTrain(f, model)
                pdTrainGenerator(f, model)
                break
    save_model_to_file(model, 'model_s', 'model_w')
                

if __name__=='__main__':
    main()