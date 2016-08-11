import math
import numpy as np
from sklearn import preprocessing

#one column
def Zscore(vector):
    now = np.array(vector)
    now = preprocessing.scale(now)
    for i in range(0, now.size):
        if now[i] < -2:
            now[i] = 0.5
            continue
        if now[i] >= -2 and now[i]<-1:
            now[i] = 0.6
            continue
        if now[i] >=-1 and now[i]<0:
            now[i] = 0.7
            continue
        if now[i] >= 0 and now[i] < 1:
            now[i] = 0.8
            continue
        if now[i] >= 1 and now[i] < 2:
            now[i] = 0.9
            continue
        if now[i] >= 2:
            now[i] = 1
            continue
    return now

def max_min_process(vector,reverse = 0):
    now = np.array(vector)
    max1 = np.max(vector)
    min1 = np.min(vector)

    dif = max1-min1
    if reverse == 0:
        for i in  range(0,now.size):
            now[i] = (now[i] - min1)/dif
    if reverse ==1:
        for i in range(0, now.size):
            now[i] = (max1 - now[i]) / dif
    return now

def minus_by_one(vector ,cf = 1 ):
    now = 1-(np.array(vector))*cf
    return now

def kpi_to_standard(kpi_result,aver_score = 85,chaxishu = 0):
    standard_scaler = preprocessing.StandardScaler()
    kpi_result = standard_scaler.fit_transform(kpi_result)

    for i in range(0,len(kpi_result)):
        kpi_result[i] = kpi_result[i]*(standard_scaler.scale_*(100-aver_score)/(1-standard_scaler.mean_))*(1+chaxishu)+aver_score
        if kpi_result[i]>100:
            kpi_result[i] = 100
        kpi_result[i] = round(kpi_result[i],1)

    return kpi_result

