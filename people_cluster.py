#coding:GBK

import numpy as np
import csv

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import preprocessing
import copy
import math
from pre_process import *

_data_dir_ = "t_bsfwt_stard.csv"
group_num = 5
feature_end = 9
filter_num = '24419'
#weight = [0.17, 0.23, 0.28, 0.11, 0.08, 0.06 ,0.05]
weight = [0.082719141,0.061460086,0.050417881,0.127127107,0.174003716,0.230284059,0.27398801] #广州转置

inverse = [0,2,3]

def compute_kpi(deal_data):
    kpi_result = []

    for item in deal_data:
        tmp = 0
        for i in range(0,len(weight)):
            if i in inverse:
                tmp += (1-item[i]) * weight[i]
            else:
                tmp += item[i] * weight[i]
        kpi_result.append(tmp)

    kpi_result = kpi_to_standard(kpi_result,chaxishu=0.2)

    return kpi_result


def init_data_get(dir,fileter = filter_num):
    file = open(dir)
    reader = csv.reader(file)
    init_data = []
    count = 0
    for item in reader:
        tmp = []

        if count!=0 and fileter != None and fileter != item[1][0:5]:
            count+=1
            continue
        tmp.append(item[1])
        for i in range(2,feature_end):
            tmp.append(item[i])
        init_data.append(tmp)
        count += 1
    return init_data

#筛选获取列。数值化
def data_read(dir,fileter = filter_num):
    file = open(dir)
    file.readline()
    reader = csv.reader(file)
    train = []
    count = 0
    for item in reader:
        tmp = []

        if fileter!=None and fileter!=item[1][0:5]:
            continue
        for i in range(2, feature_end):
            if item[i] !='':
                tmp.append(float(item[i]))
            else:
                tmp.append(0.0)

        count+=1
        train.append(tmp)
    file.close()
    return np.array(train)

#预处理
def data_process():
    max_min_scaler = preprocessing.MinMaxScaler()
    return max_min_scaler

def kpi_process(data):
    chaoshi = minus_by_one(data.T[0].T,0.01)
    chuangkou = Zscore(data.T[1].T)
    banli = max_min_process(data.T[2].T, 1)
    #banli = Zscore(banli)
    dengdai  = max_min_process(data.T[3].T, 1)
    #dengdai = Zscore(dengdai)
    riy = Zscore(data.T[4].T)
    rir = Zscore(data.T[5].T)
    zk = Zscore(data.T[6].T)

    new_data = np.c_[chaoshi,chuangkou,banli,dengdai,riy,rir,zk]
    return new_data


def tans_data(now_data):
    for item in now_data:
        for i in range(0, len(weight)):
            if i in inverse:
                item[i]= (1 - item[i]) * math.sqrt(weight[i])
            else:
                item[i] = item[i] * math.sqrt(weight[i])
    return now_data


def tans_data_inv(now_data):
    for item in now_data:
        for i in range(0, len(weight)):
            if i in inverse:
                item[i]= (1 - item[i]) / math.sqrt(weight[i])
            else:
                item[i] = item[i] / math.sqrt(weight[i])
    return now_data
#聚类
def cluster_1(scale_data):
    clt = KMeans(init='k-means++', n_clusters=group_num, n_init=10)
    clt.fit(scale_data)

    return clt

#max&min
def max_min(data,clt):
    min_x = []
    max_x = []
    sum_num = [0 for i in range(0,group_num)]
    for i in range(0, group_num):
        tmp_mi = []
        tmp_ma = []
        for j in range(0, len(data[0])):
            tmp_mi.append(1000000.0)
            tmp_ma.append(0.0)
        min_x.append(tmp_mi)
        max_x.append(tmp_ma)

    count = 0
    for item in clt.labels_:
        for j in range(0, len(data[0])):
            if min_x[item][j] > data[count][j]:
                min_x[item][j] = data[count][j]
            if max_x[item][j] < data[count][j]:
                max_x[item][j] = data[count][j]
        sum_num[item] += 1

        count += 1
    return (max_x, min_x, sum_num)

#输出
def data_out(clt,center,fileter = filter_num):
    init_data = init_data_get(_data_dir_)
    all_group = [ [] for i in range(0,group_num)]
    read_data_now = data_read(_data_dir_)

    kpidata = kpi_process(read_data_now)

    max_x, min_x, sum_num = max_min(read_data_now,clt)
    kpi = compute_kpi(kpidata)

    sum_kpi = [[] for i in range(0,group_num)]
    for i in range(0,len(clt.labels_)):
        all_group[clt.labels_[i]].append(init_data[i+1]+[clt.labels_[i]]+[kpi[i]]+kpidata.tolist()[i])
        sum_kpi[clt.labels_[i]].append(kpi[i])
    writer = csv.writer(open(str(group_num) +"result.csv", "wb"))
    writer.writerow(init_data[0]+['label','kpi'])
    for i in range(0,group_num):
        for j in range(0,len(all_group[i])):
            writer.writerow(all_group[i][j])
        writer.writerow([])

        writer.writerow(['min']+min_x[i])
        writer.writerow(['center']+list(center[i])+[' ']+[np.average(sum_kpi[i])])
        writer.writerow(['max'] + max_x[i])
        writer.writerow(['total'] + [sum_num[i]])
        writer.writerow([])


for i in range(3,6):
    group_num = i
    feature_end = 13
    filter_num = '24419'
    data = data_read(_data_dir_)
    #pca = PCA(n_components=group_num).fit(data)

    max_min_scaler = data_process()
    data_max_min = max_min_scaler.fit_transform(data)
    tans_data(data_max_min)

    clt = cluster_1(data_max_min)

    center = clt.cluster_centers_
    tans_data_inv(center)
    center = max_min_scaler.inverse_transform(center)


    print center

    data_out(clt,center)
