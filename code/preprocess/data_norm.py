# coding: utf-8

import sys
import cPickle
import numpy as np
import pandas as pd

sys.path.append("../")
from param_config import config

# z-score: y = (x-mean)/std
def _z_score(df_data):
    df = df_data.drop(['date', 'symbol', 'value', 'label'], axis=1)
    cols = list(df.columns)
    df_data['mean'] = df.apply(np.mean, axis=1)
    df_data['std'] = df.apply(np.std, axis=1)

    df_data.ix[:, cols] = df_data.apply((lambda row: (row[cols] - row['mean']) / row['std']), axis=1)
    del df_data['mean']
    del df_data['std']

#min-max: y = (x-min)/(max-min)
def _min_max(df_data):
    df = df_data.drop(['date', 'symbol', 'value', 'label'], axis=1)
    cols = list(df.columns)
    df_data['min'] = df.apply(min, axis=1)
    df_data['max'] = df.apply(max, axis=1)

    df_data.ix[:, cols] = df_data.apply((lambda row: (row[cols] - row['min'])/(row['max'] - row['min'])), axis=1)
    del df_data['min']
    del df_data['max']

#decimal-scaling: y = x/10^j, where j is the smallest integer such that max(|y|)<1
def _decimal_scaling(df_data):
    df = df_data.drop(['date', 'symbol', 'value', 'label'], axis=1)
    cols = list(df.columns)
    df_data['j'] = df.apply(lambda x: -(int)(np.log10(1 / max(x))), axis=1)

    df_data.ix[:, cols] = df_data.apply((lambda row: row[cols] / (row[cols] - row['j'])), axis=1)
    del df_data['j']

'''使用简单缩放的方法，分类置为[0,1]'''


def data_norm(df_data):
    norm_strategy = config.norm_strategy
    if norm_strategy == 'z-score':
        _z_score(df_data)
    elif norm_strategy == 'min-max':
        _min_max(df_data)
    elif norm_strategy == 'decimal-scaling':
        _decimal_scaling(df_data)