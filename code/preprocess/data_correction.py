# coding: utf-8

import sys
import os.path
import MySQLdb
import pandas as pd
import heapq
sys.path.append("../")
from param_config import config
from data_acquire import sql_query

def get_data():
    if os.path.isfile(config.corr_file):
        corr_df = pd.read_csv(config.corr_file, dtype={'index':str})
        corr_df = corr_df.set_index('index')
        return corr_df

    stocks_data = sql_query(config.all_sql)

    df = None
    for code, stock_data in stocks_data.groupby('code'):
        # stock_data['label'] = stock_data.apply(classify, axis=1)
        stock_data = stock_data.set_index('time')
        stock_data = stock_data.drop(['code'], axis=1)
        stock_data = stock_data.rename(columns={'close': code})
        if df is not None:
            df = df.join(stock_data)
        else:
            df = stock_data
    corr_df = df.corr(method='pearson')
    corr_df.to_csv(config.corr_file)
    return corr_df

def get_most_corr_code(code):
    corr_df = get_data()
    codes = list(corr_df.sort(code, ascending=False).index[1:config.same_stock_num+1])
    print "same as {} is {}".format(code, codes)
    return codes