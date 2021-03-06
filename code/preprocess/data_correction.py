# coding: utf-8

import sys
import os.path
import MySQLdb
import pandas as pd
import heapq
sys.path.append("../")
from param_config import config
from indicator_acquire import sql_query

def get_data(begin_date, end_date):
    if os.path.isfile(config.corr_file):
        corr_df = pd.read_csv(config.corr_file, dtype={'index':str})
        corr_df = corr_df.set_index('index')
        return corr_df

    stocks_data = sql_query(['select symbol, date, close from tech_ind where date between {} and {}'.format(begin_date, end_date)])

    df = None
    for code, stock_data in stocks_data.groupby('symbol'):
        stock_data = stock_data.set_index('date')
        stock_data = stock_data.drop(['symbol'], axis=1)
        stock_data = stock_data.rename(columns={'close': code})
        if df is not None:
            df = df.join(stock_data)
        else:
            df = stock_data
    corr_df = df.corr(method='pearson')
    corr_df.to_csv(config.corr_file)
    return corr_df

def get_most_corr_code(code, begin_date, end_date):
    corr_df = get_data(begin_date, end_date)
    index = corr_df.sort(code, ascending=False).index
    codes = [x for x in index if corr_df.ix[x, code] > config.simility_threhold]
    print "same as {} is {}".format(code, codes)
    return codes