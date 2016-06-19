# coding: utf-8

import sys
sys.path.append("../")
from param_config import config

def _filter_apply(row, up_threhold, down_threhold):
    row_cols = row.columns
    for ix in xrange(1, len(row_cols)):
        ratio = row[row_cols[ix]] / row[row_cols[ix - 1]]
        if  ratio > up_threhold and ratio < down_threhold:
            return False
    return True

def _filter_large_up_down(df_data, cols, up_threhold, down_threhold):
    df = df_data.ix[:, cols]
    if config.is_dis:
        df['not_drop'] = df.apply(lambda x: max(x) < up_threhold and min(x) > down_threhold, axis=1)
    else:
        df['not_drop'] = df.apply(lambda x: _filter_apply(x, up_threhold, down_threhold), axis=1)

    df_data = df_data[df.not_drop]
    return df_data

'''
过滤包含急升急跌的时间点的数据
因为觉得这很有可能是因为外部刺激引起的，暂时不考虑这些外部刺激，
而且这些状况的样本数量比较少，即使想抓取也比较难（后期可以采用adaboost或加权重的方式）
但目前采取的最基本的过滤策略是将特征中隔天收盘价涨跌9%以上的数据去掉
'''
def _filter_close_large_up_down(df_data):
    cols = [x for x in df_data.columns if 'close' in x]
    return _filter_large_up_down(df_data, cols, config.close_up_threhold, config.close_down_threhold)

'''
暂时将成交额度非常大的当作噪声
一是因为同过滤收盘价同样的道理，
二是数据归一化的时候有影响
'''
def _filter_amount_large_up_down(df_data):
    cols = [x for x in df_data.columns if 'close' in x]
    return _filter_large_up_down(df_data, cols, config.amount_up_threhold, config.amount_down_threhold)


def filtering(df_data):
    df_data = _filter_close_large_up_down(df_data)
    if 'amount' in config.cols_dimension:
        df_data = _filter_amount_large_up_down(df_data)

    return df_data
