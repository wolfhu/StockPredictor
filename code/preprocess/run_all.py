# coding: utf-8

import numpy as np

from data_acquire import query
from filtering import filtering
from data_norm import data_norm
from param_config import config
from data_correction import get_most_corr_code
import os

#######################
## load data ##
#######################

stocks_info = []
codes800 = set()
with open(config.code_file, 'r') as f:
    for line in f.readlines()[-800:]:
        phrases = line.split(',')
        code = phrases[3]

        date_str = phrases[5].split()[0].split('/')
        start_date = date_str[0]
        if int(date_str[1]) < 10:
            start_date += '0' + date_str[1]
        else:
            start_date += date_str[1]
        if int(date_str[2]) < 10:
            start_date += '0' + date_str[2]
        else:
            start_date += date_str[2]
        stocks_info.append((code, start_date))

'''
sql_list = []
for stock_info in stocks_info:
    sql = "SELECT time, code, close FROM price_amount_ratio WHERE code = '" + stock_info[0] + "' and time > '" + stock_info[1] +"'"
    sql_list.append(sql)
'''
codes = get_most_corr_code("000001")
codes.append("000001")
sql_list = [config.get_sql_by_code(codes)]

train_data, test_data_list = query(sql_list)

assert len(train_data) != 0

#######################
## filter noise ##
#######################
train_data = filtering(train_data)

#######################
## data norm ##
#######################
train_data.index = range(len(train_data))
data_norm(train_data)
for test_data in test_data_list:
    data_norm(test_data)

#######################
## data store ##
#######################
cmd = "rm -f " + config.code_file_path + "/*"
os.system(cmd)
# 输入符合preprocess格式的文本形式
for name, single_data in train_data.groupby('code'):
    tmp_data = single_data.drop(['code', 'value', 'label'], axis=1)
    features = tmp_data.values.tolist()
    values = single_data['value'].tolist()
    labels = single_data['label'].tolist()
    with open('{}{}{}.txt'.format(config.code_file_path, config.file_prefix, name), 'w') as fp:
        for ix in xrange(len(labels)):
            fp.write('{};0 {};1 {}\n'.format(labels[ix], " ".join([str(x) for x in features[ix]]), values[ix]))

cmd = "rm -f " + config.test_file_path + "/*"
os.system(cmd)
#跟上面不一样的是label位换为了code
for index in xrange(len(test_data_list)):
    single_data = test_data_list[index]
    tmp_data = single_data.drop(['code', 'value', 'label'], axis=1)
    features = tmp_data.values.tolist()
    values = single_data['value'].tolist()
    codes = single_data['code'].tolist()
    with open('{}{}{}.txt'.format(config.test_file_path, config.file_prefix, str(index)), 'w') as fp:
        for ix in xrange(len(codes)):
            fp.write('{};0 {};1 {}\n'.format(codes[ix], " ".join([str(x) for x in features[ix]]), values[ix]))

#shuffle
data = train_data.ix[np.random.permutation(len(train_data)), :]
data.index = range(len(data))

cmd = "rm -f " + config.shuffle_file_path + "/*"
os.system(cmd)

step = len(data) / 11
for ix in xrange(0, len(data), step):
    single_data = data.ix[ix:min(len(data), ix+step-1), :]
    tmp_data = single_data.drop(['code', 'value', 'label'], axis=1)
    features = tmp_data.values.tolist()
    values = single_data['value'].tolist()
    labels = single_data['label'].tolist()
    with open('{}{}{}.txt'.format(config.shuffle_file_path, config.file_prefix, ix), 'w') as fp:
        for ix in xrange(len(labels)):
            fp.write('{};0 {};1 {}\n'.format(labels[ix], " ".join([str(x) for x in features[ix]]), values[ix]))
