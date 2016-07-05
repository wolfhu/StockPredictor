# coding: utf-8

import numpy as np
import pandas as pd

from indicator_acquire import query
from indicator_acquire import sql_query
from filtering import filtering
from data_norm import data_norm
from param_config import config
from data_correction import get_most_corr_code
import os
import xlrd as xl
from sklearn.cluster import DBSCAN

#######################
## load data ##
#######################
#拿到当前中证800成分股
corr_df = pd.read_csv(config.corr_file, dtype={'index':str})
corr_df = corr_df.set_index('index')

codes = xl.open_workbook('./zz800_all_new.xls').sheets()[0]
valid_codes = set([codes.row_values(ix)[3] for ix in xrange(codes.nrows) if codes.row_values(ix)[6] == '\\N' ])
index = set(corr_df.index)
index.difference_update(valid_codes)

corr_df = corr_df.drop(list(index), axis=0)
corr_df = corr_df.drop(list(index), axis=1)

corr_df = 1.001 - corr_df
corr_df = corr_df.fillna(0.0)

db = DBSCAN(eps=0.4, min_samples=40, metric='precomputed').fit(corr_df)

core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
real_labels = [x for x in db.labels_ if x != -1]
n_clusters = len(set(real_labels))
cluster_data_num = len(real_labels)

print('Number of core_samples: %d' % len(set(db.core_sample_indices_ )))
print('Estimated number of clusters: %d' % n_clusters)
print('Estimated number of cluster data: %d' % cluster_data_num)

all_stocks = corr_df.index
valid_stocks = [all_stocks[ix] for ix in range(len(db.labels_)) if db.labels_[ix] != -1]
sql_list = config.get_sql_by_code([all_stocks[ix] for ix in db.core_sample_indices_])

train_data, test_data_list = query([sql_list])

#######################
## filter noise ##
#######################
#train_data = filtering(train_data)

#######################
## data norm ##
#######################
train_data = train_data.sort_values(['date'])
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
for name, single_data in train_data.groupby('symbol'):
    tmp_data = single_data.drop(['date', 'symbol', 'value', 'label'], axis=1)
    features = tmp_data.values.tolist()
    values = single_data['value'].tolist()
    symbols = single_data['symbol'].tolist()
    with open('{}{}{}.txt'.format(config.code_file_path, config.file_prefix, name), 'w') as fp:
        for ix in xrange(len(symbols)):
            fp.write('{};0 {};1 {}\n'.format(symbols[ix], " ".join([str(x) for x in features[ix]]), values[ix]))

cmd = "rm -f " + config.test_file_path + "/*"
os.system(cmd)
#跟上面不一样的是label位换为了code
for index in xrange(len(test_data_list)):
    single_data = test_data_list[index]
    tmp_data = single_data.drop(['date', 'symbol', 'value', 'label'], axis=1)
    features = tmp_data.values.tolist()
    values = single_data['value'].tolist()
    codes = single_data['symbol'].tolist()
    with open('{}{}{}.txt'.format(config.test_file_path, config.file_prefix, str(index)), 'w') as fp:
        for ix in xrange(len(codes)):
            fp.write('{};0 {};1 {}\n'.format(codes[ix], " ".join([str(x) for x in features[ix]]), values[ix]))

#shuffle
#data = train_data.ix[np.random.permutation(len(train_data)), :]
train_data = train_data.sort_values(['date'])

cmd = "rm -f " + config.shuffle_file_path + "/*"
os.system(cmd)

tmp_data = train_data.drop(['date', 'symbol', 'value', 'label'], axis=1)
features = tmp_data.values.tolist()
values = train_data['value'].tolist()
symbols = train_data['symbol'].tolist()
with open('{}{}{}.txt'.format(config.shuffle_file_path, config.file_prefix, 0), 'w') as fp:
    for ix in xrange(len(symbols)):
        fp.write('{};0 {};1 {}\n'.format(symbols[ix], " ".join([str(x) for x in features[ix]]), values[ix]))