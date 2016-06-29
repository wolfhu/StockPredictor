# coding: utf-8

import MySQLdb
import pandas as pd
import itertools
from random import shuffle
from Queue import Queue
import threading
import os.path
import sys
sys.path.append("../")
from param_config import config


# 打开数据库连接，执行sql进行数据库读取
def sql_query(sql_list):
    stocks_list = []
    db = MySQLdb.connect(config.ip, config.user, config.passwd, config.database, config.port)
    for sql in sql_list:
        stock_data = pd.read_sql_query(sql, con=db)
        stock_data = stock_data.dropna()
        stocks_list.append(stock_data)
    db.close()

    stocks_data = pd.concat(stocks_list)
    return stocks_data

class StockConsumer(threading.Thread):
    def __init__(self, queue, lock, parent_train_result, parent_test_result):
        super(StockConsumer, self).__init__()
        self.queue = queue
        self.lock = lock
        self.parent_train_result = parent_train_result
        self.parent_test_result = parent_test_result

        self.train_result = []
        self.test_result = []

    def run(self):
        single_stock_data = None
        while True:
            self.lock.acquire()
            if not self.queue.empty():
                single_stock_data = self.queue.get()
                self.lock.release()
            else:
                self.lock.release()
                break

            #有的没有数据，过滤掉
            if len(single_stock_data) < config.before + config.after + config.interval * config.left4test:
                print '{} does not have enough data'.format(single_stock_data.ix[0, 'symbol'])
                continue

            features_and_result_train = []
            features_and_result_test = []

            # 循环添加
            for ix in xrange(0, len(single_stock_data) - config.before - config.after, config.interval):
                after_price = single_stock_data.ix[ix+config.before+config.after-1, 'close'] / single_stock_data.ix[ix+config.before-1, 'close'] - 1

                feature_df = single_stock_data.ix[ix:ix + config.before - 1, config.cols_dimension]
                features = reduce(lambda x, y: x + y, feature_df.values.tolist())
                features.extend([single_stock_data.loc[ix, 'date'], single_stock_data.loc[0, 'symbol'], after_price])

                if after_price <= -config.threhold_up_down:
                    features.append(0)
                    features_and_result_train.append(features)
                elif after_price >= config.threhold_up_down:
                    features.append(1)
                    features_and_result_train.append(features)
                elif config.n_classes == 3 and after_price < config.fluctuation_threhold and after_price > -config.fluctuation_threhold:
                    features.append(2)
                    features_and_result_train.append(features)

            features_and_result_test = features_and_result_train[-config.left4test:]
            features_and_result_train = features_and_result_train[:-config.left4test]
            # 有的没有数据，过滤掉
            if len(features_and_result_train) == 0:
                print '{} does not have enough data'.format(single_stock_data.ix[0, 'symbol'])
                continue
            else:
                print single_stock_data.loc[0, 'symbol'], len(single_stock_data), len(features_and_result_train), len(features_and_result_test)
            self.train_result.extend(features_and_result_train)
            self.test_result.append(features_and_result_test)
            print "A stock has finished and {} stocks left".format(self.queue.qsize())

        self.lock.acquire()
        self.parent_train_result.extend(self.train_result)
        self.parent_test_result.extend(self.test_result)
        self.lock.release()

def query(sql_list):
    '''
    if os.path.isfile(config.all_file_path) :
        train_data = pd.read_csv(config.all_file_path)
        return train_data
    '''

    stocks_data = sql_query(sql_list)

    #将每只股票划分开加入queue
    stocks_data_queue = Queue()
    for tmp, single_stock_data in stocks_data.groupby('symbol'):
        single_stock_data = single_stock_data.sort(['date'])
        single_stock_data.index = range(len(single_stock_data))
        stocks_data_queue.put(single_stock_data)

    train_result = []
    test_result = []
    config.thread_list = []
    queueLock = threading.Lock()
    for ix in xrange(config.thread_size):
        stock_consumer = StockConsumer(stocks_data_queue,
                                       queueLock, train_result, test_result)
        config.thread_list.append(stock_consumer)
    for stock_consumer in config.thread_list:
        stock_consumer.start()
    for stock_consumer in config.thread_list:
        stock_consumer.join()

    feature_cols = []
    for ix in xrange(config.before):
        for col in config.cols_dimension:
            feature_cols.append(col + '_' + str(ix))
    feature_cols.extend(['date', 'symbol', 'value', 'label'])
    train_data = pd.DataFrame(train_result, columns=feature_cols)

    #标准化test_result

    test_original_data = [[x[i] for x in test_result] for i in xrange(config.left4test)]
    test_data_list = []
    for test_original_datum in test_original_data:
        test_data_list.append(pd.DataFrame(test_original_datum, columns=feature_cols))

    #train_data.to_csv(config.original_file, index=False)
    return train_data, test_data_list


