# -*- coding: UTF-8 -*-
from pandas.io import sql as sqlutil
import MySQLdb
import pandas as pd
import numpy as np
import talib
import xlrd as xl
from datetime import datetime, timedelta
from Queue import Queue
import threading

import sys
sys.path.append("../")
from param_config import config

class StockConsumer(threading.Thread):
    def __init__(self, queue, read_lock, read_db, write_lock, write_db, start_date):
        super(StockConsumer, self).__init__()
        self.queue = queue
        self.read_lock = read_lock
        self.read_db = read_db
        self.write_lock = write_lock
        self.write_db = write_db
        self.start_date = start_date

    def run(self):
        sql = None
        interval = timedelta(weeks=8)

        while True:
            self.read_lock.acquire()
            if not self.queue.empty():
                sql = self.queue.get()
                stock_df = pd.read_sql_query(sql, con=self.read_db)
                self.read_lock.release()
            else:
                self.read_lock.release()
                break

            if len(stock_df) == 0:
                continue

            '''在这期间上市的股票，上市初期两个月去掉'''
            stock_df['date'] = pd.to_datetime(stock_df['date'], format='%Y%m%d')
            stock_begin_date = stock_df.ix[0, 'date']
            print type(stock_begin_date), type(self.start_date)
            if stock_begin_date > self.start_date:
                filter_date = stock_begin_date + interval
                stock_df[stock_df.date < filter_date] = np.nan
            stock_df = stock_df.dropna()
            stock_df.index = range(len(stock_df))

            #计算指标
            calculate_indicator(stock_df)

            #写入数据库
            self.write_lock.acquire()
            stock_df.to_sql(name='tech_ind', con=self.write_db, flavor='mysql', if_exists='append', index=False, chunksize=300)
            self.write_lock.release()
            stock_df = None

            print "A stock has finished and {} stocks left".format(self.queue.qsize())


def calculate_indicator(stock_df):
    periods = [3, 5, 10, 20, 30, 60]
    # MA
    for period in periods:
        stock_df['MA' + str(period)] = talib.MA(stock_df['close'].values, timeperiod=period)

    # EMA
    periods = [3, 5, 10, 20, 30, 60]
    for period in periods:
        stock_df['EMA' + str(period)] = talib.EMA(stock_df['close'].values, timeperiod=period)

    # AMTMA
    periods = [5, 10, 20]
    for period in periods:
        stock_df['AMTMA' + str(period)] = talib.MA(stock_df['amount'].values, timeperiod=period)

    # ATR
    periods = [5, 10, 20]
    for period in periods:
        stock_df['ATR' + str(period)] = talib.ATR(stock_df['high'].values, stock_df['low'].values,
                                                  stock_df['close'].values, timeperiod=period)

    # ADX
    period = 14
    stock_df['ADX' + str(period)] = talib.ADX(stock_df['high'].values, stock_df['low'].values,
                                              stock_df['close'].values, timeperiod=period)

    # MACD
    stock_df['MACD_DIFF'], stock_df['MACD_DEA'], stock_df['MACD_HIST'] = talib.MACD(
        stock_df['close'].values, fastperiod=12, slowperiod=26, signalperiod=9)

    # CCI
    period = 14
    stock_df['CCI' + str(period)] = talib.CCI(stock_df['high'].values, stock_df['low'].values,
                                              stock_df['close'].values, timeperiod=period)

    # MFI
    period = 14
    stock_df['MFI' + str(period)] = talib.MFI(stock_df['high'].values, stock_df['low'].values,
                                              stock_df['close'].values, stock_df['volume'].values,
                                              timeperiod=period)

    # ROCP
    periods = [5, 10, 20]
    for period in periods:
        stock_df['ROCP' + str(period)] = talib.ROCP(stock_df['close'].values, timeperiod=period)

def main():
    # 范围内的股票
    codes = xl.open_workbook(config.code_file).sheets()[0]

    # 生成sql
    sqls = Queue()
    start_date = '20070102'
    end_date = '20160622'
    sql_prefix = "SELECT symbol, date, open, close, high, low, volume, amount  FROM "
    sql_middle = " where amount != 0 and symbol = "
    sql_post = " and date between '" + start_date + "' and '" + end_date + "' order by date asc"
    for ix in range(codes.nrows):
        code = codes.row_values(ix)[0]
        table_type = codes.row_values(ix)[5]
        if table_type == 105:
            table_name = 'sz_kline_day_complexbeforeright'
        if table_type == 101:
            table_name = 'sh_kline_day_complexbeforeright'

        sql = sql_prefix + table_name + sql_middle + code + sql_post
        sqls.put(sql)

    #启动线程
    read_db = MySQLdb.connect("yf-cbg-fb-gushitong14.yf01.baidu.com", "jingong", "jingongpw", "jingong_work", 8081)
    write_db = MySQLdb.connect("10.100.47.147", "root", "rootsafe", "Tech_Indicators", 8080)
    read_lock = threading.Lock()
    write_lock = threading.Lock()
    start_date = pd.to_datetime(start_date, format='%Y%m%d')
    for ix in xrange(config.thread_size):
        stock_consumer = StockConsumer(sqls, read_lock, read_db, write_lock, write_db, start_date)
        stock_consumer.start()
        stock_consumer.join()

    read_db.close()
    write_db.close()

if __name__ == '__main__':
    main()
    print "Done"