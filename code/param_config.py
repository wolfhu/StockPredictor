# coding: utf-8

import os
import numpy as np


############
## Config ##
############
class ParamConfig:
    def __init__(self):

        # data acquire related
        self.stock_code_prefix = ['000', '600']
        self.n_classes = 2
        self.before = 20
        self.after = 5
        self.interval = 1
        self.left4test = 20  #每只股票的最后n个留下来作为测试机，做成n个横截面
        self.threhold_up_down = 0.02
        self.fluctuation_threhold = 0.0
        self.thread_size = 10
        self.is_dis = True
        # cols_dimension = ['open', 'high', 'low', 'close', 'amount']
        self.cols_dimension = ['close']
        self.start_time = '20070101'
        self.end_time = '20151231'
        self.sql = "SELECT * FROM price_amount_ratio WHERE name like '%银行' and time between '20070101' and '20150202'"
        self.all_sql = "SELECT * FROM price_amount_ratio WHERE time between '20070101' and '20150202'"
        self.sql_prefix = "SELECT * FROM price_amount_ratio WHERE code in ("
        self.sql_postfix = ") and time between '" + self.start_time + "' and '" + self.end_time + "'"
        self.code_file = "../../data/CodeMap.xls"
        self.ip = "10.100.47.147"
        self.user = "root"
        self.passwd = "rootsafe"
        self.database = "StockFeatureBase"
        self.port = 8080


        #filter related
        self.close_up_threhold = 1.09
        self.close_down_threhold = 0.91
        self.amount_up_threhold = 9
        self.amount_down_threhold = 0.11

        #norm related
        self.strategy = 'min-max'

        #corr realated
        self.same_stock_num = 30

        #file related
        self.all_file_path = '/home/deepDataBase/800/dis/data/'
        self.code_file_path = '/home/deepDataBase/800/dis/'
        self.test_file_path = '/home/deepDataBase/800/test_dis/'
        self.predict_result_path = '/home/deepDataBase/800/test_dis_result/'
        self.shuffle_file_path = '/home/deepDataBase/800/dis_shuffle/'
        #self.original_file = '/home/deepDataBase/000001same'
        self.corr_file = '/home/xutao/generateData/corr_2007'
        self.file_prefix = "data_"
        self.shuffle_file_num = 11

        #predict related
        self.predict_output_node_num = 2

        self.paddle_local = True

        #evaluate related
        self.iteration = '02000' #验证集上表现最好的轮数
        self.select_restrict_num = 10 #每次最多取十支股票

        '''
        ## CV params
        self.n_runs = 3
        self.n_folds = 3
        self.stratified_label = "query"
        '''


    def get_sql_by_code(self, codes):
        code_sql = ",".join(codes)
        sql = self.sql_prefix + code_sql + self.sql_postfix
        return sql

## initialize a param config					
config = ParamConfig()