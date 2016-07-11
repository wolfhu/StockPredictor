# coding: utf-8

import os
import numpy as np


############
## Config ##
############
class ParamConfig:
    def __init__(self):

        ''' --------------数据获取时的相关配置 --------------------------- '''
        self.n_classes = 2 #最后的结果是分成三类还是分成两类，三类为：上涨、下跌、震荡，两类为：上涨、下跌
        self.before = 20  #M天的过去数据预测N天之后，M=20
        self.after = 5    #M天的过去数据预测N天之后，N=5
        self.interval = 5  #取样本时中间跨度，即隔几天取作为一个样本
        self.left4test = 20  #每只股票的最后n个样本留下来作为测试机，做成n个横截面
        self.threhold_up_down = 0.05    #涨跌幅的阈值，N天后涨跌幅超过该值，才算是真正的涨跌
        self.fluctuation_threhold = 0.0  #分三类时，震荡的阈值，涨跌幅小于此才算作震荡
        self.thread_size = 4            #并行的线程数
        # cols_dimension = ['open', 'high', 'low', 'close', 'amount']
        self.cols_dimension = ['open', 'close', 'high', 'low', 'MA3', 'MA5', 'MA10', 'MA20', 'MA30', 'MA60', 'EMA3', 'EMA5', 'EMA10', 'EMA20', 'EMA30', 'EMA60']
        self.sql_prefix = "SELECT * FROM tech_ind WHERE symbol in ("
        self.sql_postfix = ")"
        self.code_file = "../../data/zz800_all_new.xls"
        self.ip = "10.100.47.147"
        self.user = "root"
        self.passwd = "rootsafe"
        self.database = "Tech_Indicators"
        self.port = 8080


        #filter related
        self.close_up_threhold = 1.09
        self.close_down_threhold = 0.91
        self.amount_up_threhold = 9
        self.amount_down_threhold = 0.11

        #norm related
        self.norm_strategy = 'min-max'

        #corr realated
        self.same_stock_num = 30

        #file related
        self.code_file_path = '/home/deepDataBase/000060same/code/'
        self.test_file_path = '/home/deepDataBase/000060same/test/'
        self.shuffle_file_path = '/home/deepDataBase/000060same/time/'
        self.corr_file = '/home/xutao/generateData/corr_2007'
        self.file_prefix = "data_"
        self.shuffle_file_num = 1

        #predict related
        self.predict_output_node_num = 2

        self.paddle_local = False

        self.dnn_strategy = 'partitioned'

        #evaluate related
        self.iteration = '02000' #验证集上表现最好的轮数
        self.select_restrict_num = 10 #每次最多取十支股票

        '''
        ## CV params
        self.n_runs = 3
        self.n_folds = 3
        self.stratified_label = "query"
        '''

    def get_sql_by_code(self, codes, begin_date, end_date):
        code_sql = ",".join(codes)
        sql = self.sql_prefix + code_sql + self.sql_postfix + ' where date between {} and {}'.format(begin_date, end_date)
        return sql

## initialize a param config					
config = ParamConfig()