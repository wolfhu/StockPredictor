# coding: utf-8

import pandas as pd
import os
import sys
sys.path.append("../")
from param_config import config

class Evaluator(object):
    def init(self):
        self.df_list = []
        for file_name in os.listdir(config.predict_result_path):
            df_data = []
            with open(config.predict_result_path + file_name, 'r') as f:
                predict_reuslt = f.readlines()
                for line in predict_reuslt:
                    lines = line.split()
                    code = lines[0]
                    if config.predict_output_node_num == 2:
                        real_num = float(lines[-3])
                    elif config.predict_output_node_num == 1:
                        real_num = float(lines[-2])
                    up_node_output = float(lines[-1])
                    df_data.append([code, real_num, up_node_output])

                self.df_list.append(pd.DataFrame(df_data, columns=['code', 'real', 'predict_up']))

    def filter_and_select(self):
        self.filter_df_list = []
        for df in self.df_list:
            filter_df = None
            #首先过滤非上涨的
            if config.predict_output_node_num == 2:
                filter_df = df[df.predict_up > 0.5]
            elif config.predict_output_node_num == 1:
                filter_df = df[df.predict_up > 0]

            #进一步严格条件
            #1.每次最多取前十支
            filter_df = filter_df.sort(['predict_up'])
            filter_df = filter_df.iloc[:min(config.select_restrict_num, len(filter_df)),:]

            self.filter_df_list.append(filter_df)

    #计算平均胜率
    def get_win_rate(self):
        sum_rate = 0.0
        for df in self.filter_df_list:
            win_num = len(df[df.real > 0])
            if len(df):
                sum_rate += (win_num + 0.0) / len(df)
        averg_win_rate = sum_rate / len(self.filter_df_list)
        return averg_win_rate

    #计算收益率
    def get_return_rate(self):
        filter_sum_return_rate = 0.0
        sum_return_rate = 0.0
        for ix in xrange(len(self.df_list)):
            df = self.df_list[ix]
            filter_df = self.filter_df_list[ix]
            if not len(filter_df):
                continue

            filter_sum_return_rate += filter_df['real'].mean()
            sum_return_rate += df['real'].mean()
        averg_return_rate = sum_return_rate / len(self.df_list)
        filter_averg_return_rate = filter_sum_return_rate / len(self.df_list)

        return filter_averg_return_rate, averg_return_rate



