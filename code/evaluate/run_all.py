# coding: utf-8

import os
import stock_select
import sys
sys.path.append("../")
from param_config import config

project_path = '/home/xutao/StockPredictor/'

#paddle预测配置
model_name = 'pass-{}'.format(config.iteration)
if not config.paddle_local:
    timestamp = '2016061511'
    model_path = '/app/insight/EBG/gushitong/xutao/000001same/train/output_{}/model_output/{}'.format(timestamp, model_name)
    model_local_path = '/home/deepDataBase/000001same/model/'
else:
    model_local_path = '/root/paddle/run/train/local/run_package/result/'

paddle_network_conf_path = '/root/paddle/run/network_conf/'
paddle_predict_path = '/root/paddle/run/predict/'

if not config.paddle_local:
    cmd = ["cd " + model_local_path + " && hadoop fs -get " + model_path + " ./"]
else:
    cmd = []
cmd.append("cd " + paddle_network_conf_path + " && rm -f ./trainer_config.conf && cp " + project_path + "config/rnn_c_4predict_trainer_config.conf ./trainer_config.conf && sh ./run.sh")
cmd.append("cd " + paddle_predict_path + " && cp " + paddle_network_conf_path + "binary_conf ./myModel/")
cmd.append("rm -f ./myModel/model/* && cp " + model_local_path + model_name + "/* ./myModel/model/")
cmd = " ; ".join(cmd)
os.system(cmd)

#执行预测
cmd = ["rm -f " + config.predict_result_path + "*"]
cmd.append("cd " + paddle_predict_path)
param2 = "0 0"
for file_name in os.listdir(config.test_file_path):
    input_file = config.test_file_path + file_name
    output_file = config.predict_result_path + file_name
    cmd.append("cat " + input_file + " | ./predict myModel '" + param2 + "' > " + output_file)

cmd = " ; ".join(cmd)
os.system(cmd)

evaluator = stock_select.Evaluator()
evaluator.init()
evaluator.filter_and_select()
win_rate = evaluator.get_win_rate()
filter_averg_return_rate, averg_return_rate = evaluator.get_return_rate()
print "Averg win rate is {}\nAverg return rate is {},\nAll stocks' averg return rate is {} in the same time".format(win_rate, filter_averg_return_rate, averg_return_rate)

