# coding: utf-8

import os
import sys
sys.path.append("../")
from param_config import config

project_path = '/home/xutao/StockPredictor/'

#运行pre_val_upload_mv.sh
paddle_preprocess_path = '/root/paddle/run/preprocess/'
cmd = ["cd " + paddle_preprocess_path]
if config.paddle_local:
    cmd.append("rm -f ./pre_val_upload.sh && cp " + project_path + "config/pre_val_upload.sh ./ && sh pre_val_upload.sh")
else:
    cmd.append("rm -f ./pre_val_upload_mv.sh && cp " + project_path + "config/pre_val_upload_mv.sh ./ && sh pre_val_upload_mv.sh")
cmd = " && ".join(cmd)
os.system(cmd)

#配置网络
if config.paddle_local:
    paddle_train_path = '/root/paddle/run/train/local/run_package/conf/'
else:
    paddle_train_path = '/root/paddle/run/train/cluster/run_package/'
cmd = ["cd " + paddle_train_path + " && rm -f ./common.conf ./trainer_config.conf && cp " + project_path + "config/rnn_c_trainer_config.conf ./trainer_config.conf "]
if config.paddle_local:
    cmd.append("cp " + project_path + "config/local_common.conf ./common.conf")
    cmd.append("cd .. && ./run.sh cpu > /dev/null 2> /dev/null")
else:
    cmd.append("cp "  + project_path + "config/cluster_common.conf ./common.conf")
    cmd.append("./submit.sh cpu > /dev/null 2> /dev/null")

cmd = " && ".join(cmd)
os.system(cmd)

print "Deploy successfully"