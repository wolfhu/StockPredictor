__author__ = 'tao'
import cPickle
import json
from lasagne.layers import get_all_param_values

file_names = {'201604':'201604_model_file', '201605':'201605_model_file', '201606':'201606_model_file'}
for input_file, output_file in file_names.iteritems():
    with open(input_file, 'r') as f:
        network = cPickle.load(f)

    params = get_all_param_values(network)
    lines = []
    for param in params:
        lines.append(json.dumps(param.tolist()))

    with open(output_file, 'w') as f:
        for ix in range(len(lines)):
            f.write(lines[ix])
            f.write('\n')
