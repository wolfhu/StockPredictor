# coding: utf-8

from result import Result

def read():
    with open('../../data/601088.result', 'r') as f:
        lines = f.readlines()

    result_list = []
    result_template = None
    ix = 0
    while ix < len(lines):
        line = lines[ix]
        if 'learning_rate' in line:
            tmp = line.split(',')
            learning_rate = float(tmp[0].split()[-1])
            strategy = tmp[1].split()[-1]
            punishment = float(tmp[2].split()[-1])
            result_template = Result(strategy, learning_rate, punishment)
            ix += 2
        elif 'val' in line:
            val_winrate_05 = float(line.split()[-1])
            ix += 1
            val_pos_num_05 = float(lines[ix].split()[-1])
            ix += 1
            test_winrate_05 = float(lines[ix].split()[-1])
            ix += 1
            test_pos_num_05 = float(lines[ix].split()[-1])

            ix += 1
            val_winrate_06 = float(lines[ix].split()[-1])
            ix += 1
            val_pos_num_06 = float(lines[ix].split()[-1])
            ix += 1
            test_winrate_06 = float(lines[ix].split()[-1])
            ix += 1
            test_pos_num_06 = float(lines[ix].split()[-1])

            ix += 1
            val_winrate_07 = float(lines[ix].split()[-1])
            ix += 1
            val_pos_num_07 = float(lines[ix].split()[-1])
            ix += 1
            test_winrate_07 = float(lines[ix].split()[-1])
            ix += 1
            test_pos_num_07 = float(lines[ix].split()[-1])

            ix += 1
            val_winrate_08 = float(lines[ix].split()[-1])
            ix += 1
            val_pos_num_08 = float(lines[ix].split()[-1])
            ix += 1
            test_winrate_08 = float(lines[ix].split()[-1])
            ix += 1
            test_pos_num_08 = float(lines[ix].split()[-1])

            ix += 1
            val_winrate_09 = float(lines[ix].split()[-1])
            ix += 1
            val_pos_num_09 = float(lines[ix].split()[-1])
            ix += 1
            test_winrate_09 = float(lines[ix].split()[-1])
            ix += 1
            test_pos_num_09 = float(lines[ix].split()[-1])

            ix += 1
            epoch = int(lines[ix].split()[1])
            ix += 4
            val_acc = float(lines[ix].split()[2]) / 100
            ix += 2
            test_acc = float(lines[ix].split()[2]) / 100

            single_result = result_template.get_new_result(epoch,
                val_winrate_05, val_pos_num_05, test_winrate_05, test_pos_num_05,
                val_winrate_06, val_pos_num_06, test_winrate_06, test_pos_num_06,
                val_winrate_07, val_pos_num_07, test_winrate_07, test_pos_num_07,
                val_winrate_08, val_pos_num_08, test_winrate_08, test_pos_num_08,
                val_winrate_09, val_pos_num_09, test_winrate_09, test_pos_num_09,
                val_acc, test_acc)
            result_list.append(single_result)
            ix += 2
        else:
            print line
            ix += 1
    return result_list

if __name__ == '__main__':
    result_list = read()
    #filtering
    result_list = filter(lambda item: item.val_pos_num_05 > 100,  result_list)

    result_list.sort(key=lambda item: 1 * (item.val_winrate_09 + item.test_winrate_09) + 0 * (item.val_acc + item.test_acc), reverse=True)
    for ix in xrange(10):
        print result_list[ix]