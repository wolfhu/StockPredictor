# coding: utf-8

class Result(object):
    def __init__(self, strategy, learning_rate, punishment):
        self.strategy = strategy
        self.learning_rate = learning_rate
        self.punishment = punishment

        self.epoch = 0
        self.val_winrate_05 = 0
        self.val_winrate_06 = 0
        self.val_winrate_07 = 0
        self.val_winrate_08 = 0
        self.val_winrate_09 = 0
        self.test_winrate_05 = 0
        self.test_winrate_06 = 0
        self.test_winrate_07 = 0
        self.test_winrate_08 = 0
        self.test_winrate_09 = 0
        self.val_pos_num_05 = 0
        self.val_pos_num_06 = 0
        self.val_pos_num_07 = 0
        self.val_pos_num_08 = 0
        self.val_pos_num_09 = 0
        self.test_pos_num_05 = 0
        self.test_pos_num_06 = 0
        self.test_pos_num_07 = 0
        self.test_pos_num_08 = 0
        self.test_pos_num_09 = 0

        self.val_acc = 0
        self.test_acc = 0

    def get_new_result(self, val_winrate_05, val_pos_num_05, test_winrate_05, test_pos_num_05,
                       val_winrate_06, val_pos_num_06, test_winrate_06, test_pos_num_06,
                       val_winrate_07, val_pos_num_07, test_winrate_07, test_pos_num_07,
                       val_winrate_08, val_pos_num_08, test_winrate_08, test_pos_num_08,
                       val_winrate_09, val_pos_num_09, test_winrate_09, test_pos_num_09,
                       val_acc, test_acc):
        result = Result(self.strategy, self.learning_rate, self.punishment)
        result.epoch = 0
        result.val_winrate_05 = val_winrate_05
        result.val_pos_num_05 = val_pos_num_05
        result.test_winrate_05 = test_winrate_05
        result.test_pos_num_05 = test_pos_num_05

        result.val_winrate_06 = val_winrate_06
        result.val_pos_num_06 = val_pos_num_06
        result.test_winrate_06 = test_winrate_06
        result.test_pos_num_06 = test_pos_num_06

        result.val_winrate_07 = val_winrate_07
        result.val_pos_num_07 = val_pos_num_07
        result.test_winrate_07 = test_winrate_07
        result.test_pos_num_07 = test_pos_num_07

        result.val_winrate_08 = val_winrate_08
        result.val_pos_num_08 = val_pos_num_08
        result.test_winrate_08 = test_winrate_08
        result.test_pos_num_08 = test_pos_num_08

        result.val_winrate_09 = val_winrate_09
        result.val_pos_num_09 = val_pos_num_09
        result.test_winrate_09 = test_winrate_09
        result.test_pos_num_09 = test_pos_num_09

        result.val_acc = val_acc
        result.test_acc = test_acc

        return result
