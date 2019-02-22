# -*- coding: utf-8 -*-
import numpy as np
from sklearn.metrics import f1_score


def evaluate(pred, gold):
    """
    evaluate accuracy and macro-F1 of ABSA task
    """
    pred_count = np.zeros(3, dtype='int32')
    gold_count = np.zeros(3, dtype='int32')
    hit_count = np.zeros(3, dtype='int32')

    # number of testing documents
    n_test = len(gold)
    error_cases = {}
    for i in range(n_test):
        y_p = int(pred[i])
        y_g = gold[i]
        # print('y_p=', y_p)
        pred_count[y_p] += 1
        gold_count[y_g] += 1
        if y_p == y_g:
            hit_count[y_p] += 1
        else:
            error_cases[i] = [y_p, y_g]
    # number of true predictions
    total_hit = sum(hit_count)
    # accuracy
    acc = float(total_hit) / n_test
    # macro_f1
    macro_f = f1_score(y_true=gold, y_pred=pred, labels=[0, 1, 2], average='macro')
    result_string = ''
    result_string = '%sneg: recall: %s/%s, precision: %s/%s \n' % (result_string,
                                                                   hit_count[0], gold_count[0], hit_count[0],
                                                                   pred_count[0])
    result_string = '%spos: recall: %s/%s, precision: %s/%s \n' % (result_string,
                                                                   hit_count[1], gold_count[1], hit_count[1],
                                                                   pred_count[1])
    result_string = '%sneu: recall: %s/%s, precision: %s/%s \n' % (result_string,
                                                                   hit_count[2], gold_count[2], hit_count[2],
                                                                   pred_count[2])
    return acc, macro_f, result_string, error_cases
