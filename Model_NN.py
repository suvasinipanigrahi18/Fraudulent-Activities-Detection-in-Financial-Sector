import numpy as np
from Evaluation import evaluation
from Neural_Network import train_nn


def Model_NN(train_data, train_target, test_data, test_target):
    train_data = np.resize(train_data, (train_data.shape[0], 100))
    test_data = np.resize(train_data, (test_data.shape[0], 100))
    pred = np.zeros(test_target.shape)
    for i in range(train_target.shape[1]):
        out, model2 = train_nn(train_data, train_target[:, i], test_data)  # DNN
        pred[:, i] = np.reshape(np.asarray(out), (out.shape[0]))
    pred[pred < 0.5] = 0
    pred[pred >= 0.5] = 1
    Eval = evaluation(pred, test_target)
    return Eval, pred

