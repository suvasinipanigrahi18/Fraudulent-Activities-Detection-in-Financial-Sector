import numpy as np
from Evaluation import evaluation
from Global_Vars import Global_vars
from Model_ACNet import Model_ACNet
from Model_AM1DCNN import Model_AM1DCNN


def Objective_Function_ML(Soln):
    Feat = Global_vars.Feat
    Tar = Global_vars.Target
    Tar = np.reshape(Tar, (-1, 1))
    Fitn = np.zeros(Soln.shape[0])
    dimension = len(Soln.shape)
    if dimension == 2:
        learnper = round(Feat.shape[0] * 0.75)
        for i in range(Soln.shape[0]):
            sol = np.round(Soln[i, :]).astype(np.int16)
            Train_Data = Feat[:learnper, :]
            Train_Target = Tar[:learnper, :]
            Test_Data = Feat[learnper:, :]
            Test_Target = Tar[learnper:, :]
            Eval, predict = Model_ACNet(Train_Data, Train_Target, Test_Data, Test_Target, sol)
            Eval = evaluation(predict, Test_Target)
            Fitn[i] = 1 / (Eval[7] + Eval[13]) + Eval[8]
        return Fitn
    else:
        learnper = round(Feat.shape[0] * 0.75)
        sol = np.round(Soln).astype(np.int16)
        Train_Data = Feat[:learnper, :]
        Train_Target = Tar[:learnper, :]
        Test_Data = Feat[learnper:, :]
        Test_Target = Tar[learnper:, :]
        Eval, predict = Model_ACNet(Train_Data, Train_Target, Test_Data, Test_Target, sol)
        Eval = evaluation(predict, Test_Target)
        Fitn = 1 / (Eval[7] + Eval[13]) + Eval[8]
        return Fitn


def Objective_Function_DL(Soln):
    Feat_1 = Global_vars.Feat1
    Feat_2 = Global_vars.Feat2
    Feat_3 = Global_vars.Feat3
    Target = Global_vars.Target
    Fitn = np.zeros(Soln.shape[0])
    dimension = len(Soln.shape)
    if dimension == 2:
        for i in range(Soln.shape[0]):
            sol = np.round(Soln[i, :]).astype(np.int16)
            Eval, pred = Model_AM1DCNN(Feat_1, Feat_2, Feat_3, Target, sol)
            Eval = evaluation(pred, Target)
            Fitn[i] = 1 / (Eval[4] + Eval[7])
        return Fitn
    else:
        sol = np.round(Soln).astype(np.int16)
        Eval, pred = Model_AM1DCNN(Feat_1, Feat_2, Feat_3, Target, sol)
        Eval = evaluation(pred, Target)
        Fitn = 1 / (Eval[4] + Eval[7])
        return Fitn
