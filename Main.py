import numpy as np
import pandas as pd
import random as rn
from numpy import matlib
from sklearn import decomposition
import Global_Vars
from Model_ACNet import Model_ACNet
from Model_AM1DCNN import Model_AM1DCNN
from Model_Adaboost import Model_Adaboost
from Model_GRU import Model_GRU
from Model_LSTM import Model_LSTM
from Model_NN import Model_NN
from Model_RNN import Model_RNN
from Model_SVM import Model_SVM, SVM_Feat
from Model_TSNE import Model_tsne
from Objfun import Objective_Function_DL, Objective_Function_ML
from PROPOSED import PROPOSED
from PTA import PTA
from Plot_results import *
from SGO import SGO
from TSO import TSO
from WPA import WPA


def find_string(l1, s):
    matched_indexes = []
    for i in range(len(l1)):
        if s == l1[i]:
            matched_indexes.append(i)
    return np.asarray(matched_indexes)


# Read Dataset
an = 0
if an == 1:
    Directory = './Datasets/Dataset1/PS_20174392719_1491204439457_log.csv'
    Data = pd.read_csv(Directory)[:50000]
    tar = Data['isFraud']
    Data.drop(labels=['isFraud'], inplace=True, axis=1)
    Datas = np.asarray(Data)
    for col in range(Datas.shape[1]):
        if isinstance(Datas[:, col][0], str):
            uniq = np.unique((Datas[:, col]))
            for ind_uniq in range(len(uniq)):
                ind = find_string(Datas[:, col], uniq[ind_uniq])
                for index in range(len(ind)):
                    Datas[ind[index]][col] = ind_uniq
    Target = np.asarray(tar)
    np.save('Data.npy', Datas)
    np.save('Target.npy', Target)

# PCA Feature
an = 0
if an == 1:
    data = np.load('Data.npy', allow_pickle=True)
    feat = []
    for i in range(len(data)):
        print(i)
        pca = decomposition.PCA(n_components=1)
        X_std_pca = pca.fit_transform(data[i].reshape(-1, 1))
        feat.append(X_std_pca)
    np.save('PCA_Feature_1.npy', feat)

## t-distributed Stochastic Neighbourhood Embedding (t-SNE) feature
an = 0
if an == 1:
    data = np.load('Data.npy', allow_pickle=True)
    dat = Model_tsne(data)
    np.save('TSNE_Feature_2.npy', dat)

## SVM feature
an = 0
if an == 1:
    data = np.load('Data.npy', allow_pickle=True)
    Target = np.load('Target.npy', allow_pickle=True).reshape(-1, 1)
    feat_Svm = SVM_Feat(data, Target)
    np.save('SVM_Feature_3.npy', feat_Svm)

### feature Concatenation
an = 0
if an == 1:
    Feat_1 = np.load('PCA_Feature_1.npy', allow_pickle=True)
    Feat_2 = np.load('TSNE_Feature_2.npy', allow_pickle=True)
    Feat_3 = np.load('SVM_Feature_3.npy', allow_pickle=True)
    Feat_1 = np.reshape(Feat_1,(Feat_1.shape[0], Feat_1.shape[1]*Feat_1.shape[2]))
    Feat = np.concatenate((Feat_1, Feat_2, Feat_3), axis=1)
    np.save('Features.npy', Feat)


# optimization for Classification
an = 0
if an == 1:
    Feat = np.load('Features.npy', allow_pickle=True)
    Target = np.load('Target.npy', allow_pickle=True)
    Global_Vars.Feat = Feat
    Global_Vars.Target = Target
    Npop = 10
    Chlen = 3
    xmin = matlib.repmat([5, 5, 300], Npop, 1)
    xmax = matlib.repmat([255, 50, 1000], Npop, 1)
    fname = Objective_Function_ML
    initsol = np.zeros((Npop, Chlen))
    for p1 in range(initsol.shape[0]):
        for p2 in range(initsol.shape[1]):
            initsol[p1, p2] = np.random.uniform(xmin[p1, p2], xmax[p1, p2])
    Max_iter = 50

    print("TSO...")
    [bestfit1, fitness1, bestsol1, time1] = TSO(initsol, fname, xmin, xmax, Max_iter)

    print("SGO...")
    [bestfit2, fitness2, bestsol2, time2] = SGO(initsol, fname, xmin, xmax, Max_iter)

    print("PTA...")
    [bestfit3, fitness3, bestsol3, time3] = PTA(initsol, fname, xmin, xmax, Max_iter)

    print("WPA...")
    [bestfit4, fitness4, bestsol4, time4] = WPA(initsol, fname, xmin, xmax, Max_iter)

    print("PROPOSED...")
    [bestfit5, fitness5, bestsol5, time5] = PROPOSED(initsol, fname, xmin, xmax, Max_iter)

    BestSol = [bestsol1, bestsol2, bestsol3, bestsol4, bestsol5]

    np.save('BestSol_ml.npy', BestSol)

##  Classification
an = 0
if an == 1:
    k_fold = 5
    Feat = np.load('Features.npy', allow_pickle=True)
    Targets = np.load('Target.npy', allow_pickle=True)
    best = np.load('BestSol_ml.npy', allow_pickle=True)
    Feature = Feat
    EVAL = []
    for i in range(k_fold):
        Eval = np.zeros((10, 25))
        for j in range(best.shape[0]):  # For all algorithms
            Total_Index = np.arange(Feature[i].shape[0])
            Test_index = np.arange(((i - 1) * (Feature[i].shape[0] / k_fold)) + 1, i * (Feature[i].shape[0] / k_fold))
            Train_Index = np.setdiff1d(Total_Index, Test_index)
            Train_Data = Feature[i][Train_Index, :]
            Train_Target = Targets[i][Train_Index, :]
            Test_Data = Feature[i][Test_index, :]
            Test_Target = Targets[i][Test_index, :]
            sol = np.round((best[j, :])).astype(int)
            Eval[i, :] = Model_ACNet(Train_Data, Train_Target, Test_Data, Test_Target, sol[j].astype(int))
        Eval[5, :], pred1 = Model_NN(Train_Data, Train_Target, Test_Data, Test_Target)
        Eval[6, :], pred2 = Model_SVM(Train_Data, Train_Target, Test_Data, Test_Target)
        Eval[7, :], pred3 = Model_Adaboost(Train_Data, Train_Target, Test_Data, Test_Target)
        Eval[8, :], pred4 = Model_ACNet(Train_Data, Train_Target, Test_Data, Test_Target)
        Eval[9, :] = Eval[4, :]
        EVAL.append(Eval)
    np.save('Eval_Fold.npy', EVAL)

# Optimization for Classification
an = 0
if an == 1:
    Feat1 = np.load('PCA_Feature_1.npy', allow_pickle=True)
    Feat2 = np.load('TSNE_Feature_2.npy', allow_pickle=True)
    Feat3 = np.load('SVM_Feature_3.npy', allow_pickle=True)
    Tar = np.load('Target.npy', allow_pickle=True)
    Global_Vars.Feat1 = Feat1
    Global_Vars.Feat2 = Feat2
    Global_Vars.Feat3 = Feat3
    Global_Vars.Tar = Tar
    Npop = 10
    Chlen = 3
    xmin = matlib.repmat([5, 5, 300], Npop, Chlen)
    xmax = matlib.repmat([255, 50, 1000], Npop, Chlen)
    initsol = np.zeros(xmin.shape)
    for i in range(xmin.shape[0]):
        for j in range(xmin.shape[1]):
            initsol[i, j] = np.random.uniform(xmin[i, j], xmax[i, j])
    fname = Objective_Function_DL
    max_iter = 50

    print('TSO....')
    [bestfit1, fitness1, bestsol1, Time1] = TSO(initsol, fname, xmin, xmax, max_iter)

    print('SGO....')
    [bestfit2, fitness2, bestsol2, Time2] = SGO(initsol, fname, xmin, xmax, max_iter)

    print('PTA....')
    [bestfit3, fitness3, bestsol3, Time3] = PTA(initsol, fname, xmin, xmax, max_iter)

    print('WPA....')
    [bestfit4, fitness4, bestsol4, Time4] = WPA(initsol, fname, xmin, xmax, max_iter)

    print('PROPOSED....')
    [bestfit5, fitness5, bestsol5, Time5] = PROPOSED(initsol, fname, xmin, xmax, max_iter)

    sol = [bestsol1, bestsol2, bestsol3, bestsol4, bestsol5]
    np.save('Best_sol_dl.npy', sol)

## Classification
an = 0
if an == 1:
    Feat1 = np.load('PCA_Feature_1.npy', allow_pickle=True)  # Load the Feat 1
    Feat2 = np.load('TSNE_Feature_2.npy', allow_pickle=True)  # Load the Feat 2
    Feat3 = np.load('SVM_Feature_3.npy', allow_pickle=True)  # Load the Feat 3
    Tar = np.load('Target.npy', allow_pickle=True)
    best = np.load('Best_sol_dl.npy', allow_pickle=True)
    Global_Vars.Feat1 = Feat1
    Global_Vars.Feat2 = Feat2
    Global_Vars.Feat3 = Feat3
    Global_Vars.Tar = Tar
    Target = Tar
    Feat = Feat1
    EVAL = []
    Feature = [Feat1, Feat2, Feat3]
    for learn in range(len(Feature)):
        learnperc = round(Feat.shape[0] * 0.75)
        Train_Data = Feat[:learnperc, :]
        Train_Target = Target[:learnperc, :]
        Test_Data = Feat[learnperc:, :]
        Test_Target = Target[learnperc:, :]
        Eval = np.zeros((10, 25))
        for j in range(best.shape[0]):
            print(learn, j)
            sol = np.round(best[j, :]).astype(np.int16)
            Eval[j, :], pred = Model_AM1DCNN(Feat1, Feat2, Feat3, Target, sol[j].astype(int))
        Eval[5, :], pred1 = Model_LSTM(Train_Data, Train_Target, Test_Data, Test_Target)
        Eval[6, :], pred2 = Model_RNN(Train_Data, Train_Target, Test_Data, Test_Target)
        Eval[7, :], pred3 = Model_GRU(Train_Data, Train_Target, Test_Data, Test_Target)
        Eval[8, :], pred4 = Model_AM1DCNN(Feat1, Feat2, Feat3, Target)
        Eval[9, :], pred5 = Eval[4, :]
        EVAL.append(Eval)
    np.save('Eval_ALL_Feat.npy', EVAL)


Plot_Kfold_Table()
Plot_Kfold()
plot_rocMachine()
plot_rocDeep()
plot_results_conv()
Plot_Results_Feature()