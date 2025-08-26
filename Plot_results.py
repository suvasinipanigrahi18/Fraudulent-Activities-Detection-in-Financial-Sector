import numpy as np
from prettytable import PrettyTable
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn import metrics
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D


def stats(val):
    v = np.zeros(5)
    v[0] = max(val)
    v[1] = min(val)
    v[2] = np.mean(val)
    v[3] = np.median(val)
    v[4] = np.std(val)
    return v


def plot_rocMachine():
    lw = 2
    cls = ['NN', 'SVM', 'Adaboost', 'CNet', 'IWPA-A-CNet']
    colors = cycle(["m", "b", "r", "lime", "k"])
    Predicted = np.load('roc_score.npy', allow_pickle=True)
    Actual = np.load('roc_act.npy', allow_pickle=True)
    for i in range(len(Actual)):
        Dataset = ['Dataset1']
        for j, color in zip(range(5), colors):
            false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(Actual[i, 3, j], Predicted[i, 3, j])
            auc = metrics.roc_auc_score(Actual[i, 3, j], Predicted[i, 3, j])
            plt.plot(
                false_positive_rate1,
                true_positive_rate1,
                color=color,
                lw=lw,
                label=cls[j]
            )
        plt.plot([0, 1], [0, 1], "k--", lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        path = "./Results/%s_Roc_Machine.png" % (Dataset[i])
        plt.savefig(path)
        plt.show()


def plot_rocDeep():
    lw = 2
    cls = ['LSTM', 'RNN', 'GRU', '1DCNN', 'IWPA-AM-1DCNN']
    colors = cycle(["m", "b", "r", "lime", "k"])
    Predicted = np.load('roc_score_Dp.npy', allow_pickle=True)
    Actual = np.load('roc_act_Dp.npy', allow_pickle=True)
    for i in range(len(Actual)):
        Dataset = ['Dataset1']
        for j, color in zip(range(5), colors):
            false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(Actual[i, 3, j], Predicted[i, 3, j])
            auc = metrics.roc_auc_score(Actual[i, 3, j], Predicted[i, 3, j])
            plt.plot(
                false_positive_rate1,
                true_positive_rate1,
                color=color,
                lw=lw,
                label=cls[j]
            )
        plt.plot([0, 1], [0, 1], "k--", lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        path = "./Results/%s_Roc_Deep.png" % (Dataset[i])
        plt.savefig(path)
        plt.show()




def Plot_Kfold_Table():
    eval = np.load('Eval_Fold.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1_score', 'MCC',
             'FOR', 'pt',
             'ba', 'fm', 'bm', 'mk', 'PLHR', 'lrminus', 'dor', 'prevalence', 'TS']
    Table_Terms = [0, 2, 3, 4, 9, 13, 14, 15]
    Algorithm = ['TERMS', 'TSO-AM-1DCNN', 'SGO-AM-1DCNN', 'PTA-AM-1DCNN', 'WPA-AM-1DCNN', 'IWPA-AM-1DCNN']
    Classifier = ['TERMS', 'LSTM', 'RNN', 'GRU', '1DCNN', 'IWPA-AM-1DCNN']
    Activation = ['1', '2', '3', '4', '5']
    Dataset = ['Dataset 1']


    value = eval[0, 3, :, 4:]
    Table = PrettyTable()
    Table.add_column(Algorithm[0], (np.asarray(Terms))[np.asarray(Table_Terms)])
    for j in range(len(Algorithm) - 1):
        Table.add_column(Algorithm[j + 1], value[j, Table_Terms])
    print('-------------------------------------------------- ', Dataset[0], 'k Fold ',
          'Algorithm Comparison --------------------------------------------------')
    print(Table)

    Table = PrettyTable()
    Table.add_column(Classifier[0], (np.asarray(Terms))[np.asarray(Table_Terms)])
    for j in range(len(Classifier) - 1):
        Table.add_column(Classifier[j + 1], value[len(Algorithm) + j - 1, Table_Terms])
    print('-------------------------------------------------- ', Dataset[0], 'k Fold ',
          'Classifier Comparison --------------------------------------------------')
    print(Table)


def Plot_Kfold():
    eval = np.load('Eval_Fold.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1_score', 'MCC',
             'FOR', 'pt',
             'ba', 'fm', 'bm', 'mk', 'PLHR', 'lrminus', 'dor', 'prevalence', 'TS']
    Graph_Term = [0, 2, 3, 4, 9, 15, 16]
    for n in range(eval.shape[0]):
        for j in range(len(Graph_Term)):
            Graph = np.zeros((eval.shape[1], eval.shape[2]))
            for k in range(eval.shape[1]):
                for l in range(eval.shape[2]):
                    Graph[k, l] = eval[n, k, l, Graph_Term[j] + 4]

            length = np.arange(5)
            fig = plt.figure()
            ax = fig.add_axes([0.15, 0.1, 0.7, 0.8])

            ax.plot(length, Graph[:, 0], color='#010fcc', linewidth=3, marker='>', markerfacecolor='red',
                    markersize=12,
                    label='TSO-AM-1DCNN')
            ax.plot(length, Graph[:, 1], color='#08ff08', linewidth=3, marker='>', markerfacecolor='green',
                    markersize=12,
                    label='SGO-AM-1DCNN')
            ax.plot(length, Graph[:, 2], color='#fe420f', linewidth=3, marker='>', markerfacecolor='cyan',
                    markersize=12,
                    label='PTA-AM-1DCNN')
            ax.plot(length, Graph[:, 3], color='#f504c9', linewidth=3, marker='>', markerfacecolor='#fdff38',
                    markersize=12,
                    label='WPA-AM-1DCNN')
            ax.plot(length, Graph[:, 4], color='k', linewidth=3, marker='>', markerfacecolor='w', markersize=12,
                    label='IWPA-AM-1DCNN')

            ax.fill_between(length, Graph[:, 0], Graph[:, 3], color='#acc2d9', alpha=.5)  # ff8400
            ax.fill_between(length, Graph[:, 3], Graph[:, 2], color='#c48efd', alpha=.5)  # 19abff
            ax.fill_between(length, Graph[:, 2], Graph[:, 1], color='#be03fd', alpha=.5)  # 00f7ff
            ax.fill_between(length, Graph[:, 1], Graph[:, 4], color='#b2fba5', alpha=.5)  # ecfc5b
            plt.xticks(length, ('1', '2', '3', '4', '5'))
            plt.xlabel('k Fold', fontname="Arial", fontsize=12, fontweight='bold', color='#35530a')
            plt.ylabel(Terms[Graph_Term[j]], fontname="Arial", fontsize=12, fontweight='bold', color='#35530a')
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.14), ncol=3, fancybox=True, shadow=False)
            path = "./Results/Dataset_%s_kfold_%s_Alg.png" % (n + 1, Terms[Graph_Term[j]])
            plt.savefig(path)
            plt.show()

            Batch_size = ['1', '2', '3', '4', '5']
            colors = ['#be03fd', '#fe02a2', '#0804f9', '#02c14d', 'k']
            mtd_Graph = Graph[:, 5:]
            Methods = ['LSTM', 'RNN', 'GRU', '1DCNN', 'IWPA-AM-1DCNN']
            fig = plt.figure()
            ax = fig.add_axes([0.15, 0.1, 0.7, 0.8])
            triangle_width = 0.15
            for y, algorithm in enumerate(Methods):
                for x, batch_size in enumerate(Batch_size):
                    x_pos = x + y * triangle_width
                    value = mtd_Graph[x, y]
                    color = colors[y]
                    triangle_points = np.array(
                        [[x_pos - triangle_width / 2, 0], [x_pos + triangle_width / 2, 0], [x_pos, value]])
                    triangle = Polygon(triangle_points, ec='k', closed=True, color=color, linewidth=1)
                    ax.add_patch(triangle)
                    ax.plot(x_pos, value, marker='.', color=color, markersize=12)

            ax.set_xticks(np.arange(len(Batch_size)) + (len(Methods) * triangle_width) / 2)
            ax.set_xticklabels([str(size) for size in Batch_size])
            ax.legend([Line2D([0], [0], color=color, lw='4') for color in colors], Methods, loc='upper center',
                      bbox_to_anchor=(0.5, 1.14), ncol=3)
            plt.xlabel('K Fold', fontname="Arial", fontsize=12, fontweight='bold', color='#35530a')
            plt.ylabel(Terms[Graph_Term[j]], fontname="Arial", fontsize=12, fontweight='bold', color='#35530a')
            path = "./Results/Dataset_%s_kfold_%s_Med.png" % (n + 1, Terms[Graph_Term[j]])
            plt.savefig(path)
            plt.show()


def plot_results_conv():
    Result = np.load('Fitness.npy', allow_pickle=True)
    Fitness = np.load('Fitness.npy', allow_pickle=True)
    Algorithm = ['TERMS', 'TSO-AM-1DCNN', 'SGO-AM-1DCNN', 'PTA-AM-1DCNN', 'WPA-AM-1DCNN', 'IWPA-AM-1DCNN']
    for i in range(Result.shape[0]):
        Terms = ['Worst', 'Best', 'Mean', 'Median', 'Std']
        Dataset = ['Dataset1', 'Dataset2', 'Dataset3', 'Dataset4', 'Dataset5']
        Conv_Graph = np.zeros((5, 5))
        for j in range(5):
            Conv_Graph[j, :] = stats(Fitness[i, j, :])
        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], Conv_Graph[j, :])
        print('-------------------------------------------------- ', Dataset[i], 'Statistical Report ',
              '--------------------------------------------------')
        print(Table)
        length = np.arange(50)
        Conv_Graph = Fitness[i]
        plt.plot(length, Conv_Graph[0, :], color='r', linewidth=3, marker='.', markerfacecolor='red', markersize=12,
                 label='TSO-AM-1DCNN')
        plt.plot(length, Conv_Graph[1, :], color='c', linewidth=3, marker='.', markerfacecolor='green',
                 markersize=12,
                 label='SGO-AM-1DCNN')
        plt.plot(length, Conv_Graph[2, :], color='b', linewidth=3, marker='.', markerfacecolor='cyan',
                 markersize=12,
                 label='PTA-AM-1DCNN')
        plt.plot(length, Conv_Graph[3, :], color='y', linewidth=3, marker='.', markerfacecolor='magenta',
                 markersize=12,
                 label='WPA-AM-1DCNN')
        plt.plot(length, Conv_Graph[4, :], color='k', linewidth=3, marker='.', markerfacecolor='black',
                 markersize=12,
                 label='IWPA-AM-1DCNN')
        plt.xlabel('Iteration')
        plt.ylabel('Cost Function')
        plt.legend(loc=1)
        plt.savefig("./Results/%s_Conv.png" % (Dataset[i]))
        plt.show()




def Plot_Results_Feature():
    for i in range(1):
        Eval = np.load('Eval_ALL_Feat.npy', allow_pickle=True)[i]
        Terms = np.asarray(
            ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1_score',
             'MCC', 'FOR', 'PT', 'BA', 'FM', 'BM', 'MK', 'PLHR', 'Lrminus', 'DOR', 'Prevalence', 'TS'])
        Graph_Term = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).astype(int)
        Algorithm = ['TERMS', 'TSO-A-CNet ', 'SGO-A-CNet ', 'PTA-A-CNet ', 'WPA-A-CNet ', 'IWPA-A-CNet']
        Classifier = ['TERMS', 'NN', 'SVM', 'Adaboost', 'CNet', 'IWPA-A-CNet']
        value = Eval[2, :, 4:]

        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms[Graph_Term])
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], value[j, Graph_Term])
        print('-------------------------------------------------- feature Algorithm Comparison - Dataset', i + 1,
              '--------------------------------------------------')
        print(Table)

        Table = PrettyTable()
        Table.add_column(Classifier[0], Terms[Graph_Term])
        for j in range(len(Classifier) - 1):
            Table.add_column(Classifier[j + 1], value[len(Algorithm) + j - 1, Graph_Term])
        print('--------------------------------------------------- feature Classifier Comparison - Dataset', i + 1,
              '--------------------------------------------------')
        print(Table)

        Eval = np.load('Eval_ALL_Feat.npy', allow_pickle=True)[i]
        BATCH = [1, 2, 3, 4]
        for j in range(len(Graph_Term)):
            Graph = np.zeros((Eval.shape[0], Eval.shape[1]))
            for k in range(Eval.shape[0]):
                for l in range(Eval.shape[1]):
                    Graph[k, l] = Eval[k, l, Graph_Term[j] + 4]
            X = np.arange(4)
            plt.plot(BATCH, Graph[:, 0], '-.', color='m', linewidth=3, marker='*', markerfacecolor='k', markersize=16,
                     label="TSO-A-CNet")
            plt.plot(BATCH, Graph[:, 1], '-.', color='#ef4026', linewidth=3, marker='*', markerfacecolor='k',
                     markersize=16,
                     label="SGO-A-CNet")
            plt.plot(BATCH, Graph[:, 2], '-.', color='lime', linewidth=3, marker='*', markerfacecolor='k',
                     markersize=16,
                     label="PTA-A-CNet")
            plt.plot(BATCH, Graph[:, 3], '-.', color='#0804f9', linewidth=3, marker='*', markerfacecolor='k',
                     markersize=16,
                     label="WPA-A-CNet")
            plt.plot(BATCH, Graph[:, 4], '-.', color='k', linewidth=3, marker='*', markerfacecolor='white',
                     markersize=16,
                     label="IWPA-A-CNet")
            plt.xticks(X + 1, ('Feature 1', 'Feature 2', 'Feature 3', 'Fused Feature'))
            plt.ylabel(Terms[Graph_Term[j]])
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                       ncol=3, fancybox=True, shadow=True)
            path1 = "./Results/Feature_%s_ALg.png" % (Terms[Graph_Term[j]])
            plt.savefig(path1)
            plt.show()

            fig = plt.figure()
            ax = fig.add_axes([0.15, 0.1, 0.7, 0.8])
            X = np.arange(4)
            ax.bar(X + 0.00, Graph[:, 5], color='#0804f9', edgecolor='k', width=0.12, hatch="..", label="NN")
            ax.bar(X + 0.12, Graph[:, 6], color='#b1d1fc', edgecolor='k', width=0.12, hatch="..", label="SVM")
            ax.bar(X + 0.23, Graph[:, 7], color='#be03fd', edgecolor='k', width=0.12, hatch='..', label="Adaboost")
            ax.bar(X + 0.36, Graph[:, 8], color='lime', edgecolor='k', width=0.12, hatch="..", label="CNet")
            ax.bar(X + 0.48, Graph[:, 9], color='k', edgecolor='w', width=0.12, hatch="//", label="IWPA-A-CNet")
            plt.xticks(X + 0.25, ('Feature 1', 'Feature 2', 'Feature 3', 'Fused Feature'))
            plt.ylabel(Terms[Graph_Term[j]])
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True)
            path = "./Results/Feature_%s_Med.png" % ((Terms[Graph_Term[j]]))
            plt.savefig(path)
            plt.show()



def Plot_Feat_Table():
    for i in range(1):
        Eval = np.load('Eval_ALL_Feat.npy', allow_pickle=True)[i]
        Terms = np.asarray(
            ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1_score',
             'MCC', 'FOR', 'PT', 'BA', 'FM', 'BM', 'MK', 'PLHR', 'Lrminus', 'DOR', 'Prevalence', 'TS'])
        Graph_Term = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).astype(int)
        Algorithm = ['TERMS', 'Feature 1', 'Feature 2', 'Feature 3', 'Fused Feature']
        Classifier = ['TERMS', 'NN', 'SVM', 'Adaboost', 'CNet', 'IWPA-A-CNet']
        value = Eval[:, 4, 4:]

        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms[Graph_Term])
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], value[j, Graph_Term])
        print('-------------------------------------------------- feature Algorithm Comparison - Dataset', i + 1,
              '--------------------------------------------------')
        print(Table)

        # Table = PrettyTable()
        # Table.add_column(Classifier[0], Terms[Graph_Term])
        # for j in range(len(Classifier) - 1):
        #     Table.add_column(Classifier[j + 1], value[len(Algorithm) + j - 1, Graph_Term])
        # print('--------------------------------------------------- feature Classifier Comparison - Dataset', i + 1,
        #       '--------------------------------------------------')
        # print(Table)


if __name__ == '__main__':
    Plot_Feat_Table()
    # Plot_Kfold_Table()
    # Plot_Kfold()
    # plot_rocMachine()
    # plot_rocDeep()
    # plot_results_conv()
    # Plot_Results_Feature()
