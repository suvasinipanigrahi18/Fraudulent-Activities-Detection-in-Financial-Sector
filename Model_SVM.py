from sklearn.svm import SVC  # "Support Vector Classifier"
import numpy as np
from Evaluation import evaluation



def SVM_Feat(train_data, train_target):
    # Train SVM model
    svm_model = SVC(kernel='linear', C=1.0)  # Linear SVM
    svm_model.fit(train_data, train_target)

    # Extract the features
    features = svm_model.decision_function(train_data)
    return features



def Model_SVM(train_data, train_target, test_data, test_target):
    clf = SVC(kernel='linear', max_iter=25)
    pred = np.zeros((test_target.shape[0], test_target.shape[1])).astype('int')
    for i in range(train_target.shape[1]):
        clf.fit(train_data.tolist(), train_target[:, i].tolist())
        Y_pred = clf.predict(test_data.tolist())
        pred[:, i] = np.round(Y_pred.ravel())
    Eval = evaluation(pred, test_target)
    return np.asarray(Eval).ravel(), pred









