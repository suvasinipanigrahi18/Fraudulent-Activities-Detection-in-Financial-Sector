import numpy as np
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.models import Sequential
from Evaluation import evaluation


def Model_AM1DCNN(Feat_1, Feat_2, Feat_3, Target, sol=None):
    if sol is None:
        sol = [5, 5, 300]
    Eval_val = []
    Pred_val = []
    for i in range(5):
        if i == 0:
            img = Feat_1
        elif i == 1:
            img = Feat_2
        elif i == 2:
            img = Feat_3
            learnperc = round(img.shape[0] * 0.75)  # Split Training and Testing Datas
            trainX = img[:learnperc, :]
            trainy = Target[:learnperc, :]
            testX = img[learnperc:, :]
            testy = Target[learnperc:, :]
            epoches = sol[1]
            batchSize = 64
            dropout = 0.5
            hN = sol[0]
            verbose, epochs, batch_size = 0, epoches, batchSize
            n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[1], trainy.shape[1]

            Train_X = np.zeros((trainX.shape[0], n_timesteps, n_features))
            for i in range(trainX.shape[0]):
                temp = np.resize(trainX[i], (n_timesteps * n_features))
                Train_X[i] = np.reshape(temp, (n_timesteps, n_features))

            Test_X = np.zeros((testX.shape[0], n_timesteps, n_features))
            for i in range(testX.shape[0]):
                temp = np.resize(testX[i], (n_timesteps * n_features))
                Test_X[i] = np.reshape(temp, (n_timesteps, n_features))

            model = Sequential()
            model1 = model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=Feat_1))
            model2 = model1.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=Feat_2))
            model3 = model2.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=Feat_3))
            model = (model1 + model2 + model3) / 3
            model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
            model.add(Dropout(dropout))
            model.add(MaxPooling1D(pool_size=2))
            model.add(Flatten())
            model.add(Dense(hN, activation='relu'))
            model.add(Dense(n_outputs, activation='softmax'))
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'], run_eagerly=True)
            # fit network
            model.fit(Train_X, trainy, epochs=epochs, batch_size=batch_size, steps_per_epoch=sol[2], validation_data=(Test_X, testy))
            pred = model.predict(Test_X)
            Eval = evaluation(pred, testy)
            Pred_val.append(pred)
            Eval_val.append(Eval)
    return Eval_val, Pred_val
