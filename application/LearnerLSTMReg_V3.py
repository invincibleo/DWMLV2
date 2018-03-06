#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 16.11.17 10:12
# @Author  : Duowei Tang
# @Site    : https://iiw.kuleuven.be/onderzoek/emedia/people/phd-students/duoweitang
# @File    : LearnerLSTMReg
# @Software: PyCharm Community Edition

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import os
import tensorflow as tf
import keras
import shutil
import math
import matplotlib.pyplot as plt

from scipy.linalg import toeplitz
from keras.layers import Input, LSTM, Lambda
from core.Models import *
from core.Learner import Learner
from core.Metrics import *

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR

num_t_x = 10
class LearnerLSTMReg(Learner):
    def learn(self):
        model_json_file_addr, model_h5_file_addr = self.generate_pathes()
        continue_training = False
        if not os.path.exists(model_json_file_addr) or continue_training:
            self.copy_configuration_code()  # copy the configuration code so that known in which condition the model is trained

            # expected input data shape: (batch_size, timesteps, data_dim)
            regr0 = RandomForestRegressor(n_estimators=10, random_state=0, verbose=2)
            # regr0 = LinearSVR(random_state=0, loss='squared_epsilon_insensitive', dual=False, verbose=2)
            regr0.fit(self.dataset.training_total_features, self.dataset.training_total_labels[:, 0])

            regr1 = RandomForestRegressor(n_estimators=10, random_state=0, verbose=2)
            # regr1 = LinearSVR(random_state=0, loss='squared_epsilon_insensitive', dual=False, verbose=2)
            regr1.fit(self.dataset.training_total_features, self.dataset.training_total_labels[:, 1])

            predictions_all_0 = regr0.predict(self.dataset.training_total_features)
            predictions_all_1 = regr1.predict(self.dataset.training_total_features)

            Y_all = self.dataset.training_total_labels
            Y_all = np.reshape(Y_all, (-1, np.shape(Y_all)[-1]))
            predictions_all_0 = np.reshape(predictions_all_0, (np.shape(predictions_all_0)[-1], -1))
            predictions_all_1 = np.reshape(predictions_all_1, (np.shape(predictions_all_1)[-1], -1))
            predictions_all = np.concatenate((predictions_all_0, predictions_all_1), axis=1)

            return Y_all, predictions_all


    def predict(self):
        model = self.load_model_from_file()
        # model = LSTM_MIMO(num_t_x=num_t_x, num_input_dims=88, num_states=64, batch_size=self.FLAGS.train_batch_size)


        model_json_file_addr, model_h5_file_addr = self.generate_pathes()
        # load weights into new model
        model.load_weights(model_h5_file_addr)

        # generator = self.dataset.generate_sequencial_batch_data(category='validation',
        #                                                         num_t_x=num_t_x,
        #                                                         overlap=0.0,
        #                                                         batch_size=self.FLAGS.test_batch_size,
        #                                                         input_shape=self.input_shape)
        # Y_all = []
        # predictions_all = []
        # for i in range(5): # range(int(self.dataset.num_testing_data/num_t_x/self.FLAGS.test_batch_size)):
        #     X, Y = generator.next()
        #     predictions = model.predict_on_batch(X)
        #     Y_all.append(Y)
        #     predictions_all.append(predictions)
        predictions_all = model.predict(self.dataset.validation_total_features, batch_size=256, verbose=0)
        # predictions_all = model.predict(a11, batch_size=256, verbose=0)
        # predictions_all_all = predictions_all * b_std + b_mean
        # plt.plot(predictions_all_all)

        Y_all = self.dataset.validation_total_labels

        Y_all = np.reshape(Y_all, (-1, np.shape(Y_all)[-1]))
        predictions_all = np.reshape(predictions_all, (-1, np.shape(predictions_all)[-1]))

        return Y_all, predictions_all
