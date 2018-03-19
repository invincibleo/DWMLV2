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
from keras.regularizers import l2
from core.Models import *
from core.Learner import Learner
from core.Metrics import *

num_t_x = 10

class LearnerLSTMReg(Learner):
    def learn(self):
        model_json_file_addr, model_h5_file_addr = self.generate_pathes()
        continue_training = False
        if not os.path.exists(model_json_file_addr) or continue_training:
            self.copy_configuration_code()  # copy the configuration code so that known in which condition the model is trained

            # expected input data shape: (batch_size, timesteps, data_dim)

            model = SoundNet()
            model.add(LSTM(128, stateful=True, dropout=0.8))
            model.add(Dense(2, activation='linear', activity_regularizer=l2(0.0001)))

            if continue_training:
                model.load_weights("tmp/model/" + self.hash_name_hashed + "/model.h5")  # load weights into new model

            # Compile model
            model.compile(loss='mean_squared_error',
                          optimizer=keras.optimizers.Adam(lr=self.FLAGS.learning_rate,
                                                          beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0),
                          metrics=[CCC, 'mae'])

            # callbacks
            if tf.gfile.Exists('tmp/logs/tensorboard/' + str(self.hash_name_hashed)):
                shutil.rmtree('tmp/logs/tensorboard/' + str(self.hash_name_hashed))
            tensorboard = keras.callbacks.TensorBoard(
                log_dir='tmp/logs/tensorboard/' + str(self.hash_name_hashed),
                histogram_freq=100, write_grads=True, write_graph=False, write_images=False, batch_size=self.FLAGS.train_batch_size)
            model_check_point = keras.callbacks.ModelCheckpoint(
                filepath='tmp/model/' + str(self.hash_name_hashed) + '/checkpoints/' + 'weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                save_best_only=True,
                save_weights_only=True,
                mode='min')
            reduce_lr_on_plateau = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                                     factor=0.1,
                                                                     patience=10,
                                                                     epsilon=0.0005)
            def schedule(epoch):
                if epoch < 100:
                    lrate = 0.001
                elif epoch > 100 and epoch < 450:
                    lrate = 0.0001
                else:
                    lrate = 0.00005
                print("Epoch: " + str(epoch + 1) + " Learning rate: " + str(lrate) + "\n")
                return lrate

            learning_rate_schedule = keras.callbacks.LearningRateScheduler(schedule=schedule)

            model.summary()
            for i in range(500):
                lr = schedule(i)
                model.compile(loss='mean_squared_error',
                              optimizer=keras.optimizers.Adam(lr=lr,
                                                              beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0),
                              metrics=[CCC, 'mae'])
                hist = model.fit(self.dataset.training_total_features, self.dataset.training_total_labels,
                                 batch_size=self.FLAGS.train_batch_size,
                                 epochs=i+1,
                                 verbose=1,
                                 callbacks=[tensorboard],
                                 validation_data=(self.dataset.validation_total_features, self.dataset.validation_total_labels),
                                 shuffle=False,
                                 initial_epoch=i)
                model.reset_states()

            # save the model and training history
            self.save_model(hist, model)
        # else:
        #     # model = self.load_model_from_file()
        #     model = LSTM_MIMO(num_t_x=num_t_x, num_input_dims=88, num_states=64, batch_size=self.FLAGS.train_batch_size)
        #     model.load_weights(model_h5_file_addr)
        # return model

    def predict(self):
        model = self.load_model_from_file()
        # model = LSTM_MIMO(num_t_x=num_t_x, num_input_dims=88, num_states=64, batch_size=self.FLAGS.train_batch_size)


        model_json_file_addr, model_h5_file_addr = self.generate_pathes()
        # load weights into new model
        model.load_weights(model_h5_file_addr)

        predictions_all = model.predict(self.dataset.validation_total_features, batch_size=self.FLAGS.validation_batch_size, verbose=0)

        Y_all = self.dataset.validation_total_labels

        Y_all = np.reshape(Y_all, (-1, np.shape(Y_all)[-1]))
        predictions_all = np.reshape(predictions_all, (-1, np.shape(predictions_all)[-1]))

        return Y_all, predictions_all
