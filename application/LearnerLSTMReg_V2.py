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


# def get_dataset(num):
#     dataset = []
#     labels = []
#     for i in range(num):
#         # x = toeplitz(c=np.arange(i, i+10, 0.1), r=np.arange(i, i+20, 0.1))
#         x = np.arange(i, i + 10, 1).T
#         x = np.expand_dims(x, axis=1)
#         dataset.append(np.sin(x))
#         labels.append(np.sin(x))
#     return np.array(dataset), np.array(labels)[:, -1, :]

num_t_x = 10

# a, b = get_dataset(1000)
# a_mean = np.mean(a, axis=0)
# a_std = np.std(a, axis=0)
# b_mean = np.mean(b, axis=0)
# b_std = np.std(b, axis=0)
# aa = (a - a_mean) / a_std
# bb = (b - b_mean) / b_std
# a1, b1 = get_dataset(100)
# a11 = (a1 - a_mean) / a_std
# b11 = (b1 - b_mean) / b_std

class LearnerLSTMReg(Learner):
    def learn(self):
        model_json_file_addr, model_h5_file_addr = self.generate_pathes()
        continue_training = False
        if not os.path.exists(model_json_file_addr) or continue_training:
            self.copy_configuration_code()  # copy the configuration code so that known in which condition the model is trained

            # expected input data shape: (batch_size, timesteps, data_dim)


            # input_l = Input(shape=(num_t_x, 1))
            # # x = LSTM(num_t_x, stateful=False)(input_l)
            # x = keras.layers.LSTM(5, return_sequences=True)(input_l)
            # x = keras.layers.LSTM(5, return_sequences=True)(x)
            # x = keras.layers.LSTM(5)(x)
            # x = Dense(5, activation='relu')(x)
            # out = Dense(1, activation='linear')(x)
            # model = Model(input_l, out, "LSTM_reg_v2")

            # model = LSTM_MIMO(num_t_x=num_t_x, num_input_dims=88, num_states=64, batch_size=self.FLAGS.train_batch_size)

            model = DNN_LSTM_MIMO(num_t_x=num_t_x, num_input_dims=88, num_states=64)
            if continue_training:
                model.load_weights("tmp/model/" + self.hash_name_hashed + "/model.h5")  # load weights into new model

            # Compile model
            model.compile(loss='mean_squared_error',
                          optimizer=keras.optimizers.Adam(lr=self.FLAGS.learning_rate,
                                                          beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0),
                          metrics=['mae', 'mse'])

            # callbacks
            if tf.gfile.Exists('tmp/logs/tensorboard/' + str(self.hash_name_hashed)):
                shutil.rmtree('tmp/logs/tensorboard/' + str(self.hash_name_hashed))
            tensorboard = keras.callbacks.TensorBoard(
                log_dir='tmp/logs/tensorboard/' + str(self.hash_name_hashed),
                histogram_freq=10, write_graph=True, write_images=True)
            model_check_point = keras.callbacks.ModelCheckpoint(
                filepath='tmp/model/' + str(self.hash_name_hashed) + '/checkpoints/' + 'weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                save_best_only=True,
                save_weights_only=True,
                mode='min')
            reduce_lr_on_plateau = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                                     factor=0.1,
                                                                     patience=10,
                                                                     epsilon=0.0005)
            # def schedule(epoch_num):
            #     if epoch_num <= 30:
            #         learning_rate = 0.01
            #     elif epoch_num > 30 and epoch_num <= 60:
            #         learning_rate = 0.001
            #     elif epoch_num > 60 and epoch_num <= 90:
            #         learning_rate = 0.0005
            #     else:
            #         learning_rate = 0.0001
            #     return learning_rate
            def schedule(epoch):
                initial_lrate = 0.01
                drop = 0.5
                epochs_drop = 10.0
                lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
                return lrate

            learning_rate_schedule = keras.callbacks.LearningRateScheduler(schedule=schedule)

            # training_generator = self.dataset.generate_sequencial_batch_data(category='training', num_t_x=num_t_x,
            #                                                                  overlap=0.0,
            #                                                                  batch_size=self.FLAGS.train_batch_size,
            #                                                                  input_shape=self.input_shape)
            # validation_generator = self.dataset.generate_sequencial_batch_data(category='validation', num_t_x=num_t_x,
            #                                                                    overlap=0.0,
            #                                                                    batch_size=self.FLAGS.validation_batch_size,
            #                                                                    input_shape=self.input_shape)
            model.summary()
            # hist = model.fit_generator(
            #     generator=training_generator,
            #     steps_per_epoch=int(self.dataset.num_training_data/self.FLAGS.train_batch_size/num_t_x),
            #     # initial_epoch=100,
            #     epochs=15,
            #     callbacks=[tensorboard, learning_rate_schedule],
            #     validation_data=validation_generator,
            #     validation_steps=int(self.dataset.num_validation_data/self.FLAGS.train_batch_size/num_t_x),
            #     max_q_size=30,
            #     workers=20
            # )
            # hist = model.fit(aa, bb,
            #                  batch_size=self.FLAGS.train_batch_size,
            #                  epochs=1000000,
            #                  verbose=1,
            #                  callbacks=[tensorboard, learning_rate_schedule],
            #                  validation_data=(a11, b11),
            #                  shuffle=True)
            hist = model.fit(self.dataset.training_total_features, self.dataset.training_total_labels,
                             batch_size=self.FLAGS.train_batch_size,
                             epochs=50,
                             verbose=1,
                             callbacks=[tensorboard, learning_rate_schedule],
                             validation_data=(self.dataset.validation_total_features, self.dataset.validation_total_labels),
                             shuffle=True)

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
