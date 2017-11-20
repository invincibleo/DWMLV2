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

from keras.models import model_from_json, Sequential
from keras.layers import Dense, Dropout, LSTM
from core.Learner import Learner
from core.Metrics import *


class LearnerLSTMReg(Learner):
    def model(self, input_shape):
        model = Sequential()
        model.add(LSTM(32, activation='tanh', return_sequences=True, input_shape=input_shape,
                       dropout=0.8))  # returns a sequence of vectors of dimension 32
        model.add(LSTM(32, activation='tanh', return_sequences=True, dropout=0.8))
        model.add(Dense(2, activation='tanh',
                        kernel_initializer=keras.initializers.he_normal(),
                        bias_initializer=keras.initializers.zeros()))
        return model

    def learn(self):
        model_json_file_addr, model_h5_file_addr = self.generate_pathes()

        continue_training = False
        if not os.path.exists(model_json_file_addr) or continue_training:
            self.copy_configuration_code()  # copy the configuration code so that known in which condition the model is trained

            # expected input data shape: (batch_size, timesteps, data_dim)
            model = self.model(self.input_shape)

            if continue_training:
                model.load_weights("tmp/model/" + self.hash_name_hashed + "/model.h5")  # load weights into new model

            # Compile model
            model.compile(loss='mean_absolute_error',
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
                                                                     patience=5,
                                                                     epsilon=0.0005)

            hist = model.fit_generator(
                generator=self.dataset.generate_batch_data(category='training', batch_size=self.FLAGS.train_batch_size, input_shape=self.input_shape),
                steps_per_epoch=int(self.dataset.num_training_data/self.FLAGS.train_batch_size),
                # initial_epoch=100,
                epochs=1,
                callbacks=[tensorboard, reduce_lr_on_plateau],
                validation_data=self.dataset.generate_batch_data(category='validation',
                                                                batch_size=self.FLAGS.validation_batch_size, input_shape=self.input_shape),
                validation_steps=int(self.dataset.num_validation_data/self.FLAGS.train_batch_size),
            )

            # save the model and training history
            self.save_model(hist, model)
        else:
            model = self.load_model_from_file()

        return model

    def predict(self):
        model = self.load_model_from_file()

        generator = self.dataset.generate_batch_data(category='validation',
                                                  batch_size=self.FLAGS.test_batch_size,
                                                  input_shape=self.input_shape)
        Y_all = []
        predictions_all = []
        for i in range(int(self.dataset.num_testing_data/self.FLAGS.test_batch_size)):
            X, Y = generator.next()
            predictions = model.predict_on_batch(X)
            Y_all.append(Y)
            predictions_all.append(predictions)

        Y_all = np.reshape(Y_all, (-1, self.dataset.num_classes))
        predictions_all = np.reshape(predictions_all, (-1, self.dataset.num_classes))

        return Y_all, predictions_all
