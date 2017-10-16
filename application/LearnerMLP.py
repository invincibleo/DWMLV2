#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 30.08.17 14:00
# @Author  : Duowei Tang
# @Site    : https://iiw.kuleuven.be/onderzoek/emedia/people/phd-students/duoweitang
# @File    : LearnerMLP
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
from keras.layers import Dense, Dropout
from core.Learner import Learner
from core.Metrics import *


class LearnerMLP(Learner):
    def learn(self):
        model_json_file_addr, model_h5_file_addr = self.generate_pathes()

        continue_training = False
        if not os.path.exists(model_json_file_addr) or continue_training:
            # copy the configuration code so that known in which condition the model is trained
            self.copy_configuration_code()

            # model = Sequential()
            # model.add(Dense(43, input_dim=96*20, activation='sigmoid'))#InceptionV3(weights=None, classes=527)
            num_classes = self.dataset.num_classes
            time_length = int(self.FLAGS.time_resolution/0.02) + 1
            input_shape = (time_length*40,)
            model = Sequential()
            model.add(Dense(50, input_shape=input_shape, activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(50, activation='relu', kernel_initializer='uniform'))
            model.add(Dropout(0.2))
            model.add(Dense(num_classes, activation='sigmoid', kernel_initializer='uniform'))

            if continue_training:
                # load weights into new model
                self.load_model(model)

            # Compile model
            model.compile(loss='categorical_crossentropy',
                          optimizer=keras.optimizers.Adam(lr=self.FLAGS.learning_rate,
                                                          beta_1=0.9, beta_2=0.999, epsilon=1e-08),
                          metrics=['binary_accuracy', 'categorical_accuracy'])  # top3_accuracy accuracy 'categorical_crossentropy' 'categorical_accuracy' multiclass_loss

            if tf.gfile.Exists('tmp/logs/tensorboard/' + str(self.hash_name_hashed)):
                shutil.rmtree('tmp/logs/tensorboard/' + str(self.hash_name_hashed))

            tensorboard = keras.callbacks.TensorBoard(
                log_dir='tmp/logs/tensorboard/' + str(self.hash_name_hashed),
                histogram_freq=10, write_graph=True, write_images=True, write_grads=True, batch_size=100)

            model_check_point = keras.callbacks.ModelCheckpoint('tmp/model/' + str(self.hash_name_hashed) + '/checkpoints/' + 'weights.{epoch:02d}-{val_loss:.2f}.hdf5', save_best_only=True, save_weights_only=True, mode='min')

            hist = model.fit_generator(
                generator=self.dataset.generate_batch_data(category='training', batch_size=self.FLAGS.train_batch_size, input_shape=input_shape),
                steps_per_epoch=int(self.dataset.num_training_data/self.FLAGS.train_batch_size),
                # initial_epoch=100,
                epochs=10,
                callbacks=[tensorboard, model_check_point],
                validation_data=self.dataset.generate_batch_data(category='validation',
                                                                batch_size=self.FLAGS.validation_batch_size, input_shape=input_shape),
                validation_steps=int(self.dataset.num_validation_data/self.FLAGS.train_batch_size),
            )


            # tensorboard = keras.callbacks.TensorBoard(log_dir='tmp/logs/tensorboard/' + str(self.hash_name_hashed))
            # x, y, data_point_list = self.dataset.get_batch_data(category='training', batch_size=3000, input_shape=input_shape)
            # x_val, y_val, data_point_list_val = self.dataset.get_batch_data(category='validation',
            #                                                                 batch_size=self.FLAGS.validation_batch_size,
            #                                                                 input_shape=input_shape)
            # hist = model.fit(
            #     x=x,
            #     y=y,
            #     batch_size=100,
            #     epochs=10,  # 1000000
            #     validation_data=(x_val, y_val),
            #     verbose=2,
            #     callbacks=[tensorboard],
            # )

            # save the model and training history
            self.save_model(hist, model)
        else:
            model = self.load_model_from_file()

        return model

    def predict(self):
        num_classes = self.dataset.num_classes
        time_length = int(self.FLAGS.time_resolution / 0.02) + 1
        input_shape = (time_length * 40,)

        model = self.load_model_from_file()

        # (X, Y, data_point_list) = self.dataset.get_batch_data(category='testing',
        #                                                       batch_size=self.dataset.num_testing_data,
        #                                                       input_shape=input_shape)
        generator = self.dataset.generate_batch_data(category='testing',
                                                  batch_size=256,
                                                  input_shape=input_shape)
        Y_all = []
        predictions_all = []
        for i in range(int(self.dataset.num_testing_data/256)):
            X, Y = generator.next()
            predictions = model.predict_on_batch(X)
            Y_all.append(Y)
            predictions_all.append(predictions)

        Y_all = np.reshape(Y_all, (-1, self.dataset.num_classes))
        predictions_all = np.reshape(predictions_all, (-1, self.dataset.num_classes))

        return Y_all, predictions_all
