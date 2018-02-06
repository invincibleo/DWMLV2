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

from core.Models import *
from core.Learner import Learner
from core.Metrics import *


class LearnerLSTMReg(Learner):
    def learn(self):
        model_json_file_addr, model_h5_file_addr = self.generate_pathes()

        continue_training = False
        if not os.path.exists(model_json_file_addr) or continue_training:
            self.copy_configuration_code()  # copy the configuration code so that known in which condition the model is trained

            # expected input data shape: (batch_size, timesteps, data_dim)
            model = ResNet50(self.input_shape)

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
                                                                     patience=10,
                                                                     epsilon=0.0005)
            def schedule(epoch_num=100):
                if epoch_num <= 50:
                    learning_rate = 0.001
                elif epoch_num > 50 and epoch_num <= 100:
                    learning_rate = 0.0005
                else:
                    learning_rate = 0.0001
                return learning_rate
            learning_rate_schedule = keras.callbacks.LearningRateScheduler(schedule=schedule)

            training_generator = self.dataset.generate_batch_data(category='training', batch_size=self.FLAGS.train_batch_size, input_shape=self.input_shape)
            validation_generator = self.dataset.generate_batch_data(category='validation',
                                                                batch_size=self.FLAGS.validation_batch_size, input_shape=self.input_shape)
            model.summary()
            hist = model.fit_generator(
                generator=training_generator,
                steps_per_epoch=int(self.dataset.num_training_data/self.FLAGS.train_batch_size),
                # initial_epoch=100,
                epochs=150,
                callbacks=[tensorboard, learning_rate_schedule],
                validation_data=validation_generator,
                validation_steps=int(self.dataset.num_validation_data/self.FLAGS.train_batch_size),
                workers=20
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
        for i in range(3): # range(int(self.dataset.num_testing_data/self.FLAGS.test_batch_size)):
            X, Y = generator.next()
            predictions = model.predict_on_batch(X)
            Y_all.append(Y)
            predictions_all.append(predictions)

        # Y_all = np.squeeze(Y_all, 0)
        # predictions_all = np.squeeze(predictions_all, 0)
        Y_all = np.reshape(Y_all, (-1, np.shape(Y_all)[-1]))
        predictions_all = np.reshape(predictions_all, (-1, np.shape(predictions_all)[-1]))
        return Y_all, predictions_all
