#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 14/09/2017 3:25 PM
# @Author  : Duowei Tang
# @Site    : https://iiw.kuleuven.be/onderzoek/emedia/people/phd-students/duoweitang
# @File    : LearnerLSTM
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


class LearnerLSTM(Learner):
    def learn(self):
        self.hash_name_hashed = hashlib.sha1(tf.compat.as_bytes(self.FLAGS.__str__())).hexdigest()
        print(self.FLAGS.__str__())
        model_json_file_addr = "tmp/model/" + str(self.hash_name_hashed) + "/model.json"
        model_h5_file_addr = "tmp/model/" + str(self.hash_name_hashed) + "/model.h5"

        if not os.path.exists(model_json_file_addr):

            if not os.path.exists("tmp/model/" + str(self.hash_name_hashed)):
                os.makedirs("tmp/model/" + str(self.hash_name_hashed))
                os.makedirs('tmp/model/' + str(self.hash_name_hashed) + '/checkpoints/')
                shutil.copytree('../../application/', 'tmp/model/' + str(self.hash_name_hashed) + '/application/')

            num_classes = self.dataset.num_classes
            time_length = int(self.FLAGS.time_resolution/0.02) + 1
            input_shape = (time_length, -1)

            # expected input data shape: (batch_size, timesteps, data_dim)
            model = Sequential()
            model.add(LSTM(32, return_sequences=True, input_shape=input_shape))  # returns a sequence of vectors of dimension 32
            model.add(LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
            model.add(LSTM(32))  # return a single vector of dimension 32
            model.add(Dense(num_classes, activation='sigmoid',
                            kernel_regularizer=keras.regularizers.l2(0.01),
                            kernel_initializer=keras.initializers.he_normal(),
                            bias_initializer=keras.initializers.zeros()))

            # Compile model
            model.compile(loss='categorical_crossentropy',
                          optimizer=keras.optimizers.Adam(lr=self.FLAGS.learning_rate,
                                                          beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0),
                          metrics=['binary_accuracy', 'categorical_accuracy', top3_accuracy, 'top_k_categorical_accuracy'])  # top3_accuracy accuracy 'categorical_crossentropy' 'categorical_accuracy' multiclass_loss

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
                epochs=50,
                callbacks=[tensorboard, model_check_point],
                validation_data=self.dataset.generate_batch_data(category='validation',
                                                                batch_size=self.FLAGS.validation_batch_size, input_shape=input_shape),
                validation_steps=int(self.dataset.num_validation_data/self.FLAGS.train_batch_size),
            )

            # Saving the objects:
            with open('tmp/model/objs.txt', 'wb') as histFile:  # Python 3: open(..., 'wb')
                # pickle.dump([hist, model], f)
                for key, value in hist.history.iteritems():
                    histFile.write(key + '-' + ','.join([str(x) for x in value]))
                    histFile.write('\n')


            # serialize model to JSON
            model_json = model.to_json()
            with open(model_json_file_addr, "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            model.save_weights(model_h5_file_addr)
            print("Saved model to disk")
        else:
            # load json and create model
            json_file = open(model_json_file_addr, 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            model = model_from_json(loaded_model_json)
            # load weights into new model
            model.load_weights(model_h5_file_addr)
            print("Loaded model from disk")

        return model

    def predict(self):
        num_classes = self.dataset.num_classes
        time_length = int(self.FLAGS.time_resolution / 0.02) + 1
        input_shape = (time_length * 40,)

        # load json and create model
        json_file = open("tmp/model/" + self.hash_name_hashed + "/model.json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights("tmp/model/" + self.hash_name_hashed + "/model.h5")
        print("Loaded model from disk")

        (X, Y, data_point_list) = self.dataset.get_batch_data(category='testing',
                                                              batch_size=self.dataset.num_testing_data,
                                                              input_shape=input_shape)
        predictions = model.predict_on_batch(X)
        return Y, predictions