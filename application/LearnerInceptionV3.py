#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 11.08.17 23:22
# @Author  : Duowei Tang
# @Site    : https://iiw.kuleuven.be/onderzoek/emedia/people/phd-students/duoweitang
# @File    : LearnerInceptionV3
# @Software: PyCharm Community Edition
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras.layers as layers
import hashlib
from keras.models import Model
from keras.layers import (
    Input,
    Dense,
    Flatten,
    merge,
    Lambda,
    Dropout,
    GlobalAveragePooling2D,
    GlobalMaxPooling2D,
    BatchNormalization,
    Activation
)
from keras.layers.convolutional import (
    Conv2D,
    Convolution2D,
    MaxPooling2D,
    AveragePooling2D,
    ZeroPadding2D
)
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
import keras.backend as K


# Evidently this model breaks Python's default recursion limit
# This is a theano issue
import sys
import re
import os
import keras
import shutil

from keras.models import Sequential
from keras.models import model_from_json
import keras.backend as K

from core.Learner import Learner
from core.Metrics import *

from keras.engine.topology import get_source_inputs
from keras.utils.data_utils import get_file
import warnings

WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.5/inception_v3_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.5/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'


def conv2d_bn(x,
              filters,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1),
              name=None):
    """Utility function to apply conv + BN.
    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.
    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    if K.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3
    x = Conv2D(
        filters, (num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=True,
        kernel_initializer=keras.initializers.he_uniform(),
        bias_initializer=keras.initializers.Zeros(),
        name=conv_name)(x)
    x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    x = Activation('relu', name=name)(x)
    return x


def InceptionV3(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                num_second_last_layer=512,
                drop_out_rate=0.5,
                classes=1000):
    """Instantiates the Inception v3 architecture.
    Optionally loads weights pre-trained
    on ImageNet. Note that when using TensorFlow,
    for best performance you should set
    `image_data_format="channels_last"` in your Keras config
    at ~/.keras/keras.json.
    The model and the weights are compatible with both
    TensorFlow and Theano. The data format
    convention used by the model is the one
    specified in your Keras config file.
    Note that the default input image size for this model is 299x299.
    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization)
            or "imagenet" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(299, 299, 3)` (with `channels_last` data format)
            or `(3, 299, 299)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 139.
            E.g. `(150, 150, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    # input_shape = _obtain_input_shape(
    #     input_shape,
    #     default_size=299,
    #     min_size=139,
    #     data_format=K.image_data_format(),
    #     include_top=include_top)
    # input_shape = (101, 40, 1)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3

    # x = conv2d_bn(img_input, 32, 3, 3, strides=(2, 2), padding='valid')
    # x = conv2d_bn(x, 32, 3, 3, padding='valid')
    # x = conv2d_bn(x, 64, 3, 3)
    # x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv2d_bn(img_input, 80, 1, 1, padding='valid')  #### x img_input
    x = conv2d_bn(x, 192, 3, 3, padding='valid')
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # mixed 0, 1, 2: 35 x 35 x 256
    branch1x1 = conv2d_bn(x, 64, 1, 1)
    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 32, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed0')

    # mixed 1: 35 x 35 x 256
    branch1x1 = conv2d_bn(x, 64, 1, 1)
    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed1')

    # mixed 2: 35 x 35 x 256
    branch1x1 = conv2d_bn(x, 64, 1, 1)
    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed2')

    # mixed 3: 17 x 17 x 768
    branch3x3 = conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding='valid')
    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(
        branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = layers.concatenate(
        [branch3x3, branch3x3dbl, branch_pool], axis=channel_axis, name='mixed3')

    # mixed 4: 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192, 1, 1)

    branch7x7 = conv2d_bn(x, 128, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 128, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2d_bn(x, 128, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=channel_axis,
        name='mixed4')

    # mixed 5, 6: 17 x 17 x 768
    for i in range(2):
        branch1x1 = conv2d_bn(x, 192, 1, 1)

        branch7x7 = conv2d_bn(x, 160, 1, 1)
        branch7x7 = conv2d_bn(branch7x7, 160, 1, 7)
        branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

        branch7x7dbl = conv2d_bn(x, 160, 1, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 1, 7)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

        branch_pool = AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool],
            axis=channel_axis,
            name='mixed' + str(5 + i))

    # mixed 7: 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192, 1, 1)

    branch7x7 = conv2d_bn(x, 192, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 192, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2d_bn(x, 192, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=channel_axis,
        name='mixed7')

    # mixed 8: 8 x 8 x 1280
    branch3x3 = conv2d_bn(x, 192, 1, 1)
    branch3x3 = conv2d_bn(branch3x3, 320, 3, 3,
                          strides=(2, 2), padding='valid')

    branch7x7x3 = conv2d_bn(x, 192, 1, 1)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 1, 7)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 7, 1)
    branch7x7x3 = conv2d_bn(
        branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = layers.concatenate(
        [branch3x3, branch7x7x3, branch_pool], axis=channel_axis, name='mixed8')

    # mixed 9: 8 x 8 x 2048
    for i in range(2):
        branch1x1 = conv2d_bn(x, 320, 1, 1)

        branch3x3 = conv2d_bn(x, 384, 1, 1)
        branch3x3_1 = conv2d_bn(branch3x3, 384, 1, 3)
        branch3x3_2 = conv2d_bn(branch3x3, 384, 3, 1)
        branch3x3 = layers.concatenate(
            [branch3x3_1, branch3x3_2], axis=channel_axis, name='mixed9_' + str(i))

        branch3x3dbl = conv2d_bn(x, 448, 1, 1)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 384, 3, 3)
        branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, 1, 3)
        branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, 3, 1)
        branch3x3dbl = layers.concatenate(
            [branch3x3dbl_1, branch3x3dbl_2], axis=channel_axis)

        branch_pool = AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch3x3, branch3x3dbl, branch_pool],
            axis=channel_axis,
            name='mixed' + str(9 + i))
    if include_top:
        # Classification block
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        x = Dense(num_second_last_layer, activation='relu', use_bias=True,
                  kernel_initializer=keras.initializers.he_uniform(),
                  bias_initializer=keras.initializers.zeros(), name='2ndLastPrediction')(x)   #####change softmax to sigmoid
        x = Dropout(rate=drop_out_rate)(x)   ####### added by me
        x = Dense(classes, activation='sigmoid', use_bias=True,
                  kernel_initializer=keras.initializers.he_uniform(),
                  bias_initializer=keras.initializers.zeros(), name='predictions')(x)   #####change softmax to sigmoid

    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='inception_v3')

    # load weights
    if weights == 'imagenet':
        if K.image_data_format() == 'channels_first':
            if K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image data format convention '
                              '(`image_data_format="channels_first"`). '
                              'For best performance, set '
                              '`image_data_format="channels_last"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')
        if include_top:
            weights_path = get_file(
                'inception_v3_weights_tf_dim_ordering_tf_kernels.h5',
                WEIGHTS_PATH,
                cache_subdir='models',
                md5_hash='9a0d58056eeedaa3f26cb7ebd46da564')
        else:
            weights_path = get_file(
                'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5',
                WEIGHTS_PATH_NO_TOP,
                cache_subdir='models',
                md5_hash='bcbd6486424b2319ff4ef7d526e38f63')
        model.load_weights(weights_path, by_name=True)
    return model

class LearnerInceptionV3(Learner):
    def learn(self):
        self.hash_name_hashed = hashlib.sha1(tf.compat.as_bytes(self.FLAGS.__str__())).hexdigest()
        print(self.FLAGS.__str__())
        model_json_file_addr = "tmp/model/" + str(self.hash_name_hashed) + "/model.json"
        model_h5_file_addr = "tmp/model/" + str(self.hash_name_hashed) + "/model.h5"

        continue_training = False
        if not os.path.exists(model_json_file_addr) or continue_training:

            if not os.path.exists("tmp/model/" + str(self.hash_name_hashed)):
                os.makedirs("tmp/model/" + str(self.hash_name_hashed))
                os.makedirs('tmp/model/' + str(self.hash_name_hashed) + '/checkpoints/')
                shutil.copytree('../application/', 'tmp/model/' + str(self.hash_name_hashed) + '/application/')

            # model = Sequential()
            # model.add(Dense(43, input_dim=96*20, activation='sigmoid'))#InceptionV3(weights=None, classes=527)
            num_classes = self.dataset.num_classes
            time_length = int(self.FLAGS.time_resolution/0.02) + 1
            input_shape = (time_length, 40, 1)
            model = InceptionV3(input_shape=input_shape, weights=None,
                                classes=num_classes,
                                num_second_last_layer=self.FLAGS.num_second_last_layer,
                                drop_out_rate=self.FLAGS.drop_out_rate)

            if continue_training:
                # load weights into new model
                model.load_weights("tmp/model/" + self.hash_name_hashed + "/model.h5")

            # Compile model
            model.compile(loss='categorical_crossentropy',
                          optimizer=keras.optimizers.Adam(lr=self.FLAGS.learning_rate,
                                                          beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0),
                          metrics=['categorical_accuracy', top3_accuracy, 'top_k_categorical_accuracy'])  # top3_accuracy accuracy 'categorical_crossentropy' 'categorical_accuracy' multiclass_loss

            # if tf.gfile.Exists('tmp/logs/tensorboard/' + str(self.hash_name_hashed)):
            #     shutil.rmtree('tmp/logs/tensorboard/' + str(self.hash_name_hashed))

            tensorboard = keras.callbacks.TensorBoard(
                log_dir='tmp/logs/tensorboard/' + str(self.hash_name_hashed),
                histogram_freq=10, write_graph=True, write_images=True)

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
        input_shape = (time_length, 40, 1)

        # load json and create model
        json_file = open("tmp/model/" + self.hash_name_hashed + "/model.json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights("tmp/model/" + self.hash_name_hashed + "/model.h5")
        # model.load_weights("tmp/model/" + self.hash_name_hashed + "/checkpoints/weights.128-1.05.hdf5")

        print("Loaded model from disk")

        (X, Y, data_point_list) = self.dataset.get_batch_data(category='testing',
                                                              batch_size=self.dataset.num_testing_data,
                                                              input_shape=input_shape)
        predictions = model.predict_on_batch(X)
        return Y, predictions
