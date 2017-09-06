#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 25.08.17 11:02
# @Author  : Duowei Tang
# @Site    : https://iiw.kuleuven.be/onderzoek/emedia/people/phd-students/duoweitang
# @File    : Dataset
# @Software: PyCharm Community Edition
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import json
import hashlib
import tensorflow as tf
import numpy as np


class Dataset(object):

    def __init__(self, *args, **kwargs):
        self.data_list = kwargs.get('data_list', None)
        self.dataset_name = kwargs.get('dataset_name', "")
        self.dataset_dir = kwargs.get('dataset_dir', "")
        self.feature_dir = kwargs.get('feature_dir', "")
        self.dataset_list_dir = kwargs.get('dataset_list_dir', "")
        self.num_classes = int(kwargs.get('num_classes', 0))
        self.label_list = kwargs.get('label_list', [])
        self.FLAGS = kwargs.get('flag', '')
        self.preprocessing_methods = kwargs.get('preprocessing_methods', [])
        self.normalization = kwargs.get('normalization', True)
        self.training_mean = 0.0
        self.training_std = 1.0
        self.num_training_data = kwargs.get('num_training_data', 0)
        self.num_validation_data = kwargs.get('num_validation_data', 0)
        self.num_testing_data = kwargs.get('num_testing_data', 0)

        self.feature_parameter_dir = kwargs.get('feature_parameter_dir', "")
        if self.feature_parameter_dir == "":
            self.feature_parameter_dir = self.FLAGS.parameter_dir
        self.feature_parameters = json.load(open(self.feature_parameter_dir + '/feature_parameters.json', 'r'))

        if self.dataset_list_dir == "":
            self.dataset_list_dir = os.path.join('tmp/dataset/', self.dataset_name)

        if self.feature_dir == "":
            self.feature_dir = os.path.join('tmp/feature/', self.dataset_name)


    def get_dataset_file_addr(self):
        file_name = "coding-" + self.FLAGS.coding + "_timeRes-" + str(self.FLAGS.time_resolution) + ".pickle"
        if not tf.gfile.Exists(self.dataset_list_dir):
            os.makedirs(self.dataset_list_dir)
        return os.path.join(self.dataset_list_dir, file_name)

    def get_feature_file_addr(self, sub_dir, data_name):
        hash_parameters = hashlib.sha1(str(self.feature_parameters)).hexdigest()
        feature_with_parameters_dir = os.path.join(self.feature_dir,
                                                   'timeRes-' + str(self.FLAGS.time_resolution) + '-' + hash_parameters)
        feature_parameter_file_name = 'feature_parameters.json'
        tf.gfile.MakeDirs(feature_with_parameters_dir)
        tf.gfile.Copy(oldpath=os.path.join(self.feature_parameter_dir, feature_parameter_file_name),
                      newpath=os.path.join(feature_with_parameters_dir, feature_parameter_file_name),
                      overwrite=True)
        return os.path.join(feature_with_parameters_dir, sub_dir, data_name.split('.')[0] + '.pickle')

    def online_mean_variance(self, new_batch_data):
        self.num_training_data = 0
        mean = np.zeros((1, np.shape(new_batch_data)[2]))
        M2 = np.zeros((1, np.shape(new_batch_data)[2]))

        for data_point in new_batch_data:
            x = data_point
            self.num_training_data += 1
            delta = x - mean
            mean += delta / self.num_training_data
            delta2 = x - mean
            M2 += delta * delta2

        self.training_mean = mean
        if self.num_training_data < 2:
            self.training_std = float('nan')
        else:
            self.training_std = np.sqrt(M2 / self.num_training_data)

    def generate_batch_data(self, category, batch_size=100, input_shape=(1, -1)):
        X = []
        Y = []
        if category == 'training':
            working_list = self.data_list['training']
        elif category == 'validation':
            working_list = self.data_list['validation']
        elif category == 'testing':
            working_list = self.data_list['testing']

        num_data_files = len(working_list)
        random_perm = np.random.permutation(num_data_files)
        while (1):
            for i in range(0, num_data_files):
                data_idx = random_perm[i]           #random.randrange(num_data_files)
                data_point = working_list[data_idx]
                data_name = data_point.data_name
                sub_dir = data_point.sub_dir
                label_content = data_point.label_content
                feature_idx = data_point.feature_idx

                feature_file_addr = os.path.join(self.feature_dir, 'time_res' + str(self.FLAGS.time_resolution),
                                                 sub_dir, data_name.split('.')[0] + '.pickle')
                features = pickle.load(open(feature_file_addr, 'rb'))

                feature = features[feature_idx]
                # if normalization then mean and std would not be 0 and 1 separately
                feature = (feature - self.training_mean) / self.training_std
                feature = np.reshape(feature, input_shape)

                if not len(X) and not len(Y):
                    X = np.expand_dims(feature, axis=0)
                    Y = label_content #np.expand_dims(label_content, axis=0)
                else:
                    X = np.append(X, np.expand_dims(feature, axis=0), 0)
                    Y = np.append(Y, label_content, 0)          #np.expand_dims(label_content, axis=0)

                if X.shape[0] >= batch_size:
                    yield (X, Y)
                    X = []
                    Y = []

    def get_batch_data(self, category, batch_size=100, input_shape=(1, -1)):
        X = []
        Y = []
        if category == 'training':
            working_list = self.data_list['training']
        elif category == 'validation':
            working_list = self.data_list['validation']
        elif category == 'testing':
            working_list = self.data_list['testing']

        num_data_files = len(working_list)
        data_point_list = []
        random_perm = np.random.permutation(num_data_files)
        while (1):
            for i in range(0, num_data_files):
                data_idx = random_perm[i]
                data_point = working_list[data_idx]
                data_point_list.append(data_point)
                data_name = data_point.data_name
                sub_dir = data_point.sub_dir
                label_content = data_point.label_content
                feature_idx = data_point.feature_idx

                feature_file_addr = os.path.join(self.feature_dir, 'time_res' + str(self.FLAGS.time_resolution),
                                                 sub_dir, data_name.split('.')[0] + '.pickle')
                features = pickle.load(open(feature_file_addr, 'rb'))

                feature = features[feature_idx]
                # if normalization then mean and std would not be 0 and 1 separately
                feature = (feature - self.training_mean) / self.training_std
                feature = np.reshape(feature, input_shape)

                if not len(X) and not len(Y):
                    X = np.expand_dims(feature, axis=0)
                    Y = label_content #np.expand_dims(label_content, axis=0)
                else:
                    X = np.append(X, np.expand_dims(feature, axis=0), 0)
                    Y = np.append(Y, label_content, 0)

                if X.shape[0] >= batch_size:
                    return (X, Y, data_point_list)