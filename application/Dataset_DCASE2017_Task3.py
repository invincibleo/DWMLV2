#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 25.08.17 11:22
# @Author  : Duowei Tang
# @Site    : https://iiw.kuleuven.be/onderzoek/emedia/people/phd-students/duoweitang
# @File    : Dataset_DCASE2017_Task3
# @Software: PyCharm Community Edition

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import hashlib
import tensorflow as tf
import numpy as np
import math
import pickle
import random

from core.Dataset import Dataset
from core.GeneralFileAccessor import GeneralFileAccessor
from core.TimeseriesPoint import *
from core.Preprocessing import *
from core.util import *

PICKLE_FILE_ADDR = './tmp/dataset_DCASE2017task3.pickle'
FEATURE_DIR_ADDR = 'tmp/feature/DCASE2017'


class Dataset_DCASE2017_Task3(Dataset):

    def __init__(self, *args, **kwargs):
        super(Dataset_DCASE2017_Task3, self).__init__(self, *args, **kwargs)
        self.dataset_name = 'Dataset_DCASE2017_Task3'
        self.label_list = sorted(['people walking', 'car', 'large vehicle', 'people speaking', 'brakes squeaking', 'children'])
        self.encoding = kwargs.get('encoding', 'khot')
        self.num_classes = len(self.label_list)
        self.data_list = self.create_data_list()

    @staticmethod
    def audio_event_roll(meta_file, label_list, time_resolution):
        meta_file_content = GeneralFileAccessor(file_path=meta_file).read()
        end_times = np.array([float(x[1]) for x in meta_file_content])
        max_offset_value = np.max(end_times, 0)

        event_roll = np.zeros((int(math.floor(max_offset_value / time_resolution)), len(label_list)))
        start_times = []
        end_times = []

        for line in meta_file_content:
            label_name = line[-1]
            label_idx = label_list.index(label_name)
            event_start = float(line[0])
            event_end = float(line[1])

            onset = int(math.floor(event_start / time_resolution))
            offset = int(math.floor(event_end / time_resolution))

            event_roll[onset:offset, label_idx] = 1
            start_times.append(event_start)
            end_times.append(event_end)

        return event_roll

    def feature_extraction(self, audio_raw):
        # feature extraction
        audio_raw = np.reshape(audio_raw, (1, -1))
        preprocessing = Preprocessing()
        preprocessing_func = audio_raw
        for preprocessing_method in self.preprocessing_methods:
            preprocessing_func = eval('preprocessing.' + preprocessing_method)(preprocessing_func)

        return preprocessing_func

    def create_data_list(self):
        pickle_file = 'tmp/dataset/DCASE2017_onehot.pickle'
        if not tf.gfile.Exists(pickle_file) or not tf.gfile.Exists(FEATURE_DIR_ADDR):
            individual_meta_file_base_addr = os.path.join(self.dataset_dir, 'meta/street/')
            audio_file_list = [x[:-4] for x in tf.gfile.ListDirectory(individual_meta_file_base_addr)]

            data_list = {'validation': [], 'testing': [], 'training': []}
            for audio_file in audio_file_list:
                audio_meta_file_addr = os.path.join(individual_meta_file_base_addr, audio_file + '.ann')
                audio_file_addr = os.path.join(os.path.join(self.dataset_dir, 'audio/street/'), audio_file + '.wav')
                audio_raw_all, fs = GeneralFileAccessor(file_path=audio_file_addr,
                                                        mono=True).read()
                event_roll = Dataset_DCASE2017_Task3.audio_event_roll(meta_file=audio_meta_file_addr,
                                                                      time_resolution=self.FLAGS.time_resolution,
                                                                      label_list=self.label_list)

                feature_file_addr = os.path.join(FEATURE_DIR_ADDR, 'street', audio_file + '.pickle')
                if not tf.gfile.Exists(feature_file_addr):
                    feature_base_addr = '/'.join(feature_file_addr.split('/')[:-1])
                    if not tf.gfile.Exists(feature_base_addr):
                        os.makedirs(feature_base_addr)
                    save_features = True
                    features = []
                else:
                    save_features = False
                    features = pickle.load(open(feature_file_addr, 'rb'))

                if self.encoding == 'khot':
                    for point_idx in range(event_roll.shape[0]):
                        label_name = ','.join(np.array(self.label_list)[np.array(event_roll[point_idx], dtype=bool)])
                        start_time = point_idx * self.FLAGS.time_resolution
                        end_time = (point_idx + 1) * self.FLAGS.time_resolution
                        new_point = AudioPoint(
                            data_name=audio_file + '.wav',
                            sub_dir='street',
                            label_name=label_name,
                            label_content=np.reshape(event_roll[point_idx], (1, -1)),
                            extension='wav',
                            fs=fs,
                            feature_idx=point_idx,
                            start_time=start_time,
                            end_time=end_time
                        )

                        hash_name_hashed = hashlib.sha1(tf.compat.as_bytes(audio_file + str(point_idx))).hexdigest()
                        percentage_hash = int(hash_name_hashed, 16) % (100 + 1)

                        if percentage_hash < self.FLAGS.validation_percentage:
                            data_list['validation'].append(new_point)
                        elif percentage_hash < (self.FLAGS.testing_percentage + self.FLAGS.validation_percentage):
                            data_list['testing'].append(new_point)
                        else:
                            data_list['training'].append(new_point)

                        if save_features:
                            # feature extraction
                            audio_raw = audio_raw_all[start_time * fs:end_time * fs]
                            feature = self.feature_extraction(audio_raw)
                            features.append(np.reshape(feature, (1, -1)))

                elif self.encoding == 'onehot':
                    for point_idx in range(event_roll.shape[0]):
                        for idx in range(self.num_classes):
                            label_content = np.zeros((1, self.num_classes))
                            if event_roll[point_idx][idx] == 0:
                                label_name = ''
                            else:
                                label_name = self.label_list[idx]
                                label_content[0, idx] = 1

                            start_time = point_idx * self.FLAGS.time_resolution
                            end_time = (point_idx + 1) * self.FLAGS.time_resolution
                            new_point = AudioPoint(
                                            data_name=audio_file+'.wav',
                                            sub_dir='street',
                                            label_name=label_name,
                                            label_content=np.reshape(label_content, (1, -1)),
                                            extension='wav',
                                            fs=fs,
                                            feature_idx=point_idx,
                                            start_time=start_time,
                                            end_time=end_time
                                        )

                            hash_name_hashed = hashlib.sha1(tf.compat.as_bytes(audio_file + str(point_idx) + str(idx))).hexdigest()
                            percentage_hash = int(hash_name_hashed, 16) % (100 + 1)

                            if percentage_hash < self.FLAGS.validation_percentage:
                                data_list['validation'].append(new_point)
                            elif percentage_hash < (self.FLAGS.testing_percentage + self.FLAGS.validation_percentage):
                                data_list['testing'].append(new_point)
                            else:
                                data_list['training'].append(new_point)

                        if save_features:
                            # feature extraction
                            audio_raw = audio_raw_all[start_time * fs:end_time * fs]
                            feature = self.feature_extraction(audio_raw)
                            features.append(np.reshape(feature, (1, -1)))

                if save_features:
                    pickle.dump(features, open(feature_file_addr, 'wb'), 2)

            if not tf.gfile.Exists("tmp/dataset"):
                os.makedirs("tmp/dataset")
            pickle.dump(data_list, open(pickle_file, 'wb'), 2)
        else:
            data_list = pickle.load(open(pickle_file, 'rb'))
        return data_list

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
        while (1):
            data_idx = random.randrange(num_data_files)
            data_point = working_list[data_idx]
            data_name = data_point.data_name
            sub_dir = data_point.sub_dir
            label_content = data_point.label_content
            # feature_name = self.preprocessing_methods[0]
            # start_time = data_point.start_time
            # end_time = data_point.end_time
            feature_idx = data_point.feature_idx

            feature_file_addr = os.path.join(FEATURE_DIR_ADDR, sub_dir, data_name.split('.')[0] + '.pickle')
            features = pickle.load(open(feature_file_addr, 'rb'))

            feature = features[feature_idx]
            feature = np.reshape(feature, input_shape)

            if not len(X) and not len(Y):
                X = np.expand_dims(feature, axis=0)
                Y = label_content
            else:
                X = np.append(X, np.expand_dims(feature, axis=0), 0)
                Y = np.append(Y, label_content, 0)


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
        while (1):
            data_idx = random.randrange(num_data_files)
            data_point = working_list[data_idx]
            data_name = data_point.data_name
            sub_dir = data_point.sub_dir
            label_content = data_point.label_content
            # feature_name = self.preprocessing_methods[0]
            # start_time = data_point.start_time
            # end_time = data_point.end_time
            feature_idx = data_point.feature_idx

            feature_file_addr = os.path.join(FEATURE_DIR_ADDR, sub_dir, data_name.split('.')[0] + '.pickle')
            features = pickle.load(open(feature_file_addr, 'rb'))

            feature = features[feature_idx]
            feature = np.reshape(feature, input_shape)

            if not len(X) and not len(Y):
                X = np.expand_dims(feature, axis=0)
                Y = label_content
            else:
                X = np.append(X, np.expand_dims(feature, axis=0), 0)
                Y = np.append(Y, label_content, 0)


            if X.shape[0] >= batch_size:
                return (X, Y)
                X = []
                Y = []
