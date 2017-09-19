#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 25.08.17 11:09
# @Author  : Duowei Tang
# @Site    : https://iiw.kuleuven.be/onderzoek/emedia/people/phd-students/duoweitang
# @File    : Dataset_Youtube8M
# @Software: PyCharm Community Edition

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from tqdm import tqdm
import pickle
import numpy as np
import csv
import pandas
import hashlib

from core.Dataset import Dataset
from core.GeneralFileAccessor import GeneralFileAccessor
from core.TimeseriesPoint import *
from core.Preprocessing import *
from core.util import *
from core.ontologyProcessing import *


class Dataset_Youtube8M(Dataset):
    
    def __init__(self, *args, **kwargs):
        dataset_name = 'Dataset_Youtube8M'
        super(Dataset_Youtube8M, self).__init__(self, dataset_name=dataset_name, *args, **kwargs)
        self.aso = {}
        self.if_second_level_labels = kwargs.get('if_second_level_labels', False)
        self.data_list_meta = []
        self.label_list = self.get_label_list()
        self.num_classes = len(self.label_list)
        self.extensions = ['wav']
        self.data_list = self.create_data_list()

        if self.normalization:
            self.dataset_normalization()

        self.count_sets_data()

    def get_label_list(self):
        ontology_file_addr = os.path.join(self.FLAGS.parameter_dir, 'ontology.json')

        csv_file_addr = os.path.join(self.FLAGS.parameter_dir, 'balanced_train_segments.csv')
        data_list_meta = pandas.read_csv(filepath_or_buffer=csv_file_addr,
                                         sep=' ',
                                         header=None,
                                         skiprows=3
                                         )
        self.data_list_meta = dict(zip(data_list_meta[0].str.replace(',', ''), data_list_meta[3].str.split(',')))

        if self.if_second_level_labels:
            second_level_class, self.aso = OntologyProcessing.get_2nd_level_label_name_list(ontology_file_addr)
            return sorted(second_level_class)
        else:
            self.aso = OntologyProcessing.get_label_name_list(ontology_file_addr)
            label_list_set = set()
            for values in self.data_list_meta.values():
                for label in values:
                    label_list_set.add(label)
            return sorted(list(label_list_set))

    def create_data_list(self):
        datalist_pickle_file = self.get_dataset_file_addr()
        if not tf.gfile.Exists(datalist_pickle_file) or not tf.gfile.Exists(self.feature_dir):
            audio_file_list = []
            for extension in self.extensions:
                file_glob = os.path.join(self.dataset_dir, '*.' + extension)
                audio_file_list.extend(tf.gfile.Glob(file_glob))

            data_list = {'validation': [], 'testing': [], 'training': []}
            for audio_file_addr in tqdm(audio_file_list, desc='Creating features:'):
                audio_file = os.path.basename(audio_file_addr)
                audio_raw_all, fs = GeneralFileAccessor(file_path=audio_file_addr,
                                                        mono=True).read()
                num_points = int(np.floor(len(audio_raw_all) / fs))

                feature_file_addr = self.get_feature_file_addr('', audio_file)
                if not tf.gfile.Exists(feature_file_addr):
                    feature_base_addr = '/'.join(feature_file_addr.split('/')[:-1])
                    if not tf.gfile.Exists(feature_base_addr):
                        os.makedirs(feature_base_addr)
                    save_features = True
                    features = {}
                else:
                    save_features = False
                    # features = pickle.load(open(feature_file_addr, 'rb'))

                if self.if_second_level_labels:
                    label_name = self.data_list_meta[audio_file.split('.')[0]]
                    second_level_class_name = OntologyProcessing.get_2nd_level_class_label_index(label_name, self.aso,
                                                                                                 self.label_list)
                    label_name = ''
                    label_content = np.zeros((1, self.num_classes))
                    for class_name in second_level_class_name:
                        label_content[0, self.label_list.index(class_name)] = 1
                        label_name = label_name + self.aso[class_name]['name'] + ','
                else:
                    label_name_list = self.data_list_meta[audio_file.split('.')[0]]
                    label_name = ''

                if self.FLAGS.coding == 'khot':
                    label_content = np.zeros((1, self.num_classes))
                    for class_name in label_name_list:
                        label_content[0, self.label_list.index(class_name)] = 1
                        label_name = label_name + self.aso[class_name]['name'] + ','

                    for point_idx in range(num_points):
                        start_time = point_idx * self.FLAGS.time_resolution
                        end_time = (point_idx + 1) * self.FLAGS.time_resolution
                        new_point = AudioPoint(
                            data_name=audio_file + '.wav',
                            sub_dir='',
                            label_name=label_name,
                            label_content=label_content,
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
                            audio_raw = audio_raw_all[int(math.floor(start_time * fs)):int(math.floor(end_time * fs))]
                            feature = Preprocessing(parameters=self.feature_parameters).feature_extraction(dataset=self,
                                                                                                           audio_raw=audio_raw)
                            features[point_idx] = np.reshape(feature, (1, -1))
                            # features[point_idx]=feature

                elif self.FLAGS.coding == 'onehot':
                    for point_idx in range(num_points):
                        start_time = point_idx * self.FLAGS.time_resolution
                        end_time = (point_idx + 1) * self.FLAGS.time_resolution
                        for label in label_name_list:
                            label_name = label
                            label_content = np.zeros((1, self.num_classes))
                            label_content[self.label_list.index(label)] = 1
                            new_point = AudioPoint(
                                data_name=audio_file + '.wav',
                                sub_dir='',
                                label_name=label_name,
                                label_content=label_content,
                                extension='wav',
                                fs=fs,
                                feature_idx=point_idx,
                                start_time=start_time,
                                end_time=end_time
                            )
                            hash_name_hashed = hashlib.sha1(tf.compat.as_bytes(audio_file + str(point_idx) + str(label))).hexdigest()
                            percentage_hash = int(hash_name_hashed, 16) % (100 + 1)

                            if percentage_hash < self.FLAGS.validation_percentage:
                                data_list['validation'].append(new_point)
                            elif percentage_hash < (self.FLAGS.testing_percentage + self.FLAGS.validation_percentage):
                                data_list['testing'].append(new_point)
                            else:
                                data_list['training'].append(new_point)

                        if save_features:
                            # feature extraction
                            audio_raw = audio_raw_all[int(math.floor(start_time * fs)):int(math.floor(end_time * fs))]
                            feature = Preprocessing(parameters=self.feature_parameters).feature_extraction(dataset=self,
                                                                                                           audio_raw=audio_raw)
                            features[point_idx] = np.reshape(feature, (1, -1))
                            # features[point_idx]=feature

                if save_features:
                    pickle.dump(features, open(feature_file_addr, 'wb'), 2)
            pickle.dump(data_list, open(datalist_pickle_file, 'wb'), 2)
        else:
            data_list = pickle.load(open(datalist_pickle_file, 'rb'))

        # normalization, val and test set using training mean and training std
        if self.normalization:
            mean_std_file_addr = os.path.join(self.feature_dir, 'mean_std_time_res' + str(self.FLAGS.time_resolution) + '.json')
            if not tf.gfile.Exists(mean_std_file_addr):
                feature_buf = []
                batch_count = 0
                for training_point in tqdm(data_list['training'], desc='Computing training set mean and std'):
                    feature_idx = training_point.feature_idx
                    data_name = training_point.data_name
                    sub_dir = training_point.sub_dir
                    feature_file_addr = self.get_feature_file_addr(sub_dir, data_name)
                    features = pickle.load(open(feature_file_addr, 'rb'))

                    feature_buf.append(features[feature_idx])
                    batch_count += 1
                    if batch_count >= 512:
                        self.online_mean_variance(feature_buf)
                        feature_buf = []
                        batch_count = 0

                json.dump(obj=dict({'training_mean': self.training_mean.tolist(), 'training_std': self.training_std.tolist()}),
                          fp=open(mean_std_file_addr, 'wb'))
            else:
                training_statistics = json.load(open(mean_std_file_addr, 'r'))
                self.training_mean = np.reshape(training_statistics['training_mean'], (1, -1))
                self.training_std = np.reshape(training_statistics['training_std'], (1, -1))

        # count data point
        self.num_training_data = len(data_list['training'])
        self.num_validation_data = len(data_list['validation'])
        self.num_testing_data = len(data_list['testing'])
        return data_list


