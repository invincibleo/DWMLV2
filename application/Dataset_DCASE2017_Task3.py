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
import json
import random

from tqdm import tqdm
from core.Dataset import Dataset
from core.GeneralFileAccessor import GeneralFileAccessor
from core.TimeseriesPoint import *
from core.Preprocessing import *
from core.util import *


class Dataset_DCASE2017_Task3(Dataset):

    def __init__(self, *args, **kwargs):
        dataset_name = 'Dataset_DCASE2017_Task3'
        label_list = sorted(['people walking', 'car', 'large vehicle', 'people speaking', 'brakes squeaking', 'children'])
        num_classes = len(label_list)
        super(Dataset_DCASE2017_Task3, self).__init__(self, dataset_name=dataset_name,
                                                      label_list=label_list,
                                                      num_classes=num_classes,
                                                      *args, **kwargs)
        if False:
            self.data_list = self.create_data_list()
        else:
            self.data_list = self.create_data_list_by_file()

        self.data_list['training'], _, _, _ = self.balance_data_list(self.data_list['training'])

        if self.normalization:
            self.dataset_normalization()

        self.count_sets_data()

    def create_data_list(self):
        datalist_pickle_file = self.get_dataset_file_addr()
        if not tf.gfile.Exists(datalist_pickle_file) or not tf.gfile.Exists(self.feature_dir):
            individual_meta_file_base_addr = os.path.join(self.dataset_dir, 'meta/street/')
            audio_file_list = [x[:-4] for x in tf.gfile.ListDirectory(individual_meta_file_base_addr)]

            data_list = {'validation': [], 'testing': [], 'training': []}
            for audio_file in tqdm(audio_file_list, desc='Creating features:'):
                audio_meta_file_addr = os.path.join(individual_meta_file_base_addr, audio_file + '.ann')
                audio_file_addr = os.path.join(os.path.join(self.dataset_dir, 'audio/street/'), audio_file + '.wav')
                audio_raw_all, fs = GeneralFileAccessor(file_path=audio_file_addr,
                                                        mono=True).read()
                event_roll = Preprocessing.audio_event_roll(meta_file=audio_meta_file_addr,
                                                            time_resolution=self.FLAGS.time_resolution,
                                                            label_list=self.label_list)

                feature_file_addr = self.get_feature_file_addr('street', audio_file)
                if not tf.gfile.Exists(feature_file_addr):
                    feature_base_addr = '/'.join(feature_file_addr.split('/')[:-1])
                    if not tf.gfile.Exists(feature_base_addr):
                        os.makedirs(feature_base_addr)
                    save_features = True
                    features = {}
                else:
                    save_features = False
                    # features = pickle.load(open(feature_file_addr, 'rb'))

                if self.FLAGS.coding == 'khot':
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
                            audio_raw = audio_raw_all[int(math.floor(start_time * fs)):int(math.floor(end_time * fs))]
                            feature = Preprocessing(parameters=self.feature_parameters).feature_extraction(dataset=self,
                                                                                                           audio_raw=audio_raw)
                            features[point_idx] = np.reshape(feature, (1, -1))
                            # features[point_idx]=feature

                elif self.FLAGS.coding == 'onehot':
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
                                            feature_idx=point_idx * idx + idx,
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
                            audio_raw = audio_raw_all[int(math.floor(start_time * fs)):int(math.floor(end_time * fs))]
                            feature = Preprocessing(parameters=self.feature_parameters).feature_extraction(dataset=self,
                                                                                                           audio_raw=audio_raw)
                            features[point_idx] = np.reshape(feature, (1, -1))
                            # features[point_idx] = feature

                if save_features:
                    pickle.dump(features, open(feature_file_addr, 'wb'), 2)
            pickle.dump(data_list, open(datalist_pickle_file, 'wb'), 2)
        else:
            data_list = pickle.load(open(datalist_pickle_file, 'rb'))

        return data_list

    def create_data_list_by_file(self):
        datalist_pickle_file = self.get_dataset_file_addr()
        if not tf.gfile.Exists(datalist_pickle_file) or not tf.gfile.Exists(self.feature_dir):
            datalist_pickle_file = self.get_dataset_file_addr()
            if not tf.gfile.Exists(datalist_pickle_file) or not tf.gfile.Exists(self.feature_dir):
                individual_meta_file_base_addr = os.path.join(self.dataset_dir, 'evaluation_setup')
                meta_file_list = tf.gfile.ListDirectory(individual_meta_file_base_addr)

                data_list = {'validation': [], 'testing': [], 'training': []}
                for meta_file in tqdm(meta_file_list, desc='Creating features:'):
                    category = meta_file.split('.')[0].split('_')[-1]
                    if not category == 'test':
                        meta_file_addr = os.path.join(individual_meta_file_base_addr, meta_file)
                        meta_file_content = np.array(GeneralFileAccessor(file_path=meta_file_addr).read())
                        audio_file_list = set(meta_file_content[:, 0])
                        for audio_file in audio_file_list:
                            audio_file_name = audio_file.split('.')[0].split('/')[-1]
                            audio_meta_file_content = meta_file_content[meta_file_content[:, 0] == audio_file]
                            audio_file_addr = os.path.join(self.dataset_dir, audio_file)
                            audio_raw_all, fs = GeneralFileAccessor(file_path=audio_file_addr,
                                                                    mono=True).read()
                            event_roll = Preprocessing.audio_event_roll(meta_content=audio_meta_file_content,
                                                                        time_resolution=self.FLAGS.time_resolution,
                                                                        label_list=self.label_list)

                            feature_file_addr = self.get_feature_file_addr('street', audio_file_name)
                            if not tf.gfile.Exists(feature_file_addr):
                                feature_base_addr = '/'.join(feature_file_addr.split('/')[:-1])
                                if not tf.gfile.Exists(feature_base_addr):
                                    os.makedirs(feature_base_addr)
                                save_features = True
                                features = {}
                            else:
                                save_features = False

                            for point_idx in range(event_roll.shape[0]):
                                label_name = ','.join(np.array(self.label_list)[np.array(event_roll[point_idx], dtype=bool)])
                                start_time = point_idx * self.FLAGS.time_resolution
                                end_time = (point_idx + 1) * self.FLAGS.time_resolution
                                new_point = AudioPoint(
                                    data_name=audio_file_name + '.wav',
                                    sub_dir='street',
                                    label_name=label_name,
                                    label_content=np.reshape(event_roll[point_idx], (1, -1)),
                                    extension='wav',
                                    fs=fs,
                                    feature_idx=point_idx,
                                    start_time=start_time,
                                    end_time=end_time
                                )

                                if category == 'evaluate':
                                    data_list['validation'].append(new_point)
                                elif category == 'test':
                                    data_list['testing'].append(new_point)
                                else:
                                    data_list['training'].append(new_point)

                                if save_features:
                                    # feature extraction
                                    audio_raw = audio_raw_all[int(math.floor(start_time * fs)):int(math.floor(end_time * fs))]
                                    feature = Preprocessing(parameters=self.feature_parameters).feature_extraction(dataset=self,
                                                                                                                   audio_raw=audio_raw)
                                    features[point_idx] = np.reshape(feature, (1, -1))

                            if save_features:
                                pickle.dump(features, open(feature_file_addr, 'wb'), 2)
                    else: #if category is test
                        meta_file_addr = os.path.join(individual_meta_file_base_addr, meta_file)
                        meta_file_content = np.array(GeneralFileAccessor(file_path=meta_file_addr).read())
                        audio_file_list = set(meta_file_content[:, 0])
                        for audio_file in audio_file_list:
                            audio_file_name = audio_file.split('.')[0].split('/')[-1]
                            audio_meta_file_addr = os.path.join(self.dataset_dir, 'meta/street', audio_file_name + '.ann')
                            audio_file_addr = os.path.join(self.dataset_dir, audio_file)
                            audio_raw_all, fs = GeneralFileAccessor(file_path=audio_file_addr,
                                                                    mono=True).read()
                            event_roll = Preprocessing.audio_event_roll(meta_file=audio_meta_file_addr,
                                                                        time_resolution=self.FLAGS.time_resolution,
                                                                        label_list=self.label_list)

                            for point_idx in range(event_roll.shape[0]):
                                label_name = ','.join(np.array(self.label_list)[np.array(event_roll[point_idx], dtype=bool)])
                                start_time = point_idx * self.FLAGS.time_resolution
                                end_time = (point_idx + 1) * self.FLAGS.time_resolution
                                new_point = AudioPoint(
                                    data_name=audio_file_name + '.wav',
                                    sub_dir='street',
                                    label_name=label_name,
                                    label_content=np.reshape(event_roll[point_idx], (1, -1)),
                                    extension='wav',
                                    fs=fs,
                                    feature_idx=point_idx,
                                    start_time=start_time,
                                    end_time=end_time
                                )
                                data_list['testing'].append(new_point)

            pickle.dump(data_list, open(datalist_pickle_file, 'wb'), 2)
        else:
            data_list = pickle.load(open(datalist_pickle_file, 'rb'))

        return data_list

