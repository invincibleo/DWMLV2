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

        self.data_list = self.create_data_list()

        if self.normalization:
            self.dataset_normalization()

        self.count_sets_data()

    def create_data_list(self):
        datalist_pickle_file = self.get_dataset_file_addr()
        if not tf.gfile.Exists(datalist_pickle_file) or not tf.gfile.Exists(self.feature_dir):
            individual_meta_file_base_addr = os.path.join(self.dataset_dir, 'meta/street/')
            audio_file_list = [x[:-4] for x in tf.gfile.ListDirectory(individual_meta_file_base_addr)]

            data_list = {'validation': [], 'testing': [], 'training': [],
                         'v_l': [], 'te_l': [], 'tr_l': []}
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
                    labels = {}
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

                        init_pointer = int(math.floor(start_time * fs))
                        feature = audio_raw_all[init_pointer:int(init_pointer + fs * self.FLAGS.time_resolution)]
                        label = np.reshape(event_roll[point_idx], (-1))

                        if np.shape(feature) != (1764, ):
                            print('CNM')
                        hash_name_hashed = hashlib.sha1(tf.compat.as_bytes(audio_file + str(point_idx))).hexdigest()
                        percentage_hash = int(hash_name_hashed, 16) % (100 + 1)

                        if percentage_hash < self.FLAGS.validation_percentage:
                            data_list['validation'].append(feature)
                            data_list['v_l'].append(label)
                        elif percentage_hash < (self.FLAGS.testing_percentage + self.FLAGS.validation_percentage):
                            data_list['testing'].append(feature)
                            data_list['te_l'].append(label)
                        else:
                            data_list['training'].append(feature)
                            data_list['tr_l'].append(label)

            data_list['validation'] = np.vstack(data_list['validation'])
            data_list['testing'] = np.vstack(data_list['testing'])
            data_list['training'] = np.vstack(data_list['training'])
            data_list['validation'] = np.expand_dims(data_list['validation'], axis=3)
            data_list['testing'] = np.expand_dims(data_list['testing'], axis=3)
            data_list['training'] = np.expand_dims(data_list['training'], axis=3)
            data_list['v_l'] = np.vstack(data_list['v_l'])
            data_list['te_l'] = np.vstack(data_list['te_l'])
            data_list['tr_l'] = np.vstack(data_list['tr_l'])
            # data_list['training'], _, _, _ = self.balance_data_list(data_list['training'])
            # pickle.dump(data_list, open(datalist_pickle_file, 'wb'), 2)
        else:
            data_list = pickle.load(open(datalist_pickle_file, 'rb'))

        return data_list
