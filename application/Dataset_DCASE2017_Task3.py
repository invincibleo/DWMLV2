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

from core.Dataset import Dataset
from core.GeneralFileAccessor import GeneralFileAccessor
from core.TimeseriesPoint import *

PICKLE_FILE_ADDR = './tmp/dataset_DCASE2017task3.pickle'



class Dataset_DCASE2017_Task3(Dataset):

    def __init__(self, *args, **kwargs):
        super(Dataset_DCASE2017_Task3, self).__init__(self, *args, **kwargs)
        self.dataset_name = 'Dataset_DCASE2017_Task3'
        self.label_list = sorted(['people walking', 'car', 'large vehicle', 'people speaking', 'brakes squeaking', 'children'])
        self.encoding = kwargs.get('encoding', 'khot')
        self.num_classes = len(self.label_list)
        if self.encoding == 'khot':
            self.data_list = self.create_khot_data_list()
        elif self.encoding == 'onehot':
            self.data_list = self.create_onehot_data_list()

    @staticmethod
    def audio_event_roll(meta_file, label_list, time_resolution):
        meta_file_content = GeneralFileAccessor(meta_file).read()
        end_times = np.array([float(x[1]) for x in meta_file_content])
        max_offset_value = np.max(end_times, 0)

        event_roll = np.zeros((int(math.ceil(max_offset_value / time_resolution)), len(label_list)))
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

    def create_khot_data_list(self):
        pickle_file = 'tmp/dataset/DCASE2017_khot.pickle'
        if not tf.gfile.Exists(pickle_file):
            individual_meta_file_base_addr = os.path.join(self.dataset_dir, 'meta/street/')
            audio_file_list = [x[:-4] for x in tf.gfile.ListDirectory(individual_meta_file_base_addr)]

            data_list = {'validation': [], 'testing': [], 'training': []}
            for audio_file in audio_file_list:
                audio_meta_file_addr = os.path.join(individual_meta_file_base_addr, audio_file + '.ann')
                event_roll = Dataset_DCASE2017_Task3.audio_event_roll(audio_meta_file_addr,
                                                                      time_resolution=self.FLAGS.time_resolution, label_list=self.label_list)
                for point_idx in range(event_roll.shape[0]):
                    label_name = ','.join(np.array(self.label_list)[np.array(event_roll[point_idx], dtype=bool)])
                    new_point = AudioPoint(
                                    data_name=audio_file+'.wav',
                                    sub_dir='street',
                                    label_name=label_name,
                                    label_content=event_roll[point_idx],
                                    extension='wav',
                                    fs=self.FLAGS.fs,
                                    start_time=point_idx*self.FLAGS.time_resolution,
                                    end_time=(point_idx+1)*self.FLAGS.time_resolution
                                )

                    hash_name_hashed = hashlib.sha1(tf.compat.as_bytes(audio_file + str(point_idx))).hexdigest()
                    percentage_hash = int(hash_name_hashed, 16) % (100 + 1)

                    if percentage_hash < self.FLAGS.validation_percentage:
                        data_list['validation'].append(new_point)
                    elif percentage_hash < (self.FLAGS.testing_percentage + self.FLAGS.validation_percentage):
                        data_list['testing'].append(new_point)
                    else:
                        data_list['training'].append(new_point)
            os.makedirs("tmp/dataset")
            pickle.dump(data_list, open(pickle_file, 'wb'), 2)
        else:
            data_list = pickle.load(open(pickle_file, 'rb'))
        return data_list

    def create_onehot_data_list(self):
        pickle_file = 'tmp/dataset/DCASE2017_onehot.pickle'
        if not tf.gfile.Exists(pickle_file):
            individual_meta_file_base_addr = os.path.join(self.dataset_dir, 'meta/street/')
            audio_file_list = [x[:-4] for x in tf.gfile.ListDirectory(individual_meta_file_base_addr)]

            data_list = {'validation': [], 'testing': [], 'training': []}
            for audio_file in audio_file_list:
                audio_meta_file_addr = os.path.join(individual_meta_file_base_addr, audio_file + '.ann')
                event_roll = Dataset_DCASE2017_Task3.audio_event_roll(audio_meta_file_addr,
                                                                      time_resolution=self.FLAGS.time_resolution, label_list=self.label_list)
                for point_idx in range(event_roll.shape[0]):
                    for idx in range(self.num_classes):
                        label_content = np.zeros((1, self.num_classes))
                        if event_roll[point_idx][idx] == 0:
                            label_name = ''
                        else:
                            label_name = self.label_list[idx]
                            label_content[0, idx] = 1
                        new_point = AudioPoint(
                                        data_name=audio_file+'.wav',
                                        sub_dir='street',
                                        label_name=label_name,
                                        label_content=label_content,
                                        extension='wav',
                                        fs=self.FLAGS.fs,
                                        start_time=point_idx*self.FLAGS.time_resolution,
                                        end_time=(point_idx+1)*self.FLAGS.time_resolution
                                    )

                        hash_name_hashed = hashlib.sha1(tf.compat.as_bytes(audio_file + str(point_idx) + str(idx))).hexdigest()
                        percentage_hash = int(hash_name_hashed, 16) % (100 + 1)

                        if percentage_hash < self.FLAGS.validation_percentage:
                            data_list['validation'].append(new_point)
                        elif percentage_hash < (self.FLAGS.testing_percentage + self.FLAGS.validation_percentage):
                            data_list['testing'].append(new_point)
                        else:
                            data_list['training'].append(new_point)
            os.makedirs("tmp/dataset")
            pickle.dump(data_list, open(pickle_file, 'wb'), 2)
        else:
            data_list = pickle.load(open(pickle_file, 'rb'))
        return data_list