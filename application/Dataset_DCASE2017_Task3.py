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

from core.Dataset import Dataset
from core.GeneralFileAccessor import GeneralFileAccessor
from core.TimeserisePoint import *

PICKLE_FILE_ADDR = './tmp/dataset_DCASE2017task3.pickle'



class Dataset_DCASE2017_Task3(Dataset):

    def __init__(self, *args, **kwargs):
        super(Dataset_DCASE2017_Task3, self).__init__(self, *args, **kwargs)
        self.dataset_name = 'Dataset_DCASE2017_Task3'
        self.label_list = sorted(['people walking', 'car', 'large vehicle', 'people speaking', 'brakes squeaking'])
        self.data_list = self.create_data_list()

    @staticmethod
    def audio_event_roll(meta_file, label_list, time_resolution):
        meta_file_content = GeneralFileAccessor(meta_file).read()
        end_times = np.array([float(x[1]) for x in meta_file_content])
        max_offset_value = np.max(end_times, 0)

        event_roll = np.zeros((int(math.ceil(max_offset_value / time_resolution)), len(label_list)))

        for line in meta_file_content:
            label_name = line[-1]
            label_idx = label_list.index(label_name)
            event_start = float(line[0])
            event_end = float(line[1])

            onset = int(math.floor(event_start / time_resolution))
            offset = int(math.floor(event_end / time_resolution))

            event_roll[onset:offset, label_idx] = 1

        return event_roll

    def create_data_list(self):
        individual_meta_file_base_addr = os.path.join(self.dataset_dir, 'meta/street/')
        audio_file_list = [x[:-4] for x in tf.gfile.ListDirectory(individual_meta_file_base_addr)]

        for audio_file in audio_file_list:
            audio_meta_file_addr = os.path.join(individual_meta_file_base_addr, audio_file + '.ann')
            event_roll = Dataset_DCASE2017_Task3.audio_event_roll(audio_meta_file_addr,
                                                                  time_resolution=self.FLAGS.time_resolution, label_list=self.label_list)
            for point_idx in range(event_roll.shape[0]):
                label_name = zip(event_roll[point_idx][self.label_list])
                AudioPoint(
                    data_name=audio_file+'.wav',
                    sub_dir='street',
                    label_name=label_name
                )


        meta_file_addr = os.path.join(self.dataset_dir, '/meta/')
        meta_content = GeneralFileAccessor(meta_file_addr).read()

        self.event_roll = {}
        data_list = {}
        line_idx = 0
        audio_frame_size = 1
        for line in meta_content:
            line_list = line.split('\t')
            file_name = line_list[0][6:]
            sub_dir = 'audio'
            start_time = float(line_list[2])
            end_time = float(line_list[3])
            label_name = line_list[4]
            duration = end_time - start_time

            audio_meta_file_addr = os.path.join(os.path.join(self.dataset_dir, 'meta'), file_name.split('.')[0])######
            if duration >= audio_frame_size:
                i = 0
                while (int(duration / audio_frame_size)):
                    time_idx = [start_time + i*audio_frame_size, start_time + (i+1)*audio_frame_size]
                    duration = duration - audio_frame_size

                    hash_name_hashed = hashlib.sha1(tf.compat.as_bytes(file_name+str(line_idx)+str(i))).hexdigest()
                    percentage_hash = ((int(hash_name_hashed, 16) % (self.max_num_data_per_class + 1)) *
                                       (100.0 / self.max_num_data_per_class))

                    if not label_name in data_list.keys():
                        data_list[label_name] = {'subdir':sub_dir, 'validation':[], 'testing':[], 'training':[]}

                    if percentage_hash < self.validation_percentage:
                        data_list[label_name]['validation'].append((file_name, time_idx))
                    elif percentage_hash < (self.testing_percentage + self.validation_percentage):
                        data_list[label_name]['testing'].append((file_name, time_idx))
                    else:
                        data_list[label_name]['training'].append((file_name, time_idx))

                    i = i + 1


