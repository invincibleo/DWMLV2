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
from core.Dataset import Dataset

PICKLE_FILE_ADDR = './tmp/dataset_youtube8m.pickle'


class Dataset_Youtube8M(Dataset):
    
    def __init__(self, *args, **kwargs):
        self.data_list = self.create_data_list()
        self.dataset_name = 'Dataset_Youtube8M'
        super(Dataset_Youtube8M, self).__init__(self,
                                                data_list=self.data_list,
                                                dataset_name=self.dataset_name,
                                                *args, **kwargs)

    def create_data_list(self):
        if not tf.gfile.Exists(PICKLE_FILE_ADDR):
            if not tf.gfile.Exists(self.dataset_dir):
                print("Dataset directory '" + self.dataset_dir + "' not found.")
                return None
            result = {}
            # The root directory comes first, so skip it.
            file_list = []
            for extension in self.extensions:
                file_glob = os.path.join(self.dataset_dir, '*.' + extension)
                file_list.extend(tf.gfile.Glob(file_glob))

            if not file_list:
                print('No files found')
                return None

            aso = OntologyProcessing.get_label_name_list(os.path.join(os.path.dirname(__file__), 'ontology.json'))
            second_level_class = OntologyProcessing.get_2nd_level_label_name_list(os.path.join(os.path.dirname(__file__), 'ontology.json'))
            with open(os.path.join(os.path.split(os.path.realpath(__file__))[0], os.path.join(os.path.dirname(__file__), 'balanced_train_segments.csv')), 'rb') as csvfile:
                label_list = csv.reader(csvfile, delimiter=',')

                result = {}

                file_list = [os.path.basename(x) for x in file_list]  # [:-4]
                extension_name = file_list[0].split('.')[-1]
                for label in tqdm(islice(label_list, 3, None), total=22163):
                    file_name = label[0] + '.' + extension_name
                    file_label = [re.sub(r'[ "]', '', x) for x in label[3:]]

                    if file_name in file_list:
                        hash_name = file_name
                        # This looks a bit magical, but we need to decide whether this file should
                        # go into the training, testing, or validation sets, and we want to keep
                        # existing files in the same set even if more files are subsequently
                        # added.
                        # To do that, we need a stable way of deciding based on just the file name
                        # itself, so we do a hash of that and then use that to generate a
                        # probability value that we use to assign it.
                        hash_name_hashed = hashlib.sha1(tf.compat.as_bytes(hash_name)).hexdigest()
                        percentage_hash = ((int(hash_name_hashed, 16) % (self.max_num_data_per_class + 1)) *
                                           (100.0 / self.max_num_data_per_class))
                        for label in file_label:
                            second_level_class_name = OntologyProcessing.get_2nd_level_class_label_index([label], aso, second_level_class)
                            label_name = aso[second_level_class_name[0]]['name']  ###label
                            if label_name == 'Animal':
                                continue
                            if not label_name in result.keys():
                                result[label_name] = {'subdir': '', 'validation': [], 'testing': [], 'training': []}

                            if percentage_hash < self.validation_percentage:
                                result[label_name]['validation'].append(file_name)
                            elif percentage_hash < (self.testing_percentage + self.validation_percentage):
                                result[label_name]['testing'].append(file_name)
                            else:
                                result[label_name]['training'].append(file_name)
            pickle.dump(result, open(PICKLE_FILE_ADDR, 'wb'), 2)
        else:
            result = pickle.load(open(PICKLE_FILE_ADDR, 'rb'))
        return result