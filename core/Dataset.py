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

class Dataset(object):

    def __init__(self, *args, **kwargs):
        self.data_list = kwargs.get('data_list', None)
        self.dataset_name = kwargs.get('dataset_name', "")
        self.dataset_dir = kwargs.get('dataset_dir', "")
        self.num_classes = int(kwargs.get('num_classes', 0))
        self.FLAGS = kwargs.get('flag', '')
        self.preprocessing_methods = kwargs.get('preprocessing_methods', [])
        self.normalization = kwargs.get('normalization', True)
        self.training_mean = 0.0
        self.training_std = 1.0
