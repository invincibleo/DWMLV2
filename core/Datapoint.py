#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 24.08.17 16:49
# @Author  : Duowei Tang
# @Site    : https://iiw.kuleuven.be/onderzoek/emedia/people/phd-students/duoweitang
# @File    : Datapoint
# @Software: PyCharm Community Edition


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

class Datapoint(object):
    def __init__(self, *args, **kwargs):
         self.data_name = kwargs.get('data_name', "")
         self.sub_dir = kwargs.get('sub_dir', "")
         self.label_name = kwargs.get('label_name', "")
         self.data_content = kwargs.get('data_content', None)
         self.label_content = kwargs.get('label_content', None)

    @property
    def data_name(self):
        return self.data_name

    @data_name.setter
    def data_name(self, value):
        self.data_name = value

    @property
    def sub_dir(self):
        return self.sub_dir

    @sub_dir.setter
    def sub_dir(self, value):
        self.sub_dir = value

    @property
    def label_name(self):
        return self.label_name

    @label_name.setter
    def label_name(self, value):
        self.label_name = value

    @property
    def data_content(self):
        return self.data_content

    @data_content.setter
    def data_content(self, value):
        self.data_content = value

    @property
    def label_content(self):
        return self.label_content

    @label_content.setter
    def label_content(self, value):
        self.label_content = value