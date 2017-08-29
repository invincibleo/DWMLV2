#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 08.08.17 17:06
# @Author  : Duowei Tang
# @Site    : https://iiw.kuleuven.be/onderzoek/emedia/people/phd-students/duoweitang
# @File    : Learner
# @Software: PyCharm Community Edition


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta
from abc import abstractmethod

class Learner(object):
    __metaclass__ = ABCMeta

    def __init__(self, *args, **kwargs):
        self.learner_name = kwargs.get('learner_name', '')
        self.dataset = kwargs.get('dataset', None)
        self.FLAGS = kwargs.get('flag', None)

    @abstractmethod
    def learn(self):
        pass

    @abstractmethod
    def predict(self):
        pass