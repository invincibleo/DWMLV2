#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 25.08.17 10:19
# @Author  : Duowei Tang
# @Site    : https://iiw.kuleuven.be/onderzoek/emedia/people/phd-students/duoweitang
# @File    : TimeserisePoint
# @Software: PyCharm Community Edition


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from core.Datapoint import Datapoint


class TimeserisePoint(Datapoint):
    
    def __init__(self, *args, **kwargs):
        super(TimeserisePoint, self).__init__(self, *args, **kwargs)
        try:
            self.start_time = int(kwargs.get('start_time'))
            self.end_time = int(kwargs.get('end_time'))
        except TypeError:
            print('Please input start and end time!')

        self.duration = self.end_time - self.start_time

    @property
    def start_tim(self):
        return self.start_time

    @start_tim.setter
    def start_tim(self, value):
        self.start_time = value

    @property
    def end_time(self):
        return self.end_time

    @end_time.setter
    def end_time(self, value):
        self.end_time = value

    @property
    def duration(self):
        return self.duration


class AudioPoint(TimeserisePoint):
    
    def __init__(self, *args, **kwargs):
        super(AudioPoint, self).__init__(self, *args, **kwargs)
        self.fs = int(kwargs.get('fs', 0))
        self.extension = kwargs.get('extension', "")

    @property
    def fs(self):
        return self.fs

    @fs.setter
    def fs(self, value):
        self.fs = value

    @property
    def extension(self):
        return self.extension

    @extension.setter
    def extension(self, value):
        self.extension = value


class PicturePoint(Datapoint):
    def __init__(self, *args, **kwargs):
        super(PicturePoint, self).__init__(self, *args, **kwargs)
        self.extension = kwargs.get('extension', "")
        self.pic_shape = kwargs.get('pic_shape', [])

    @property
    def extension(self):
        return self.extension

    @extension.setter
    def extension(self, value):
        self.extension = value

    @property
    def pic_shape(self):
        return self.pic_shape

    @pic_shape.setter
    def pic_shape(self, value):
        self.pic_shape = value