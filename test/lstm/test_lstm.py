#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 14/09/2017 3:37 PM
# @Author  : Duowei Tang
# @Site    : https://iiw.kuleuven.be/onderzoek/emedia/people/phd-students/duoweitang
# @File    : test_lstm
# @Software: PyCharm Community Edition


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import datetime

from application.Dataset_DCASE2017_Task3 import *
from application.Dataset_Youtube8M import *
from application.LearnerLSTM import LearnerLSTM
from core.evaluation import DCASE2016_EventDetection_SegmentBasedMetrics

DATASET_DIR = "/Users/invincibleo/Box Sync/PhD/Experiment/DWML_V2/DCASE2017-baseline-system-master/applications/data/TUT-sound-events-2017-development"

def _setup_keras():
    """Setup keras backend and parameters
    """
    thread_count = 8
    os.environ['GOTO_NUM_THREADS'] = str(thread_count)
    os.environ['OMP_NUM_THREADS'] = str(thread_count)
    os.environ['MKL_NUM_THREADS'] = str(thread_count)

    if thread_count > 1:
        os.environ['OMP_DYNAMIC'] = 'False'
        os.environ['MKL_DYNAMIC'] = 'False'
    else:
        os.environ['OMP_DYNAMIC'] = 'True'
        os.environ['MKL_DYNAMIC'] = 'True'

        # Select Keras backend
        os.environ["KERAS_BACKEND"] = 'tensorflow'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        type=str,
        default='',
        help='Path to folders of labeled audios.'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='How large a learning rate to use when training.'
    )
    parser.add_argument(
        '--testing_percentage',
        type=int,
        default=10,
        help='What percentage of images to use as a test set.'
    )
    parser.add_argument(
        '--validation_percentage',
        type=int,
        default=10,
        help='What percentage of images to use as a validation set.'
    )
    parser.add_argument(
        '--train_batch_size',
        type=int,
        default=256,
        help='How many images to train on at a time.'
    )
    parser.add_argument(
        '--test_batch_size',
        type=int,
        default=-1,
        help="""\
        How many images to test on. This test set is only used once, to evaluate
        the final accuracy of the model after training completes.
        A value of -1 causes the entire test set to be used, which leads to more
        stable results across runs.\
        """
    )
    parser.add_argument(
        '--validation_batch_size',
        type=int,
        default=100,
        help="""\
        How many images to use in an evaluation batch. This validation set is
        used much more often than the test set, and is an early indicator of how
        accurate the model is during training.
        A value of -1 causes the entire validation set to be used, which leads to
        more stable results across training iterations, but may be slower on large
        training sets.\
        """
    )
    parser.add_argument(
        '--time_resolution',
        type=float,
        default=0.5,
        help="""\
        The hop of the FFT in sec.\
        """
    )
    parser.add_argument(
        '--fs',
        type=int,
        default=44100,
        help="""\
        The sampling frequency if an time-series signal is given\
        """
    )
    parser.add_argument(
        '--num_second_last_layer',
        type=int,
        default=512,
        help="""\
        \
        """
    )
    parser.add_argument(
        '--drop_out_rate',
        type=float,
        default=0.8,
        help="""\
        \
        """
    )
    parser.add_argument(
        '--coding',
        type=str,
        default='khot',
        help="""\
        one hot encoding: onehot or k hot encoding: khot
        \
        """
    )
    parser.add_argument(
        '--parameter_dir',
        type=str,
        default="parameters",
        help="""\
        parameter folder
        \
        """
    )
    FLAGS, unparsed = parser.parse_known_args()

    _setup_keras()

    dataset = Dataset_DCASE2017_Task3(dataset_dir=DATASET_DIR, flag=FLAGS, preprocessing_methods=['mel'],
                                      normalization=True, dimension=40)
    dataset.data_list['training'], _, _, _ = dataset.get_data_list_total_num_classes(dataset.data_list['training'])
    learner = LearnerLSTM(dataset=dataset, learner_name='LSTM', flag=FLAGS)
    evaluator = DCASE2016_EventDetection_SegmentBasedMetrics(class_list=dataset.label_list,
                                                             time_resolution=FLAGS.time_resolution)

    learner.learn()
    truth, prediction = learner.predict()
    evaluator.evaluate(truth, prediction, threshold=0.8)
    results = evaluator.results()
    print('F:' + str(results['class_wise_average']['F']) + '\n')
    print('ER' + str(results['class_wise_average']['ER']) + '\n')

    results_dir_addr = 'tmp/results/'
    current_time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if not tf.gfile.Exists(results_dir_addr):
        tf.gfile.MakeDirs(results_dir_addr)
        hash_FLAGS = hashlib.sha1(str(FLAGS)).hexdigest()
        results_file_dir = os.path.join(results_dir_addr, dataset.dataset_name, hash_FLAGS)
        tf.gfile.MakeDirs(results_file_dir)
        json.dump(results, open(results_file_dir + '/results_' + current_time_str + '.json', 'wb'), indent=4)
        with open(results_file_dir + 'FLAGS_' + current_time_str + '.txt', 'wb') as f:
            f.write(str(FLAGS))

if __name__ == '__main__':
    main()