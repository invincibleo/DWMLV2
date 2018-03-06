import unittest
import argparse
from hyperopt import fmin, tpe, hp

import os, sys
sys.path.append(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0])

from application.Dataset_AVEC2016_V2 import *
from application.LearnerLSTMReg_V3 import *
from application.Evaluator_AVEC2016 import *
from application.LearnerInceptionV3 import LearnerInceptionV3
from core.evaluation import DCASE2016_EventDetection_SegmentBasedMetrics
import datetime
import tensorflow as tf

#DATASET_DIR = "/users/sista/dtang/Datasets/AVEC2016"
DATASET_DIR = "/users/stadius/dspuser/dtang/Datasets/AVEC2016"

class MyTestCase(unittest.TestCase):
    def test_something(self):
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
            default=256,
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
            default=256,
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
            default=0.04,
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
            '--drop_out_rate',
            type=float,
            default=0.9,
            help="""\
            \
            """
        )
        parser.add_argument(
            '--coding',
            type=str,
            default='number',
            help="""\
            one hot encoding: onehot, k hot encoding: khot, continues value: number
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
        parser.add_argument(
            '--dimension',
            type=str,
            default="17,40,1",
            help="""\
            input dimension to the model
            \
            """
        )
        FLAGS, unparsed = parser.parse_known_args()

        setup_keras()
        dataset = Dataset_AVEC2016(dataset_dir=DATASET_DIR, flag=FLAGS, normalization=False, dimension=FLAGS.dimension, using_existing_features=True)

        learner = LearnerLSTMReg(dataset=dataset, learner_name='LSTMReg', flag=FLAGS)
        evaluator = Evaluator_AVEC2016()

        truth, prediction = learner.learn()
        # truth, prediction = learner.predict()
        evaluator.evaluate(truth, prediction)
        results = evaluator.results()
        print(results)

        print(results)
        results_dir_addr = 'tmp/results/'
        current_time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        if not tf.gfile.Exists(results_dir_addr):
            tf.gfile.MakeDirs(results_dir_addr)
        hash_FLAGS = hashlib.sha1(str(FLAGS)).hexdigest()
        results_file_dir = os.path.join(results_dir_addr, dataset.dataset_name, hash_FLAGS)
        if not tf.gfile.Exists(results_file_dir):
            tf.gfile.MakeDirs(results_file_dir)
            json.dump(results, open(results_file_dir + '/results_' + current_time_str + '.json', 'wb'), indent=4)
            json.dump(zip(truth[:, 0].tolist(), prediction[:, 0].tolist()), open(results_file_dir + '/results_' + current_time_str + '_0.json', 'a'), indent=4)
            json.dump(zip(truth[:, 1].tolist(), prediction[:, 1].tolist()), open(results_file_dir + '/results_' + current_time_str + '_1.json', 'a'), indent=4)
            with open(results_file_dir + 'FLAGS_' + current_time_str + '.txt', 'wb') as f:
                f.write(str(FLAGS))

if __name__ == '__main__':
    unittest.main()
