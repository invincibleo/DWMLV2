import unittest
import argparse
import datetime
from hyperopt import fmin, tpe, hp, STATUS_OK


from application.Dataset_DCASE2017_Task3 import *
from application.LearnerInceptionV3 import LearnerInceptionV3
from core.evaluation import DCASE2016_EventDetection_SegmentBasedMetrics


DATASET_DIR = "/media/invincibleo/Windows/Users/u0093839/Box Sync/PhD/Experiment/DWML_V2/DCASE2017-baseline-system-master/applications/data/TUT-sound-events-2017-development"
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
            default=1,
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
            default=0.5,
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

        # define an objective function
        def objective(args):
            FLAGS.learning_rate = args['lr']
            FLAGS.num_second_last_layer = args['num_second_last_layer']
            FLAGS.drop_out_rate = args['drop_out_rate']
            FLAGS.train_batch_size = args['batch_size']

            dataset = Dataset_DCASE2017_Task3(dataset_dir=DATASET_DIR, flag=FLAGS, encoding='khot',
                                              preprocessing_methods=['mel'], normalization=True, dimension=40)
            learner = LearnerInceptionV3(dataset=dataset, learner_name='InceptionV3', flag=FLAGS)
            evaluator = DCASE2016_EventDetection_SegmentBasedMetrics(class_list=dataset.label_list,
                                                                     time_resolution=FLAGS.time_resolution)

            # dataset.get_batch_data('training', 10, (-1, 40, 1))

            learner.learn()
            truth, prediction = learner.predict()
            evaluator.evaluate(truth, prediction, threshold=0.8)
            results = evaluator.results()

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

            # return {'F score': results['class_wise_average']['F'], 'Error Rate': results['class_wise_average']['ER']}
            return {'loss': results['class_wise_average']['ER'], 'status': STATUS_OK}
        # define a search space
        space = {'lr': hp.choice('lr', [0.0001, 0.001, 0.005, 0.01, 0.1]),
                 'num_second_last_layer': hp.choice('num_second_last_layer', [16, 64, 128, 256, 512, 1024]),
                 'drop_out_rate': hp.choice('drop_out_rate', [0.1, 0.3, 0.5, 0.7, 0.9]),
                 'batch_size': hp.choice('batch_size', [64, 128, 256])}

        # minimize the objective over the space
        best = fmin(objective, space, algo=tpe.suggest, max_evals=100)

        print best


if __name__ == '__main__':
    unittest.main()
