import unittest
import argparse
from hyperopt import fmin, tpe, hp


from application.Dataset_DCASE2017_Task3 import *
from application.LearnerInceptionV3 import LearnerInceptionV3
from core.evaluation import DCASE2016_EventDetection_SegmentBasedMetrics


DATASET_DIR = "/media/invincibleo/Windows/Users/u0093839/Box Sync/PhD/Experiment/SoundEventRecognition/DCASE2017-baseline-system-master/applications/data/TUT-sound-events-2017-development"
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
            '--output_graph',
            type=str,
            default='tmp/output_graph.pb',
            help='Where to save the trained graph.'
        )
        parser.add_argument(
            '--output_labels',
            type=str,
            default='tmp/output_labels.txt',
            help='Where to save the trained graph\'s labels.'
        )
        parser.add_argument(
            '--summaries_dir',
            type=str,
            default='tmp/retrain_logs',
            help='Where to save summary logs for TensorBoard.'
        )
        parser.add_argument(
            '--how_many_training_steps',
            type=int,
            default=4000,
            help='How many training steps to run before ending.'
        )
        parser.add_argument(
            '--learning_rate',
            type=float,
            default=0.01,
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
            '--eval_step_interval',
            type=int,
            default=10,
            help='How often to evaluate the training results.'
        )
        parser.add_argument(
            '--train_batch_size',
            type=int,
            default=100,
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
            '--print_misclassified_test_images',
            default=False,
            help="""\
            Whether to print out a list of all misclassified test images.\
            """,
            action='store_true'
        )
        parser.add_argument(
            '--model_dir',
            type=str,
            default='tmp/imagenet',
            help="""\
            Path to classify_image_graph_def.pb,
            imagenet_synset_to_human_label_map.txt, and
            imagenet_2012_challenge_label_map_proto.pbtxt.\
            """
        )
        parser.add_argument(
            '--bottleneck_dir',
            type=str,
            default='tmp/bottleneck',
            help='Path to cache bottleneck layer values as files.'
        )
        parser.add_argument(
            '--final_tensor_name',
            type=str,
            default='final_result',
            help="""\
            The name of the output classification layer in the retrained graph.\
            """
        )
        parser.add_argument(
            '--flip_left_right',
            default=False,
            help="""\
            Whether to randomly flip half of the training images horizontally.\
            """,
            action='store_true'
        )
        parser.add_argument(
            '--random_crop',
            type=int,
            default=0,
            help="""\
            A percentage determining how much of a margin to randomly crop off the
            training images.\
            """
        )
        parser.add_argument(
            '--random_scale',
            type=int,
            default=0,
            help="""\
            A percentage determining how much to randomly scale up the size of the
            training images by.\
            """
        )
        parser.add_argument(
            '--random_brightness',
            type=int,
            default=0,
            help="""\
            A percentage determining how much to randomly multiply the training image
            input pixels up or down by.\
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
        FLAGS, unparsed = parser.parse_known_args()

        # define an objective function
        def objective(args):
            FLAGS.learning_rate = args['lr']
            FLAGS.num_second_last_layer = args['num_second_last_layer']
            FLAGS.drop_out_rate = args['drop_out_rate']
            FLAGS.train_batch_size = args['batch_size']

            dataset = Dataset_DCASE2017_Task3(dataset_dir=DATASET_DIR, flag=FLAGS, encoding='khot', preprocessing_methods=['mel', 'normalization'])
            learner = LearnerInceptionV3(dataset=dataset, learner_name='InceptionV3', flag=FLAGS)
            evaluator = DCASE2016_EventDetection_SegmentBasedMetrics(class_list=dataset.label_list, time_resolution=FLAGS.time_resolution)

            # dataset.get_batch_data('training', 10, (-1, 40, 1))

            learner.learn()
            truth, prediction = learner.predict()
            evaluator.evaluate(truth, prediction)
            results = evaluator.results()
            print('F:' + str(results['class_wise_average']['F']) + '\n')
            print('ER' + str(results['class_wise_average']['ER']) + '\n')

            return {'F score': results['class_wise_average']['F'], 'Error Rate': results['class_wise_average']['ER']}

        # define a search space
        space = {'lr': hp.choice('lr', [0.001, 0.01, 0.1, 0.5, 1, 10]),
                 'num_second_last_layer': hp.choice('num_second_last_layer', [64, 128, 256, 512, 1024]),
                 'drop_out_rate': hp.choice('drop_out_rate', [0.1, 0.3, 0.5, 0.7, 0.9]),
                 'batch_size': hp.choice('batch_size', [50, 100, 200, 500, 800, 1000])}

        # minimize the objective over the space
        best = fmin(objective, space, algo=tpe.suggest, max_evals=100)

        print best


if __name__ == '__main__':
    unittest.main()
