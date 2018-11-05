import tensorflow as tf
from utils import get_data_info, read_data, load_word_embeddings
from model import IAN
import os
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('embedding_dim', 300, 'dimension of word embedding')
tf.app.flags.DEFINE_integer('batch_size', 128, 'number of example per batch')
tf.app.flags.DEFINE_integer('n_epoch', 10, 'number of epoch')
tf.app.flags.DEFINE_integer('n_hidden', 300, 'number of hidden unit')
tf.app.flags.DEFINE_integer('n_class', 3, 'number of distinct class')
tf.app.flags.DEFINE_integer('pre_processed', 0, 'Whether the data is pre-processed')
tf.app.flags.DEFINE_float('learning_rate', 0.01, 'learning rate')
tf.app.flags.DEFINE_float('l2_reg', 0.001, 'l2 regularization')
tf.app.flags.DEFINE_float('dropout', 0.5, 'dropout')

tf.app.flags.DEFINE_string('embedding_file_name', 'data/glove.840B.300d.txt', 'embedding file name')
tf.app.flags.DEFINE_string('dataset', 'data/restaurant/', 'the directory of dataset')

tf.app.flags.DEFINE_integer('max_aspect_len', 0, 'max length of aspects')
tf.app.flags.DEFINE_integer('max_context_len', 0, 'max length of contexts')
tf.app.flags.DEFINE_string('embedding_matrix', '', 'word ids to word vectors')


def main(_):
    start_time = time.time()

    print('Loading data info ...')
    word2id, FLAGS.max_aspect_len, FLAGS.max_context_len = get_data_info(FLAGS.dataset, FLAGS.pre_processed)

    print('Loading training data and testing data ...')
    train_data = read_data(word2id, FLAGS.max_aspect_len, FLAGS.max_context_len, FLAGS.dataset + 'train', FLAGS.pre_processed)
    test_data = read_data(word2id, FLAGS.max_aspect_len, FLAGS.max_context_len, FLAGS.dataset + 'test', FLAGS.pre_processed)

    print('Loading pre-trained word vectors ...')
    FLAGS.embedding_matrix = load_word_embeddings(FLAGS.embedding_file_name, FLAGS.embedding_dim, word2id)

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    with tf.Session(config=config) as sess:
        model = IAN(FLAGS, sess)
        model.build_model(train_data, test_data)
        model.run()

    end_time = time.time()
    print('Time Costing: %s' % (end_time - start_time))


if __name__ == '__main__':
    tf.app.run()
