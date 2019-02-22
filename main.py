import tensorflow as tf
from utils import get_data_info, read_data, load_word_embeddings
from model import IAN
from evals import *
import os
import time
import math

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '4'

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
tf.enable_eager_execution(config=config)

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('embedding_dim', 300, 'dimension of word embedding')
tf.app.flags.DEFINE_integer('n_hidden', 300, 'number of hidden unit')
tf.app.flags.DEFINE_integer('n_class', 3, 'number of distinct class')
tf.app.flags.DEFINE_float('l2_reg', 0.001, 'l2 regularization')

tf.app.flags.DEFINE_integer('max_aspect_len', 0, 'max length of aspects')
tf.app.flags.DEFINE_integer('max_context_len', 0, 'max length of contexts')
tf.app.flags.DEFINE_string('embedding_matrix', '', 'word ids to word vectors')

batch_size = 128
learning_rate = 0.01
n_epoch = 20
pre_processed = 1
embedding_file_name = 'data/glove.840B.300d.txt'
dataset = 'data/laptop/'
logdir = 'logs/'


def run(model, train_data, test_data):
    print('Training ...')
    max_acc, max_f1, step = 0., 0., -1

    train_data_size = len(train_data[0])
    train_data = tf.data.Dataset.from_tensor_slices(train_data)
    train_data = train_data.shuffle(buffer_size=train_data_size).batch(batch_size, drop_remainder=True)

    test_data_size = len(test_data[0])
    test_data = tf.data.Dataset.from_tensor_slices(test_data)
    test_data = test_data.batch(batch_size, drop_remainder=True)

    iterator = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    writer = tf.contrib.summary.create_file_writer(logdir)
    writer.set_as_default()

    for i in range(n_epoch):
        cost, predict_list, labels_list = 0., [], []
        iterator.make_initializer(train_data)
        for _ in range(math.floor(train_data_size / batch_size)):
            data = iterator.get_next()
            with tf.GradientTape() as tape:
                predict, labels = model(data, dropout=0.5)
                loss_t = tf.nn.softmax_cross_entropy_with_logits_v2(logits=predict, labels=labels)
                loss = tf.reduce_mean(loss_t)
                cost += tf.reduce_sum(loss_t)
            grads = tape.gradient(loss, model.variables)
            optimizer.apply_gradients(zip(grads, model.variables))
            predict_list.extend(tf.argmax(tf.nn.softmax(predict), 1).numpy())
            labels_list.extend(tf.argmax(labels, 1).numpy())
        train_acc, train_f1, _, _ = evaluate(pred=predict_list, gold=labels_list)
        train_loss = cost / train_data_size
        tf.contrib.summary.scalar('train_loss', train_loss)
        tf.contrib.summary.scalar('train_acc', train_acc)
        tf.contrib.summary.scalar('train_f1', train_f1)

        cost, predict_list, labels_list = 0., [], []
        iterator.make_initializer(test_data)
        for _ in range(math.floor(test_data_size / batch_size)):
            data = iterator.get_next()
            predict, labels = model(data, dropout=1.0)
            loss_t = tf.nn.softmax_cross_entropy_with_logits_v2(logits=predict, labels=labels)
            cost += tf.reduce_sum(loss_t)
            predict_list.extend(tf.argmax(tf.nn.softmax(predict), 1).numpy())
            labels_list.extend(tf.argmax(labels, 1).numpy())
        test_acc, test_f1, _, _ = evaluate(pred=predict_list, gold=labels_list)
        test_loss = cost / test_data_size
        tf.contrib.summary.scalar('test_loss', test_loss)
        tf.contrib.summary.scalar('test_acc', test_acc)
        tf.contrib.summary.scalar('test_f1', test_f1)

        if test_acc + test_f1 > max_acc + max_f1:
            max_acc = test_acc
            max_f1 = test_f1
            step = i
            saver = tf.contrib.eager.Saver(model.variables)
            saver.save('models/model_iter', global_step=step)
        print(
            'epoch %s: train-loss=%.6f; train-acc=%.6f; train-f1=%.6f; test-loss=%.6f; test-acc=%.6f; test-f1=%.6f.' % (
                str(i), train_loss, train_acc, train_f1, test_loss, test_acc, test_f1))

    saver.save('models/model_final')
    print('The max accuracy of testing results: acc %.6f and macro-f1 %.6f of step %s' % (max_acc, max_f1, step))


def main(_):
    start_time = time.time()

    print('Loading data info ...')
    word2id, FLAGS.max_aspect_len, FLAGS.max_context_len = get_data_info(dataset, pre_processed)

    print('Loading training data and testing data ...')
    train_data = read_data(word2id, FLAGS.max_aspect_len, FLAGS.max_context_len, dataset + 'train', pre_processed)
    test_data = read_data(word2id, FLAGS.max_aspect_len, FLAGS.max_context_len, dataset + 'test', pre_processed)

    print('Loading pre-trained word vectors ...')
    FLAGS.embedding_matrix = load_word_embeddings(embedding_file_name, FLAGS.embedding_dim, word2id)

    model = IAN(FLAGS)
    run(model, train_data, test_data)

    end_time = time.time()
    print('Time Costing: %s' % (end_time - start_time))


if __name__ == '__main__':
    tf.app.run()
