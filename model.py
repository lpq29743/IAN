import tensorflow as tf
from tensorflow.python.ops import math_ops
import time
import math


class IAN(object):

    def __init__(self, config, sess):
        self.embedding_dim = config.embedding_dim
        self.batch_size = config.batch_size
        self.n_epoch = config.n_epoch
        self.n_hidden = config.n_hidden
        self.n_class = config.n_class
        self.learning_rate = config.learning_rate
        self.l2_reg = config.l2_reg
        self.dropout = config.dropout

        self.max_aspect_len = config.max_aspect_len
        self.max_context_len = config.max_context_len
        self.embedding_matrix = config.embedding_matrix
        self.sess = sess

    def build_model(self, train_data, test_data):
        with tf.name_scope('inputs'):
            self.train_data_size = len(train_data[0])
            train_data = tf.data.Dataset.from_tensor_slices(train_data)
            train_data = train_data.shuffle(buffer_size=self.train_data_size)
            train_data = train_data.batch(self.batch_size).repeat(self.n_epoch)

            self.test_data_size = len(test_data[0])
            test_data = tf.data.Dataset.from_tensor_slices(test_data)
            test_data = test_data.batch(self.batch_size)

            iterator = tf.data.Iterator.from_structure(train_data.output_types, test_data.output_shapes)
            self.aspects, self.contexts, self.labels, self.aspect_lens, self.context_lens = iterator.get_next()

            self.train_init_op = iterator.make_initializer(train_data)
            self.test_init_op = iterator.make_initializer(test_data)

            self.dropout_keep_prob = tf.placeholder(tf.float32)

            aspect_inputs = tf.nn.embedding_lookup(self.embedding_matrix, self.aspects)
            aspect_inputs = tf.cast(aspect_inputs, tf.float32)
            aspect_inputs = tf.nn.dropout(aspect_inputs, keep_prob=self.dropout_keep_prob)

            context_inputs = tf.nn.embedding_lookup(self.embedding_matrix, self.contexts)
            context_inputs = tf.cast(context_inputs, tf.float32)
            context_inputs = tf.nn.dropout(context_inputs, keep_prob=self.dropout_keep_prob)

        with tf.name_scope('weights'):
            weights = {
                'aspect_score': tf.get_variable(
                    name='W_a',
                    shape=[self.n_hidden, self.n_hidden],
                    initializer=tf.random_uniform_initializer(-0.1, 0.1),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                ),
                'context_score': tf.get_variable(
                    name='W_c',
                    shape=[self.n_hidden, self.n_hidden],
                    initializer=tf.random_uniform_initializer(-0.1, 0.1),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                ),
                'softmax': tf.get_variable(
                    name='W_l',
                    shape=[self.n_hidden * 2, self.n_class],
                    initializer=tf.random_uniform_initializer(-0.1, 0.1),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                ),
            }

        with tf.name_scope('biases'):
            biases = {
                'aspect_score': tf.get_variable(
                    name='B_a',
                    shape=[self.max_aspect_len, 1],
                    initializer=tf.zeros_initializer(),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                ),
                'context_score': tf.get_variable(
                    name='B_c',
                    shape=[self.max_context_len, 1],
                    initializer=tf.zeros_initializer(),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                ),
                'softmax': tf.get_variable(
                    name='B_l',
                    shape=[self.n_class],
                    initializer=tf.zeros_initializer(),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                ),
            }

        with tf.name_scope('dynamic_rnn'):
            aspect_outputs, aspect_state = tf.nn.dynamic_rnn(
                tf.contrib.rnn.LSTMCell(self.n_hidden),
                inputs=aspect_inputs,
                sequence_length=self.aspect_lens,
                dtype=tf.float32,
                scope='aspect_lstm'
            )
            batch_size = tf.shape(aspect_outputs)[0]
            aspect_avg = tf.reduce_mean(aspect_outputs, 1)

            context_outputs, context_state = tf.nn.dynamic_rnn(
                tf.contrib.rnn.LSTMCell(self.n_hidden),
                inputs=context_inputs,
                sequence_length=self.context_lens,
                dtype=tf.float32,
                scope='context_lstm'
            )
            context_avg = tf.reduce_mean(context_outputs, 1)

            aspect_outputs_iter = tf.TensorArray(tf.float32, 1, dynamic_size=True, infer_shape=False)
            aspect_outputs_iter = aspect_outputs_iter.unstack(aspect_outputs)
            context_avg_iter = tf.TensorArray(tf.float32, 1, dynamic_size=True, infer_shape=False)
            context_avg_iter = context_avg_iter.unstack(context_avg)
            aspect_lens_iter = tf.TensorArray(tf.int64, 1, dynamic_size=True, infer_shape=False)
            aspect_lens_iter = aspect_lens_iter.unstack(self.aspect_lens)
            aspect_rep = tf.TensorArray(size=batch_size, dtype=tf.float32)
            aspect_att = tf.TensorArray(size=batch_size, dtype=tf.float32)

            def body(i, aspect_rep, aspect_att):
                a = aspect_outputs_iter.read(i)
                b = context_avg_iter.read(i)
                l = math_ops.to_int32(aspect_lens_iter.read(i))
                aspect_score = tf.reshape(tf.nn.tanh(
                    tf.matmul(tf.matmul(a, weights['aspect_score']), tf.reshape(b, [-1, 1])) + biases['aspect_score']),
                    [1, -1])
                aspect_att_temp = tf.concat(
                    [tf.nn.softmax(tf.slice(aspect_score, [0, 0], [1, l])), tf.zeros([1, self.max_aspect_len - l])], 1)
                aspect_att = aspect_att.write(i, aspect_att_temp)
                aspect_rep = aspect_rep.write(i, tf.matmul(aspect_att_temp, a))
                return (i + 1, aspect_rep, aspect_att)

            def condition(i, aspect_rep, aspect_att):
                return i < batch_size

            _, aspect_rep_final, aspect_att_final = tf.while_loop(cond=condition, body=body,
                                                                  loop_vars=(0, aspect_rep, aspect_att))
            self.aspect_atts = tf.reshape(aspect_att_final.stack(), [-1, self.max_aspect_len])
            self.aspect_reps = tf.reshape(aspect_rep_final.stack(), [-1, self.n_hidden])

            context_outputs_iter = tf.TensorArray(tf.float32, 1, dynamic_size=True, infer_shape=False)
            context_outputs_iter = context_outputs_iter.unstack(context_outputs)
            aspect_avg_iter = tf.TensorArray(tf.float32, 1, dynamic_size=True, infer_shape=False)
            aspect_avg_iter = aspect_avg_iter.unstack(aspect_avg)
            context_lens_iter = tf.TensorArray(tf.int64, 1, dynamic_size=True, infer_shape=False)
            context_lens_iter = context_lens_iter.unstack(self.context_lens)
            context_rep = tf.TensorArray(size=batch_size, dtype=tf.float32)
            context_att = tf.TensorArray(size=batch_size, dtype=tf.float32)

            def body(i, context_rep, context_att):
                a = context_outputs_iter.read(i)
                b = aspect_avg_iter.read(i)
                l = math_ops.to_int32(context_lens_iter.read(i))
                context_score = tf.reshape(tf.nn.tanh(
                    tf.matmul(tf.matmul(a, weights['context_score']), tf.reshape(b, [-1, 1])) + biases[
                        'context_score']), [1, -1])
                context_att_temp = tf.concat(
                    [tf.nn.softmax(tf.slice(context_score, [0, 0], [1, l])), tf.zeros([1, self.max_context_len - l])],
                    1)
                context_att = context_att.write(i, context_att_temp)
                context_rep = context_rep.write(i, tf.matmul(context_att_temp, a))
                return (i + 1, context_rep, context_att)

            def condition(i, context_rep, context_att):
                return i < batch_size

            _, context_rep_final, context_att_final = tf.while_loop(cond=condition, body=body,
                                                                    loop_vars=(0, context_rep, context_att))
            self.context_atts = tf.reshape(context_att_final.stack(), [-1, self.max_context_len])
            self.context_reps = tf.reshape(context_rep_final.stack(), [-1, self.n_hidden])

            self.reps = tf.concat([self.aspect_reps, self.context_reps], 1)
            self.predict = tf.matmul(self.reps, weights['softmax']) + biases['softmax']

        with tf.name_scope('loss'):
            self.ce_cost = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.predict, labels=self.labels)
            self.total_cost = tf.reduce_sum(self.ce_cost)
            self.cost = tf.reduce_mean(self.ce_cost)
            self.global_step = tf.Variable(0, name="tr_global_step", trainable=False)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost,
                                                                                               global_step=self.global_step)

        with tf.name_scope('predict'):
            self.correct_pred = tf.equal(tf.argmax(self.predict, 1), tf.argmax(self.labels, 1))
            self.accuracy = tf.reduce_sum(tf.cast(self.correct_pred, tf.int64))

        summary_loss = tf.summary.scalar('loss', self.cost)
        summary_acc = tf.summary.scalar('acc', self.accuracy)
        self.train_summary_op = tf.summary.merge([summary_loss, summary_acc])
        self.test_summary_op = tf.summary.merge([summary_loss, summary_acc])
        timestamp = str(int(time.time()))
        _dir = 'logs/' + str(timestamp) + '_r' + str(self.learning_rate) + '_b' + str(self.batch_size) + '_l' + str(
            self.l2_reg)
        self.train_summary_writer = tf.summary.FileWriter(_dir + '/train', self.sess.graph)
        self.test_summary_writer = tf.summary.FileWriter(_dir + '/test', self.sess.graph)

    def analysis(self, train_data, test_data):
        timestamp = str(int(time.time()))

        aspects, contexts, labels, aspect_lens, context_lens = train_data
        with open('analysis/train_' + str(timestamp) + '.txt', 'w') as f:
            for sample, num in self.get_batch_data(aspects, contexts, labels, aspect_lens, context_lens, len(aspects),
                                                   False, 1.0):
                aspect_atts, context_atts, correct_pred = self.sess.run(
                    [self.aspect_atts, self.context_atts, self.correct_pred], feed_dict=sample)
                for a, b, c in zip(aspect_atts, context_atts, correct_pred):
                    a = str(a).replace('\n', '')
                    b = str(b).replace('\n', '')
                    f.write('%s\n%s\n%s\n' % (a, b, c))
        print('Finishing analyzing training data')

        aspects, contexts, labels, aspect_lens, context_lens = test_data
        with open('analysis/test_' + str(timestamp) + '.txt', 'w') as f:
            for sample, num in self.get_batch_data(aspects, contexts, labels, aspect_lens, context_lens, len(aspects),
                                                   False, 1.0):
                aspect_atts, context_atts, correct_pred = self.sess.run(
                    [self.aspect_atts, self.context_atts, self.correct_pred], feed_dict=sample)
                for a, b, c in zip(aspect_atts, context_atts, correct_pred):
                    a = str(a).replace('\n', '')
                    b = str(b).replace('\n', '')
                    f.write('%s\n%s\n%s\n' % (a, b, c))
        print('Finishing analyzing testing data')

    def run(self):
        saver = tf.train.Saver(tf.trainable_variables())

        print('Training ...')
        self.sess.run(tf.global_variables_initializer())
        self.sess.run([self.train_init_op, self.test_init_op])
        max_acc, step = 0., -1
        for i in range(self.n_epoch):
            cost, acc = 0., 0
            for _ in range(math.ceil(self.train_data_size / self.batch_size)):
                _, loss, accuracy, step, summary = self.sess.run(
                    [self.optimizer, self.total_cost, self.accuracy, self.global_step,
                     self.train_summary_op], feed_dict={self.dropout_keep_prob: self.dropout})
                cost += loss
                acc += accuracy
                self.train_summary_writer.add_summary(summary, step)

            train_loss = cost / self.train_data_size
            train_acc = acc / self.train_data_size

            cost, acc = 0., 0
            for _ in range(math.ceil(self.test_data_size / self.batch_size)):
                loss, accuracy, step, summary = self.sess.run([self.total_cost, self.accuracy, self.global_step, self.test_summary_op], feed_dict={self.dropout_keep_prob: 1.0})
                cost += loss
                acc += accuracy
                self.test_summary_writer.add_summary(summary, step)

            test_loss = cost / self.test_data_size
            test_acc = acc / self.test_data_size

            if test_acc > max_acc:
                max_acc = test_acc
                step = i
                saver.save(self.sess, 'models/model_iter', global_step=step)
            print('epoch %s: train-loss=%.6f; train-acc=%.6f; test-loss=%.6f; test-acc=%.6f;' % (
            str(i), train_loss, train_acc, test_loss, test_acc))

        saver.save(self.sess, 'models/model_final')
        print('The max accuracy of testing results is %s of step %s' % (max_acc, step))

        # print('Analyzing ...')
        # saver.restore(self.sess, tf.train.latest_checkpoint('models/'))
        # self.analysis(train_data, test_data)
