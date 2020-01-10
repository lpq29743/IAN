import tensorflow as tf


class IAN(tf.keras.Model):

    def __init__(self, config):
        super(IAN, self).__init__()

        self.embedding_dim = config.embedding_dim
        self.n_hidden = config.n_hidden
        self.n_class = config.n_class
        self.l2_reg = config.l2_reg

        self.max_aspect_len = config.max_aspect_len
        self.max_context_len = config.max_context_len
        self.embedding_matrix = config.embedding_matrix

        self.aspect_lstm = tf.keras.layers.LSTM(self.n_hidden,
                                                return_sequences=True,
                                                recurrent_initializer='glorot_uniform',
                                                stateful=True)
        self.context_lstm = tf.keras.layers.LSTM(self.n_hidden,
                                                 return_sequences=True,
                                                 recurrent_activation='sigmoid',
                                                 recurrent_initializer='glorot_uniform',
                                                 stateful=True)

        self.aspect_w = tf.contrib.eager.Variable(tf.random_normal([self.n_hidden, self.n_hidden]), name='aspect_w')
        self.aspect_b = tf.contrib.eager.Variable(tf.zeros([self.n_hidden]), name='aspect_b')
        self.context_w = tf.contrib.eager.Variable(tf.random_normal([self.n_hidden, self.n_hidden]), name='context_w')
        self.context_b = tf.contrib.eager.Variable(tf.zeros([self.n_hidden]), name='context_b')
        self.output_fc = tf.keras.layers.Dense(self.n_class, kernel_regularizer=tf.keras.regularizers.l2(l=self.l2_reg))

    def call(self, data, dropout=0.5):
        aspects, contexts, labels, aspect_lens, context_lens = data
        aspect_inputs = tf.nn.embedding_lookup(self.embedding_matrix, aspects)
        aspect_inputs = tf.cast(aspect_inputs, tf.float32)
        aspect_inputs = tf.nn.dropout(aspect_inputs, keep_prob=dropout)

        context_inputs = tf.nn.embedding_lookup(self.embedding_matrix, contexts)
        context_inputs = tf.cast(context_inputs, tf.float32)
        context_inputs = tf.nn.dropout(context_inputs, keep_prob=dropout)

        aspect_outputs = self.aspect_lstm(aspect_inputs)
        aspect_avg = tf.reduce_mean(aspect_outputs, 1)

        context_outputs = self.context_lstm(context_inputs)
        context_avg = tf.reduce_mean(context_outputs, 1)

        aspect_att = tf.nn.softmax(tf.nn.tanh(tf.einsum('ijk,kl,ilm->ijm', aspect_outputs, self.aspect_w,
                                                        tf.expand_dims(context_avg, -1)) + self.aspect_b),
                                   axis=1)
        aspect_rep = tf.reduce_sum(aspect_att * aspect_outputs, 1)
        context_att = tf.nn.softmax(tf.nn.tanh(tf.einsum('ijk,kl,ilm->ijm', context_outputs, self.context_w,
                                                         tf.expand_dims(aspect_avg, -1)) + self.context_b),
                                    axis=1)
        context_rep = tf.reduce_sum(context_att * context_outputs, 1)

        rep = tf.concat([aspect_rep, context_rep], 1)
        predict = self.output_fc(rep)

        return predict, labels
