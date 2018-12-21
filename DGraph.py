from collections import namedtuple

import tensorflow as tf

from hyperparams import Hyperparams as hp
from data_load import get_batch_data, load_doc_vocab, load_sum_vocab
from rouge_tensor import rouge_l_fscore
from modules import *

io_pairs = namedtuple(typename='io_pairs', field_names='input output')

class Graph():
    def __init__(self, is_training=True, T_Decoder=False):
        self.graph = tf.Graph()
        self.vocab_size = len(load_doc_vocab()[0])  # load_doc_vocab returns: de2idx, idx2de

        with self.graph.as_default():
            if is_training:
                self.xy, self.num_batch = get_batch_data(T_Decoder=True)  # (N, article_T+summary_T+1), ()
            else:  # inference
                self.xy = tf.placeholder(tf.int32, shape=(None, hp.article_maxlen+hp.summary_maxlen+1))

            self.decoder_inputs = tf.concat((tf.ones_like(self.xy[:, :1]) * 2, self.xy[:, :-1]), -1)  # 2:<S> # define decoder inputs

            # self._add_encoder(is_training=is_training)
            self.ml_loss = self._add_ml_loss(is_training=is_training)
            self.loss = self.ml_loss

            if is_training:
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.optimizer = tf.train.AdamOptimizer(learning_rate=hp.lr, beta1=0.9, beta2=0.98, epsilon=1e-8)

                grads_and_vars_ml = self.optimizer.compute_gradients(loss=self.ml_loss)
                grad_ml, vars_ml = zip(*grads_and_vars_ml) # parse grad and var

                # add gradient clipping
                clipped_grad_ml, globle_norm_ml = tf.clip_by_global_norm(grad_ml, hp.maxgradient)
                self.globle_norm_ml = globle_norm_ml
                self.train_op_ml  = self.optimizer.apply_gradients(grads_and_vars=zip(clipped_grad_ml, vars_ml),
                                                                   global_step=self.global_step)
                '''
                # training wihtout gradient clipping
                self.train_op_ml  = self.optimizer.apply_gradients(grads_and_vars=grads_and_vars_ml,
                                                                   global_step=self.global_step)
                '''

                # Summary
                tf.summary.scalar('globle_norm_ml', globle_norm_ml)
                tf.summary.scalar('loss', self.loss)

                self.merged = tf.summary.merge_all()

        self.filewriter = tf.summary.FileWriter(hp.tb_dir + '/train', self.graph)


    def _add_decoder(self, is_training, decoder_inputs, inside_loop=False, reuse=None):
        with self.graph.as_default():

            # Decoder
            self.dec = embedding(decoder_inputs,
                                 vocab_size=self.vocab_size,
                                 num_units=hp.hidden_units,
                                 # lookup_table=self.get_embedding_table(concated=True),
                                 scale=True,
                                 scope="decoder_embed",
                                 reuse=reuse)

            self.batch_outp_emb = self.dec

            with tf.variable_scope("decoder"):
                ## Positional Encoding
                if hp.sinusoid:
                    self.dec += positional_encoding(decoder_inputs,
                                                    num_units=hp.hidden_units,
                                                    zero_pad=False,
                                                    scale=False,
                                                    scope="dec_pe",
                                                    reuse=reuse)
                else:
                    self.dec += embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(decoder_inputs)[1]), 0),
                                                  [tf.shape(decoder_inputs)[0], 1]),
                                          vocab_size=hp.article_maxlen+hp.summary_maxlen+1,
                                          num_units=hp.hidden_units,
                                          zero_pad=False,
                                          scale=False,
                                          scope="dec_pe",
                                          reuse=reuse)

                ## Dropout
                self.dec = tf.layers.dropout(self.dec,
                                             rate=hp.dropout_rate,
                                             training=tf.convert_to_tensor(is_training))

                ## Blocks
                for i in range(hp.num_blocks):
                    with tf.variable_scope("num_blocks_{}".format(i)):
                        # remove the input embedding
                        self.dec = multihead_attention(queries=self.dec,
                                                       keys=self.dec,
                                                       num_units=hp.hidden_units,
                                                       num_heads=hp.num_heads,
                                                       dropout_rate=hp.dropout_rate,
                                                       is_training=is_training,
                                                       causality=True,
                                                       scope="self_attention",
                                                       inside_loop=inside_loop,
                                                       reuse=reuse)

                        self.dec = feedforward(self.dec,
                                               num_units=[hp.ffw_unit, hp.hidden_units],
                                               inside_loop=inside_loop,
                                               reuse=reuse)

            # Final linear projection
            self.logits = tf.layers.dense(self.dec, self.vocab_size, name='final_output_dense', reuse=reuse)
            return self.logits


    def _add_ml_loss(self, is_training):
        logits = self._add_decoder(is_training=is_training, decoder_inputs=self.decoder_inputs)

        with self.graph.as_default():
            self.preds = tf.to_int32(tf.argmax(logits, axis=-1)) # shape: (batch_size, max_timestep)
            self.istarget = tf.to_float(tf.not_equal(self.xy, 0)) # shape: (batch_size, max_timestep)
            self.acc_full = tf.reduce_sum(tf.to_float(tf.equal(self.preds, self.xy)) * self.istarget) / (
                tf.reduce_sum(self.istarget))
            self.acc_sum = tf.reduce_sum(tf.to_float(tf.equal(self.preds[hp.article_maxlen+1:], self.xy[hp.article_maxlen+1:])) * self.istarget) / (
                tf.reduce_sum(self.istarget))

            self.rouge = tf.reduce_sum(rouge_l_fscore(self.preds[hp.article_maxlen+1:], self.xy[hp.article_maxlen+1:])) / float(hp.batch_size)

            tf.summary.scalar('acc in full range', self.acc_full)
            tf.summary.scalar('acc only for summary part', self.acc_sum)

            tf.summary.scalar('rouge', self.rouge)

            ml_loss = -100
            if is_training:
                # Loss
                self.y_smoothed = label_smoothing(tf.one_hot(self.xy, depth=self.vocab_size))
                loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.y_smoothed)
                ml_loss = tf.reduce_sum(loss * self.istarget, name='fake_ml_loss') / (tf.reduce_sum(self.istarget))

        return ml_loss


    def get_filewriter(self):
        return self.filewriter

    def get_input_output(self, is_training=False):
        if is_training:
            return io_pairs(input=[self.x, self.y], output=[self.logits, self.loss])
        else:
            return io_pairs(input=[self.x, self.y], output=[self.logits])

    def get_embedding_table(self, scope='', concated=False):
        if not concated:
            with tf.variable_scope(scope, reuse=True):  # get lookup table
                lookup_table = tf.get_variable('lookup_table')
        else:
            lookup_table = self.graph.get_tensor_by_name("concated_lookup_table:0")

        return lookup_table

    def get_batch_embedding(self):
        """ only return the embedding of current batch"""
        return [self.batch_inp_emb, self.batch_outp_emb]

    def get_gradients(self):
        return self.grads_and_vars
