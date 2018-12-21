# -*- coding: utf-8 -*-
#/usr/bin/python3
# TRANSFORMER + RL (nsml version)

from __future__ import print_function
import os, codecs
import logging

import numpy as np
import tensorflow as tf
# from nltk.translate.bleu_score import corpus_bleu

from modules import *
from DGraph import Graph
from hyperparams import Hyperparams as hp
from data_load import load_doc_vocab, load_sum_vocab, load_data
from rouge_tensor import rouge_l_sentence_level

def train():
    if not os.path.exists(hp.logdir):
        os.mkdir(hp.logdir)

    # Load vocabulary
    """
    de2idx, idx2de = load_doc_vocab()
    en2idx, idx2en = load_sum_vocab()
    """

    print("Constructing graph...")
    train_g = Graph("train")

    print("Start training...")
    with train_g.graph.as_default():
        sv = tf.train.Supervisor(logdir=hp.logdir)
        config = tf.ConfigProto(allow_soft_placement=True,
                                log_device_placement=False)
        config.gpu_options.allow_growth = True

        with sv.managed_session(config=config) as sess:
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            print("Start training: epoches={}, num batch={}".format(hp.num_epochs, train_g.num_batch))
            for epoch in range(1, hp.num_epochs+1):
                print("Starting {}-th epoch".format(epoch))

                if sv.should_stop():
                    break

                for step in range(train_g.num_batch):
                    true_step = step + (epoch - 1) * train_g.num_batch

                    if true_step % hp.train_record_steps == 0:
                        summary, _ = sess.run([train_g.merged, train_g.train_op_ml])
                        train_g.filewriter.add_summary(summary, true_step)
                    else:
                        sess.run(train_g.train_op_ml)

                    if true_step % hp.checkpoint_steps == 0:
                        sv.saver.save(sess, hp.logdir + '/model_epoch_%02d_step_%d' % (epoch, true_step))

                    if true_step > 0 and true_step % hp.eval_record_steps == 0:
                        # pass
                        eval(cur_step=true_step, write_file=False)

                    # iteration indent
                # epoch indent
                # eval(cur_step=true_step, write_file=True)
    print("Done")



def eval(type='eval', cur_step=0, write_file=True):
    # Load graph
    g = Graph(is_training=False)
    print("Eval Graph loaded")

    # Load data
    X, Sources, Targets = load_data(type=type)

    word2idx, idx2word = load_sum_vocab()


    # Start session
    with g.graph.as_default():
        sv = tf.train.Supervisor()
        with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            sv.saver.restore(sess, tf.train.latest_checkpoint(hp.logdir))
            print("Restored!")

            mname = open(hp.logdir + '/checkpoint', 'r').read().split('"')[1] # model name

            ## Inference
            if not os.path.exists('results'): os.mkdir('results')

            if write_file:
                ### Write to file
                fout2 = codecs.open("results/eval-pred", "w", "utf-8")
                fout3 = codecs.open("results/eval-title", "w", "utf-8")
                fout =  codecs.open("results/eval-" + type + "_" + mname, "w", "utf-8")

            list_of_refs, hypotheses = [], []
            num_batch = len(X) // hp.batch_size
            print("num batch: ", num_batch, "len(X): ", len(X) )
            for i in range(num_batch):
                ### Get mini-batches
                x = X[i*hp.batch_size: (i+1)*hp.batch_size]
                sources = Sources[i*hp.batch_size: (i+1)*hp.batch_size]
                targets = Targets[i*hp.batch_size: (i+1)*hp.batch_size]

                ### Autoregressive inference
                seperate_symbol = 0 - np.ones((hp.batch_size, 1))
                preds = np.zeros((hp.batch_size, hp.summary_maxlen), np.int32)
                preds = np.concatenate([x, seperate_symbol, preds], axis=-1)

                for j in range(hp.article_maxlen+1, hp.article_maxlen+1+hp.summary_maxlen):
                    _preds = sess.run(g.preds, {g.xy: preds})
                    preds[:, j] = _preds[:, j]

                for source, target, pred in zip(sources, targets, preds): # sentence-wise
                    got = " ".join(idx2word[idx] for idx in pred).split("</S>")[0].strip()

                    if write_file:
                        sentence_to_write = "-source: {}\n-expected: {}\n-got: {}\n\n".format(source, target, got)

                        print(sentence_to_write)
                        fout.write(sentence_to_write)
                        fout2.write(got.strip() + '\n')
                        fout3.write(target.strip() + '\n')

                        fout.flush()
                        fout2.flush()
                        fout3.flush()

                    # bleu score
                    ref = target.split()
                    hypothesis = got.split()
                    if len(ref) > 3 and len(hypothesis) > 3:
                        list_of_refs.append(ref)
                        hypotheses.append(hypothesis)

            ## Calculate bleu score and rouge
            rouge = rouge_l_sentence_level(hypotheses, list_of_refs)
            # (eval_sentences, ref_sentences):

            rouge = np.mean(rouge)

    return None


if __name__ == '__main__':
    logging.info("START")
    logging.info("tensorflow version: ", tf.__version__)

    from tensorflow.python.client import device_lib as dl
    logging.info("local device: ", dl.list_local_devices())

    train()
    # eval()
