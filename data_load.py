# -*- coding: utf-8 -*-
#/usr/bin/python3

# build-in
from __future__ import print_function
import codecs
import logging

# external library
import tensorflow as tf
import numpy as np
import regex

# self-defined
from hyperparams import Hyperparams as hp


def load_doc_vocab():
    logging.info("Loading doc vocab...")
    word2idx = {}
    idx2word = {}
    for line in codecs.open(hp.doc_dict, 'r', 'utf-8').readlines():
        idx, word = line.split()
        word2idx[word] = int(idx)
        idx2word[int(idx)] = word

    print(idx)
    print("Size of doc dict: {}".format(len(word2idx)))
    return word2idx, idx2word


def load_sum_vocab():
    logging.info("Loading sum vocab...")
    word2idx = {}
    idx2word = {}
    for line in codecs.open(hp.sum_dict, 'r', 'utf-8').readlines():
        idx, word = line.split()
        word2idx[word] = int(idx)
        idx2word[int(idx)] = word

    print("Size of doc dict: {}".format(len(word2idx)))
    return word2idx, idx2word


def create_data(source_path, target_path, T_Decoder=False):
    """
    Args:
        source_path (str):  path to source article file
        target_path (str):  path to target summary file
        T_Decoder (bool):   whether to create data for Encoder-Decoder
            transformer or Decoder-only transformer
    Returns:
        (X, Y, Sources, Targets) or (XY, Sources, Targets) depends on the
        value of T_Decoder;
        where each entry in XY is the concatenation of each entry in X and Y
        i.e. xy = (x, special_symbol, y)
    """
    logging.info("Creating data...")

    article2idx, idx2article = load_doc_vocab()
    sum2idx, idx2sum = load_sum_vocab()

    source_file = open(source_path, 'r', encoding='utf-8')
    target_file = open(target_path, 'r', encoding='utf-8')

    if not T_Decoder:
        X, Y, Sources, Targets = [], [], [], []
    else:
        XY, Sources, Targets = [], [], []
    cur_ariticle_idx = 0

    while True:
        source_sent = source_file.readline()
        target_sent = target_file.readline()

        if not source_sent:
            if target_sent:
                raise ValueError("inconsistent number of articles in source and target file")
            break

        if cur_ariticle_idx % 1000000 == 0:
            print("\tPreparing {}-th article matrix".format(cur_ariticle_idx))

        # if cur_ariticle_idx == 400:
        #     break  # TEMP

        source_sent = source_sent.split()
        target_sent = target_sent.split()

        # remove short sentences & chop long sentences
        if len(source_sent) < hp.article_minlen or len(target_sent) < hp.summary_minlen:
            continue

        if len(source_sent) >= hp.article_maxlen:
            source_sent = source_sent[:(hp.article_maxlen-1)] # 1 for </S>

        if len(target_sent) >= hp.summary_maxlen:
            target_sent = target_sent[:(hp.summary_maxlen-1)]

        x = [article2idx.get(word, 1) for word in (source_sent + [u"</S>"])]
        y = [sum2idx.get(word, 1) for word in (target_sent + [u"</S>"]) ]

        if len(x) <= hp.article_maxlen:
            x = np.lib.pad(x, [0, hp.article_maxlen - len(x)], 'constant', constant_values=(0, 0))
        if len(y) <= hp.summary_maxlen:
            y = np.lib.pad(y, [0, hp.summary_maxlen - len(y)], 'constant', constant_values=(0, 0))

        try:
            assert len(x) == hp.article_maxlen
            assert len(y) == hp.summary_maxlen
        except AssertionError as error:
            print("current article length: ", len(x), "current article maxlen: ", hp.article_maxlen)
            print("current summary length: ", len(y), "current summary maxlen: ", hp.summary_maxlen)

        if not T_Decoder:
            X.append(x)
            Y.append(y)
        else:
            # here x and y are both 1D array
            xy = np.concatenate([x, [4], y], axis=0)  ### TODO (un hardcode 4 )
            XY.append(xy)

        Sources.append(" ".join(source_sent).strip())
        Targets.append(" ".join(target_sent).strip())

        cur_ariticle_idx += 1

    source_file.close()
    target_file.close()

    if not T_Decoder:
        X = np.array(X)
        Y = np.array(Y)
        print("number of data: ", X.shape, Y.shape)
        return X, Y, Sources, Targets
    else:
        XY = np.array(XY)
        print("number of data: ", XY.shape)
        return XY, Sources, Targets


def create_test_data(source_sents):
    """
    only for giga summary data
    """
    print("Creating data...")
    article2idx, idx2article = load_doc_vocab()

    doc_sents = list(map(lambda line: line.split(), source_sents))

    # Index
    X, Sources = [], []

    cur_ariticle_idx = 0
    for source_sent in doc_sents:
        if cur_ariticle_idx % 100000 == 0:
            print("\tPreparing {}-th article matrix".format(cur_ariticle_idx))

        # if cur_ariticle_idx == 200:
        #   break  # TEMP

        x = [article2idx.get(word, 1) for word in (source_sent + [u"</S>"]) ]

        if len(x) <= hp.article_maxlen:
            x = np.lib.pad(x, [0, hp.article_maxlen - len(x)], 'constant', constant_values=(0, 0))

            try:
                assert len(x) == hp.article_maxlen
            except AssertionError as error:
                print("current article length: ", len(x), "current article maxlen: ", hp.article_maxlen)

            X.append(x)
            Sources.append(" ".join(source_sent).strip())
        cur_ariticle_idx += 1
    X = np.array(X)
    return X, Sources


def load_data(type='train', T_Decoder=False):
    LEGAL_TYPE = ('train', 'eval', 'test', 'eval_tmp')  # TODO: remove test
    if type not in LEGAL_TYPE:
        raise TypeError('Invalid type: should be train/test/eval/eval_tmp')

    if type == 'train' or type == 'eval_tmp':
        doc_path = hp.source_train
        sum_path = hp.target_train
    elif type == 'eval':
        doc_path = hp.source_valid
        sum_path = hp.target_valid
    elif type == 'test':
        doc_path = hp.source_test

    if type == 'train':
        if not T_Decoder:
            X, Y, Sources, Targets = create_data(doc_path, sum_path, T_Decoder)
            return X, Y
        else:
            XY, Sources, Targets = create_data(doc_path, sum_path, T_Decoder)
            return XY
    elif type == 'eval' or type == 'eval_tmp':
        # since in the eval stage, we only need the article data
        X, Y, Sources, Targets = create_data(doc_path, sum_path, T_Decoder=False)
        return X, Sources, Targets


def get_batch_data(T_Decoder=False):
    print("getting batch_data...")
    if not T_Decoder:
        X, Y = load_data(type='train', T_Decoder=T_Decoder)
        num_batch = len(X) // hp.batch_size
        X = tf.convert_to_tensor(X, tf.int32)
        Y = tf.convert_to_tensor(Y, tf.int32)
        input_queues = tf.train.slice_input_producer([X, Y])    # Produces a slice of each `Tensor` in `tensor_list`
        x, y = tf.train.shuffle_batch(input_queues,
                                    num_threads=8,
                                    batch_size=hp.batch_size,
                                    capacity=hp.batch_size*64,
                                    min_after_dequeue=hp.batch_size*32,
                                    allow_smaller_final_batch=False)
        return x, y, num_batch # (N, T), (N, T), ()
    else:
        XY = load_data(type='train', T_Decoder=T_Decoder)
        num_batch = len(XY) // hp.batch_size
        XY = tf.convert_to_tensor(XY, tf.int32)
        input_queues = tf.train.slice_input_producer([XY])    # Produces a slice of each `Tensor` in `tensor_list`
        xy = tf.train.shuffle_batch(input_queues,
                                    num_threads=8,
                                    batch_size=hp.batch_size,
                                    capacity=hp.batch_size*64,
                                    min_after_dequeue=hp.batch_size*32,
                                    allow_smaller_final_batch=False)
        # when input_queues only has length 1, it seems that shuffle_batch()
        # will return a single Tensor() orther than a list
        return xy, num_batch # (N, article_T+summary_T+1), ()


if __name__ == '__main__':
    # XY = load_data(type='train', T_Decoder=True)
    # print(XY.shape)

    get_batch_data(T_Decoder=True)  # generate a Tensor
    get_batch_data(T_Decoder=False) # generate a List
