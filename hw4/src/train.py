# RNN with pretrained word2vec to do text sentiment classification

import argparse
import string
import os
import pickle
import gensim
import numpy as np
# import matplotlib.pyplot as plt

from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import GRU, Bidirectional
from keras.layers import Dense, Dropout
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

# GPU setting
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
set_session(sess)

# Global param setting
valid_num = 0

def load_data(label_path, nolabel_path):
    label_sentences = []
    nolabel_sentences = []
    labels = []

    with open(label_path) as f:
        for line in f:
            line = line.strip().split(' ')
            labels.append(int(line[0]))
            words = [word.lower() for word in line[2:] if not word.isdigit()]
            label_sentences.append(words)

    with open(nolabel_path) as f:
        for line in f:
            line = line.strip().split(' ')
            words = [word.lower() for word in line if not word.isdigit()]
            nolabel_sentences.append(words)

    return label_sentences, nolabel_sentences, labels

def embedding(label_sentences, labels):
    wv_model = gensim.models.Word2Vec.load('model/punc_word2vec_gensim_sg')
    train_x = []
    train_y = []
    for line in label_sentences:
        sentence = []
        for word in line:
            if word in wv_model.wv:
                sentence.append(wv_model.wv[word])
            else:
                sentence.append(np.zeros(200))
        train_x.append(sentence)

    train_x = pad_sequences(train_x, dtype=np.float32)
    train_y = to_categorical(np.asarray(labels))
    train_x, valid_x = train_x[valid_num:], train_x[:valid_num]
    train_y, valid_y = train_y[valid_num:], train_y[:valid_num]
    return (valid_x, valid_y), (train_x, train_y)

def add_rnn(model, units, drop_rate, is_input, is_last=False):
    params = {'units':units,
             'activation':'tanh',
             'dropout':drop_rate,
             'recurrent_dropout':drop_rate,
             }
    if not is_last:
        params['return_sequences'] = True
    if is_input:
        model.add(Bidirectional(GRU(**params), input_shape=(39, 200)))
    else:
        model.add(Bidirectional(GRU(**params)))

def add_fc(model, units, drop_rate):
    model.add(Dense(units=units))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.05))
    model.add(Dropout(drop_rate))

def RNN():
    model = Sequential()
    add_rnn(model, 128, 0.5, is_input=True)
    add_rnn(model, 128, 0.5, is_input=False, is_last=True)

    add_fc(model, 256, 0.5)
    add_fc(model, 128, 0.5)
    add_fc(model, 64, 0.5)

    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', \
        optimizer= RMSprop(lr = 0.001), metrics=['accuracy'])
    return model

def train(model, train_x, train_y, valid_x, valid_y):
    BATCH_SIZE = 128
    EPOCHS = 18
    callbacks = [
        ModelCheckpoint('model/punc_weights.{epoch:02d}-{val_acc:.3f}.hdf5',\
                monitor='val_acc', save_best_only=True, period=1, verbose=0),]
    history = model.fit(train_x, train_y, batch_size=BATCH_SIZE, epochs=EPOCHS,\
                            validation_data=(valid_x, valid_y), callbacks=callbacks)
    return history

def evaluate(model, train_x, train_y, valid_x, valid_y):
    result_train = model.evaluate(train_x, train_y)
    print("\nTrain Acc:",result_train[1])
    result_valid = model.evaluate(valid_x, valid_y)
    print("\nValid Acc:",result_valid[1])

# def plot_history(history):
#     plt.plot(history.history['acc'])
#     plt.plot(history.history['val_acc'])
#     plt.title('Training Procedure')
#     plt.ylabel('Accuracy')
#     plt.xlabel('Epoch')
#     plt.legend(['train', 'valid'], loc='upper left')
#     plt.show()

def main(args):
    global valid_num
    valid_num = args.valid_num
    label_sentences, nolabel_sentences, labels = \
    load_data(args.label_path, args.nolabel_path)
    (valid_x, valid_y), (train_x, train_y) = embedding(label_sentences, labels)

    print('train_x.shape:'+str(train_x.shape))
    print('valid_x.shape:'+str(valid_x.shape))

    model = RNN()
    history = train(model, train_x, train_y, valid_x, valid_y)
    evaluate(model, train_x, train_y, valid_x, valid_y)
    # plot_history(history)

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--train_label_path', type=str,
                        default='data/training_label.txt', dest='label_path',
                        help='Path to label data')
    parser.add_argument('-n', '--train_nolabel_path', type=str,
                        default='data/training_nolabel.txt', dest='nolabel_path',
                        help='Path to nolabel data')
    parser.add_argument('-v', '--valid_num', type=int, default=20000,
                         dest='valid_num', help='Size of validation data')
    main(parser.parse_args())
