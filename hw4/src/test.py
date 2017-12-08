import argparse
import os
import string
import pickle
import gensim
import numpy as np

from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

# GPU setting
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
set_session(sess)

def load_data(test_path):
    test_sentences = []
    with open(test_path) as f:
        for i, line in enumerate(f):
            if i != 0:
                words = []
                for segment in line.strip().split(',')[1:]:
                    words.extend([word.lower() for word in segment.split(' ') if not word.isdigit()])
                test_sentences.append(words)
    return embedding(test_sentences)

def embedding(test_sentences):
    wv_model = gensim.models.Word2Vec.load('model/punc_word2vec_gensim_sg')
    test_x = []
    for line in test_sentences:
        sentence = []
        for word in line:
            if word in wv_model.wv:
                sentence.append(wv_model.wv[word])
            else:
                sentence.append(np.zeros(200))
        test_x.append(sentence)
    test_x = pad_sequences(test_x, maxlen=39, dtype=np.float32)
    return test_x

def predict(model, test_x, output_path):
    result = model.predict(test_x, batch_size=256)
    with open(output_path, 'w') as f_out:
        print('id,label', file=f_out)
        for i in range(test_x.shape[0]):
            label = np.argmax(result[i])
            print('%d,%d' %(i, label), file=f_out)

def main(args):
    test_x = load_data(args.test_path)
    print('test_x.shape:'+str(test_x.shape))

    if args.output_path is None:
        print('Error: Argument --output_path for the path of prediction result')
        return

    model = load_model(args.model_path)

    predict(model, test_x, args.output_path)

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--test_data_path', type=str,
                        default='data/testing_data.txt', dest='test_path',
                        help='Path to testing data')
    parser.add_argument('-l', '--model_path', type=str,
                        default='model/punc_weights.18-0.827.hdf5', dest='model_path',
                        help='Path to load the model')
    parser.add_argument('-o', '--output_path', type=str, dest='output_path',
                        help='Path to save the prediction result')
    main(parser.parse_args())
