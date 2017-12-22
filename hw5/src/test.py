import argparse
import numpy as np
import os

from keras.models import load_model
import keras.backend as K

# GPU setting
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
set_session(sess)

def load_data(test_path):
    users = []
    movies = []
    with open(test_path) as f:
        for n, line in enumerate(f):
            if n == 0: # line 0 contains no data
                continue
            line = line.strip().split(',')
            user, movie = int(line[1]), int(line[2])
            users.append(int(line[1]))
            movies.append(int(line[2]))
    return np.asarray(users, dtype=np.int16), np.asarray(movies, dtype=np.int16)

def rmse(y_true, y_pred, use_np=False):
    if use_np:
        return np.sqrt(((y_pred - y_true)**2).mean())
    else:
        return K.sqrt(K.mean(K.pow(y_pred - y_true, 2)))

def predict(model, test_users, test_movies, output_path):
    result = model.predict([test_users, test_movies])
    result[result>5] = 5.0
    result[result<1] = 1.0
    with open(output_path, 'w') as f_out:
            print('TestDataID,Rating', file=f_out)
            for i in range(1, result.shape[0]+1):
                print('%d,%.1f' %(i, result[i-1]), file=f_out)

def main(args):
    test_users, test_movies = load_data(args.test_path)

    if args.output_path is None:
        print('Error: Argument --output_path for the path of prediction result')
        return

    model = load_model(args.model_path, custom_objects={'rmse': rmse})

    predict(model, test_users, test_movies, args.output_path)

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--test_data_path', type=str,
                        default='data/test.csv', dest='test_path',
                        help='Path to testing data')
    parser.add_argument('-l', '--model_path', type=str,
                        default='model/MF_512d.42-0.853.hdf5', dest='model_path',
                        help='Path to load the model')
    parser.add_argument('-o', '--output_path', type=str, dest='output_path',
                        help='Path to save the prediction result')
    main(parser.parse_args())
