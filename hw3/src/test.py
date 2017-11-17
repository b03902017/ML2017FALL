import argparse
import csv
import os
import numpy as np

from keras.utils import np_utils
from keras.models import load_model

# GPU setting
# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
# set_session(sess)

def load_data(test_path):
    test_x = []
    with open(test_path) as f_test:
        for row_n, row in enumerate(csv.reader(f_test)):
            if row_n != 0: # row0 contains no data
                row = row[1].split(' ')
                test_x.append(row)
    test_x = np.array(test_x, dtype=np.float32)
    test_x = test_x/255

    return test_x

def predict(model, test_x, output_path):
    res = model.predict(test_x)
    with open(output_path, 'w') as f_out:
        print('id,label', file=f_out)
        for i in range(test_x.shape[0]):
            label = np.argmax(res[i])
            print('%d,%d' %(i, label), file=f_out)

def main(args):
    test_x = load_data(args.test_path)
    print('test_x.shape:'+str(test_x.shape))
    if args.output_path is None:
        print('Error: Argument --output_path for the path of prediction result')
        return
    if args.model is None:
        print('Error: Argument --model [model] to decide the applied model')
        return
    elif args.model not in ['CNN', 'DNN']:
        print('Error: Selected model %s does not exist'%args.model)
        return

    model = load_model(args.model_path)
    if args.model == 'CNN':
        test_x = test_x.reshape(test_x.shape[0],48,48,1)

    predict(model, test_x, args.output_path)

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, dest='model',
                        help='Which model to apply: CNN / DNN')
    parser.add_argument('-t', '--test_data_path', type=str,
                        default='data/test.csv', dest='test_path',
                        help='Path to testing data')
    parser.add_argument('-l', '--model_path', type=str,
                        default='model.hdf5', dest='model_path',
                        help='Path to load the model')
    parser.add_argument('-o', '--output_path', type=str, dest='output_path',
                        help='Path to save the prediction result')
    main(parser.parse_args())
