import argparse
import csv
import os
import numpy as np
# import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import LeakyReLU
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils.vis_utils import plot_model

# GPU setting
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
set_session(sess)

# Global param setting
valid_num = 0

def load_data(train_path):
    train_x = []
    train_y = []
    with open(train_path) as f_train:
        for row_n, row in enumerate(csv.reader(f_train)):
            if row_n != 0: # row0 contains no data
                train_y.append(row[0])
                train_x.append(row[1].split(' '))
    train_x = np.array(train_x, dtype=np.float32)
    train_x = train_x/255

    train_y = np.array(train_y, dtype=np.uint8)
    train_y = np_utils.to_categorical(train_y, 7)


    return (train_x[:valid_num], train_y[:valid_num]), (train_x[valid_num:], train_y[valid_num:])

def evaluate(model, train_x, train_y, valid_x, valid_y):
    result_train = model.evaluate(train_x, train_y)
    print("\nTrain CNN Acc:",result_train[1])
    result_valid = model.evaluate(valid_x, valid_y)
    print("\nValid CNN Acc:",result_valid[1])

def add_cnn(model, filter_num, drop_rate, filter_shape=(3,3), is_input=False, pooling=True):
    if is_input:
        model.add(Conv2D(filter_num, filter_shape, padding='same', input_shape=(48,48,1)))
    else:
        model.add(Conv2D(filter_num, filter_shape, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.05))

    if pooling:
        model.add(MaxPooling2D((2,2)))
    model.add(Dropout(drop_rate))

def add_fc(model, units, drop_rate=0, with_drop=True, is_input=False):
    if is_input:
        model.add(Dense(units=units, input_dim=48*48))
    else:
        model.add(Dense(units=units))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.05))
    if with_drop:
        model.add(Dropout(drop_rate))

def CNN():
    model = Sequential()
    add_cnn(model, 64, 0.2, filter_shape=(5,5), is_input=True, pooling=False)
    add_cnn(model, 64, 0.2)
    add_cnn(model, 128, 0.3)
    add_cnn(model, 256, 0.4)
    add_cnn(model, 512, 0.4)

    model.add(Flatten())
    add_fc(model, 512, 0.5)
    add_fc(model, 256, 0.5)
    model.add(Dense(units=7, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def DNN():
    model = Sequential()
    add_fc(model, 1024, 0.3, is_input=True)
    add_fc(model, 1024, 0.4)
    add_fc(model, 512, 0.4)
    add_fc(model, 256, 0.5)
    add_fc(model, 128, 0.5)
    model.add(Dense(units=7, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train(model, train_x, train_y, valid_x, valid_y, model_dir, generate=True):
    BATCH_SIZE = 128
    STEPS_PER_EPOCH = 500
    PATIENCE = 5
    EPOCHS = 40
    callbacks = [
        # EarlyStopping(monitor='val_acc', patience=PATIENCE, verbose=0),
        ModelCheckpoint(model_dir+'/weights.{epoch:02d}-{val_acc:.2f}.hdf5',\
                monitor='val_acc', save_best_only=True, period=5, verbose=0),
    ]
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if generate:
        datagen = ImageDataGenerator(
                rotation_range=20, horizontal_flip=True,
                zoom_range=0.2, shear_range=0.1, fill_mode='nearest')
        history = model.fit_generator(datagen.flow(train_x, train_y, batch_size=BATCH_SIZE),\
                            steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS,\
                            validation_data=(valid_x, valid_y), callbacks=callbacks)
    else:
        history = model.fit(train_x, train_y, batch_size=BATCH_SIZE,\
                            epochs=EPOCHS,\
                            validation_data=(valid_x, valid_y), callbacks=callbacks)
    return history

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
    (valid_x, valid_y), (train_x, train_y) = \
    load_data(args.train_path)
    print('train_x.shape:'+str(train_x.shape))
    print('valid_x.shape:'+str(valid_x.shape))

    if args.model is None:
        print('Error: Argument --model [model] to decide the applied model')
        return
    elif args.model not in ['CNN', 'DNN']:
        print('Error: Selected model %s does not exist'%args.model)
        return

    if args.model == 'CNN':
        train_x = train_x.reshape(train_x.shape[0],48,48,1)
        valid_x = valid_x.reshape(valid_x.shape[0],48,48,1)
        model = CNN()
        history = train(model, train_x, train_y, valid_x, valid_y, \
                        args.model_dir, generate=True)
    elif args.model == 'DNN':
        model = DNN()
        history = train(model, train_x, train_y, valid_x, valid_y, \
                        args.model_dir, generate=False)

    evaluate(model, train_x, train_y, valid_x, valid_y)
    # plot_history(history)

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, dest='model',
                        help='Which model to apply: CNN / DNN')
    parser.add_argument('-t', '--train_data_path', type=str,
                        default='data/train.csv', dest='train_path',
                        help='Path to training data')
    parser.add_argument('-d', '--model_dir', type=str,
                        default='model', dest='model_dir',
                        help='Dir to save the model')
    parser.add_argument('-v', '--valid_num', type=int, default=500,
                         dest='valid_num', help='Size of validation data')
    main(parser.parse_args())
