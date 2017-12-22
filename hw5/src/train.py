# Implement Matrix Factorization by keras

import argparse
import numpy as np
import os
# import matplotlib.pyplot as plt

import keras.backend as K
from keras.models import Sequential
from keras.layers import Input, Add, Dot, Flatten
from keras.layers import Embedding
from keras.optimizers import Adam
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras import regularizers

# GPU setting
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
set_session(sess)

# Global param setting
valid_num = 0

def load_data(train_path):
    users = []
    movies = []
    ratings = []
    with open(train_path) as f:
        for n, line in enumerate(f):
            if n == 0: # line 0 contains no data
                continue
            line = line.strip().split(',')
            user, movie, rating = int(line[1]), int(line[2]), int(line[3])
            users.append(int(line[1]))
            movies.append(int(line[2]))
            ratings.append(int(line[3]))

    users = np.asarray(users, dtype=np.int16)
    movies = np.asarray(movies, dtype=np.int16)
    ratings = np.asarray(ratings, dtype=np.int16)

    rand_perm = np.random.permutation(users.shape[0])
    train_index, valid_index = rand_perm[valid_num:], rand_perm[:valid_num]
    val_users, users = users[valid_index], users[train_index]
    val_movies, movies = movies[valid_index], movies[train_index]
    val_ratings, ratings = ratings[valid_index], ratings[train_index]
    print(val_users.shape, users.shape)
    return (users, movies, ratings), (val_users, val_movies, val_ratings)

def rmse(y_true, y_pred, use_np=False):
    if use_np:
        return np.sqrt(((y_pred - y_true)**2).mean())
    else:
        return K.sqrt(K.mean(K.pow(y_pred - y_true, 2)))

def MF():
    embed_dim = 64
    user_num, movie_num = 6040, 3952
    user_input = Input(shape=(1,))
    movie_input = Input(shape=(1,))
    user_embed = Embedding(user_num+1, embed_dim, embeddings_initializer='lecun_uniform', embeddings_regularizer=regularizers.l2(1e-5))(user_input)
    movie_embed = Embedding(movie_num+1, embed_dim, embeddings_initializer='lecun_uniform', embeddings_regularizer=regularizers.l2(1e-5))(movie_input)
    user_embed = Flatten()(user_embed)
    movie_embed = Flatten()(movie_embed)

    user_bias = Flatten()(Embedding(user_num+1, 1, embeddings_initializer='zero')(user_input))
    movie_bias = Flatten()(Embedding(movie_num+1, 1, embeddings_initializer='zero')(movie_input))

    dot = Dot(axes=-1)([user_embed, movie_embed])
    out = Add()([dot, user_bias, movie_bias])

    model = Model(inputs=[user_input, movie_input], outputs=out)
    model.compile(loss='mse', optimizer=Adam(lr=5e-4), metrics=[rmse])
    return model

def train(model, users, movies, ratings, val_users, val_movies, val_ratings):
    BATCH_SIZE = 512
    EPOCHS = 20
    callbacks = [
        ModelCheckpoint('model/MF_384d.{epoch:02d}-{val_rmse:.3f}.hdf5', monitor='val_rmse',\
                        save_best_only=True, mode='min', period=1, verbose=0),]
    history = model.fit([users, movies], ratings, batch_size=BATCH_SIZE, epochs=EPOCHS,\
                        validation_data=([val_users, val_movies], val_ratings), callbacks=callbacks)

    return history

def evaluate(model, val_users, val_movies, val_ratings):
    result = model.predict([val_users, val_movies])
    result[result>5] = 5
    result[result<1] = 1
    result = result.reshape((result.shape[0],))
    print(rmse(val_ratings, result, True))

# def plot_history(history):
#     plt.plot(history.history['rmse'])
#     plt.plot(history.history['val_rmse'])
#     plt.title('Training Procedure')
#     plt.ylabel('RMSE')
#     plt.xlabel('Epoch')
#     plt.legend(['train', 'valid'], loc='upper left')
#     plt.show()

def main(args):
    global valid_num
    valid_num = args.valid_num
    (users, movies, ratings), (val_users, val_movies, val_ratings) = \
    load_data(args.train_path)

    model = MF()
    history = train(model, users, movies, ratings, val_users, val_movies, val_ratings)
    evaluate(model, train_x, train_y, valid_x, valid_y)
    # plot_history(history)

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train_path', type=str,
                        default='data/train.csv', dest='train_path',
                        help='Path to train data')
    parser.add_argument('-v', '--valid_num', type=int, default=180000,
                         dest='valid_num', help='Size of validation data')
    main(parser.parse_args())
