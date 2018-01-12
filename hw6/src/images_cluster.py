import numpy as np
from sklearn.cluster import KMeans

import keras.backend as K
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

# GPU setting
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
set_session(sess)

def load_images(images_path):
    images = np.load(images_path)
    images.astype(np.float32)
    images = images/255.
    mean = np.mean(images, axis=1).reshape((images.shape[0],1))
    images = images-mean
    print(images.shape)
    return images

def build_model():
    encode_dim=64
    input_img = Input(shape=(784,))

    encoded = Dense(128, activation='relu')(input_img)
    encoded = Dense(64, activation='relu')(encoded)
    encoded = Dense(encode_dim)(encoded)

    decoded = Dense(64, activation='relu')(encoded)
    decoded = Dense(128, activation='relu')(decoded)
    decoded = Dense(784, activation='tanh')(decoded)

    encoder = Model(inputs=input_img, outputs=encoded)
    autoencoder = Model(inputs=input_img, outputs=decoded)

    encoder.compile(optimizer='adam', loss='mse')
    autoencoder.compile(optimizer='adam', loss='mse')
    return encoder, autoencoder

def train(encoder, autoencoder, images, model_path):
    autoencoder.fit(images, images, epochs=20, batch_size=512, shuffle=True)
    encoder.save(model_path)

def kmeans(encoded_imgs, label_path):
    images_labels = KMeans(n_clusters=2, max_iter=10000, n_jobs=8).fit(encoded_imgs).labels_
    np.save(label_path, images_labels)
    return images_labels
    
def main(args):
    images = load_images(args.images_path)
    encoder, autoencoder = build_model()
    train(encoder, autoencoder, images, args.model_path)
    encoded_imgs = encoder.predict(images)
    images_labels = kmeans(encoded_imgs)
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--images_path', type=str,
                        default='data/image.npy', dest='images_path',
                        help='Path to images data')
    parser.add_argument('-m', '--model_path', type=str,
                        default='model/encoder_64.hdf5', dest='model_path',
                        help='Path to save the encoder model')
    parser.add_argument('-l', '--label_path', type=str,
                        default='images_labels.npy', dest='label_path',
                        help='Path to save the images labels')
    main(parser.parse_args())
