from skimage import io
from skimage import transform
import numpy as np
import os
import argparse

def load_images(images_path):
    images = []
    img_shape = (600,600,3)
    for f in os.listdir(images_path):
        if '.jpg' in f:
            img = io.imread(os.path.join(images_path, f))
            # img = transform.resize(img, img_shape)
            img = img.flatten()
            images.append(img)
    images = np.array(images)
    images = images.T
    print(images.shape)
    return images

def PCA(images):
    mean_face = np.mean(images, axis=1).reshape((images.shape[0],1))
    U, s, V = np.linalg.svd(images - mean_face, full_matrices=False)
    # print(U.shape, s.shape, V.shape)
    return mean_face, U

def reconstruct(img, mean_face, U, img_shape, K=4):
    face = img.reshape((img.shape[0], 1)) - mean_face
    eigenfaces = U[:, :K]
    weights = np.dot(face.T, eigenfaces)
    reconstruct_img = np.dot(weights, eigenfaces.T)
    # print(face.shape, eigenfaces.shape, weights.shape, reconstruct_img.shape)
    M = reconstruct_img.flatten() + mean_face.flatten()
    M -= np.min(M)
    M /= np.max(M)
    M = (M * 255).astype(np.uint8)
    M = np.reshape(M, img_shape)
    return M

def main(args):
    images = load_images(args.images_path)
    mean_face, U = PCA(images)

    img = io.imread(os.path.join(args.images_path, args.target_file))
    img_shape = img.shape
    print(img_shape)
    img = img.flatten()
    M = reconstruct(img, mean_face, U, img_shape, 4)
    io.imsave('reconstruction.jpg', M)
    reconstruct(img, mean_face, U, K=4)

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--images_path', type=str,
                        default='images', dest='images_path',
                        help='Path to images')
    parser.add_argument('-t', '--target_file', type=str,
                        default='0.jpg', dest='target_file',
                        help='Path to load the target_file')
    main(parser.parse_args())
