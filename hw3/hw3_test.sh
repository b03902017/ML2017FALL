#/usr/bin bash

wget https://www.dropbox.com/s/whl5prvjo313vl8/2_weights.39-0.76.hdf5?dl=1 -O model.hdf5
CUDA_VISIBLE_DEVICES='' python3 src/test.py -m CNN -t $1 -o $2
