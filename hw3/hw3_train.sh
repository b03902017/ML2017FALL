#/usr/bin bash

CUDA_VISIBLE_DEVICES='0' python3 src/train.py -m CNN -t $1
