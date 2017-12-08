#/usr/bin bash
w2v_file1="model/punc_word2vec_gensim_sg"
if [ ! -f "$w2v_file1" ]
then
  wget https://www.dropbox.com/s/iy1bcvglq2dhgad/punc_word2vec_gensim_sg?dl=1 -O model/punc_word2vec_gensim_sg
  wget https://www.dropbox.com/s/iaeqvvu501fmm8v/punc_word2vec_gensim_sg.syn1neg.npy?dl=1 -O model/punc_word2vec_gensim_sg.syn1neg.npy
  wget https://www.dropbox.com/s/vjl6xpnc3shfuat/punc_word2vec_gensim_sg.wv.syn0.npy?dl=1 -O model/punc_word2vec_gensim_sg.wv.syn0.npy
fi
python3 src/test.py -t $1 -o $2
