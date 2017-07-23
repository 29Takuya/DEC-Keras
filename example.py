from keras_dec import DeepEmbeddingClustering
from keras.datasets import mnist
import numpy as np


def get_mnist():
    np.random.seed(1234) # set seed for deterministic ordering
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_all = np.concatenate((x_train, x_test), axis = 0)
    Y = np.concatenate((y_train, y_test), axis = 0)
    X = x_all.reshape(-1,x_all.shape[1]*x_all.shape[2])
    
    p = np.random.permutation(X.shape[0])
    X = X[p].astype(np.float32)*0.02
    Y = Y[p]
    return X, Y

#X, Y  = get_mnist()

X = np.load('/home-nfs/shimada/work/english_mfcc.npz')
X = X['arr_0']
Y = None

c = DeepEmbeddingClustering(n_clusters=100, input_dim=39)
c.initialize(X, finetune_iters=100000, layerwise_pretrain_iters=50000)
c.cluster(X, y=Y)
