from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, LSTM, Lambda, Merge, Reshape, GRU, Input, merge, Flatten
import keras
#from keras.optimizers import SGD
from keras.optimizers import SGD, RMSprop
#from keras.regularizers import l2, activity_l2, l1
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras import optimizers


def r2_keras(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )


def fcnn(input_dim):
    input_layers = Input(shape = (input_dim,))
    d1 = Dense(150)(input_layers)
    d1 = BatchNormalization()(d1)
    d2 = Activation('relu')(d1)
    #d2 = Dense(100)(d2)
    #d2 = BatchNormalization()(d2)
    #d2 = Activation('relu')(d2)
    #d2 = Dense(50)(d2)
    #d2 = BatchNormalization()(d2)
    #d2 = Activation('relu')(d2)
    #d2 = Dense(25)(d2)
    d2 = Dense(1)(d2)
    out_layer = Activation('relu')(d2)
    rmsprop = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    model = Model(input = input_layers, output = out_layer)
    model.compile(loss = 'mae', optimizer='adam', metrics = [r2_keras])
    return model
