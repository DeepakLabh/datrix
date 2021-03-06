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


def fcnn(input_dim, input_str_dim):
    input_layer1 = Input(shape = (input_dim,))
    input_layers = [input_layer1]
    input_layer2 = Input(shape = (input_str_dim,))
    input_layers.append(input_layer2)

    d1 = Dense(200)(input_layer1)
    #d1 = Dropout(0.1)(d1)
    d1 = Reshape((200,1))(d1)
    d1 = LSTM(30, return_sequences=False, go_backwards = False, activation='tanh', inner_activation='hard_sigmoid')(d1)
    d1 = BatchNormalization()(d1)
    d2 = Activation('relu')(d1)

    s1 = Dense(5)(input_layer2)
    s1 = Dropout(0.1)(s1)
    s1 = BatchNormalization()(s1)
    s2 = Activation('relu')(s1)

    d2 = merge([d2, s2], mode= 'concat', concat_axis= -1)
    d2 = Activation('relu')(d2)

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
